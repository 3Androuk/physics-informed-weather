"""Tests for the streaming (low-RAM) per-year download path.

`_download_year` must write batches STRAIGHT into an on-disk memmap. Buffering
a whole year costs ~19 GiB at 20 channels, which exceeds the 4 GiB cap on a
BriCS login node and would force the download onto a billed GPU node.

xarray is not needed: `_open_da` / `_year_sub` are patched with array-backed
fakes, so these run anywhere the rest of the suite runs.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from data import download_era5 as dl


class _FakeSel:
    def __init__(self, arr):
        self.values = arr


class _FakeSub:
    """Minimal stand-in for a sliced xarray DataArray."""

    def __init__(self, arr, fail_slices=None, nan_slices=None):
        self._arr = arr
        self._fail = dict(fail_slices or {})   # (start, stop) -> remaining failures
        self._nan = set(nan_slices or ())      # (start, stop) yielding NaN
        self.reads = []
        T, C, H, W = arr.shape
        self.sizes = {"time": T, "channel": C, "latitude": H, "longitude": W}

    def isel(self, time):
        key = (time.start, time.stop)
        self.reads.append(key)
        if self._fail.get(key, 0) > 0:
            self._fail[key] -= 1
            raise OSError("simulated GCS stall")
        chunk = self._arr[time].copy()
        if key in self._nan:
            chunk[0, 0, 0, 0] = np.nan
        return _FakeSel(chunk)


class _FakeDA:
    def __init__(self, lat, lon):
        self._c = {"latitude": _FakeSel(lat), "longitude": _FakeSel(lon)}

    def __getitem__(self, k):
        return self._c[k]


def _patched(arr, sub):
    """Patch the two network entry points to serve `sub` / fake coords."""
    lat = np.linspace(-60, 60, arr.shape[2]).astype("float32")
    lon = np.linspace(0, 359, arr.shape[3]).astype("float32")
    return (mock.patch.object(dl, "_open_da", lambda *a, **k: _FakeDA(lat, lon)),
            mock.patch.object(dl, "_year_sub", lambda *a, **k: sub),
            lat, lon)


class DownloadStreamTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # (T, C, H, W) — small, but the same rank/dtype as the real thing.
        self.arr = rng.standard_normal((7, 3, 5, 6)).astype("float32")
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out = Path(self.tmpdir.name) / "train_2007.npy.tmp"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self, sub, batch=3, max_retries=3):
        p_open, p_year, lat, lon = _patched(self.arr, sub)
        with p_open, p_year:
            return dl._download_year({}, [-60, 60], 2007, 1, batch, 0, 8,
                                     max_retries, self.out)

    def test_streams_exact_contents(self):
        sub = _FakeSub(self.arr)
        shape, lat, lon = self._run(sub, batch=3)

        self.assertEqual(shape, self.arr.shape)
        written = np.load(self.out)
        np.testing.assert_array_equal(written, self.arr)
        self.assertEqual(len(lat), self.arr.shape[2])
        self.assertEqual(len(lon), self.arr.shape[3])

    def test_batches_cover_the_year_without_overlap(self):
        sub = _FakeSub(self.arr)
        self._run(sub, batch=3)
        # 7 fields at batch 3 -> [0:3], [3:6], [6:7]
        self.assertEqual(sub.reads, [(0, 3), (3, 6), (6, 7)])

    def test_never_allocates_a_full_year_in_ram(self):
        """The guard that matters: no in-RAM array the size of the year."""
        sub = _FakeSub(self.arr)
        real_empty = np.empty
        big = []

        def spy(shape, *a, **k):
            if isinstance(shape, tuple) and len(shape) == 4:
                big.append(shape)
            return real_empty(shape, *a, **k)

        with mock.patch.object(np, "empty", spy):
            self._run(sub, batch=3)
        self.assertEqual(big, [], f"full-year buffer reintroduced: {big}")

    def test_batch_size_bounds_peak_memory(self):
        """Each read is at most `batch` fields, whatever the year length."""
        sub = _FakeSub(self.arr)
        self._run(sub, batch=2)
        self.assertTrue(all(stop - start <= 2 for start, stop in sub.reads),
                        sub.reads)

    def test_retries_then_succeeds(self):
        sub = _FakeSub(self.arr, fail_slices={(3, 6): 2})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            shape, _, _ = self._run(sub, batch=3, max_retries=3)
        self.assertEqual(shape, self.arr.shape)
        np.testing.assert_array_equal(np.load(self.out), self.arr)

    def test_gives_up_after_max_retries(self):
        sub = _FakeSub(self.arr, fail_slices={(0, 3): 99})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            with self.assertRaises(RuntimeError):
                self._run(sub, batch=3, max_retries=2)

    def test_nan_fails_fast_and_is_not_retried(self):
        """A NaN field is bad data, not a stalled connection."""
        sub = _FakeSub(self.arr, nan_slices={(3, 6)})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            with self.assertRaises(dl.BadFieldData):
                self._run(sub, batch=3, max_retries=4)
        # [0:3] once, [3:6] once — no retry storm on the bad batch.
        self.assertEqual(sub.reads, [(0, 3), (3, 6)])


if __name__ == "__main__":
    unittest.main()
