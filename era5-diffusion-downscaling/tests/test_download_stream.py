"""Tests for the low-memory per-year download path.

Two constraints, both measured on a BriCS login node against its 4 GiB cgroup
cap (`MemoryMax=4294967296`):

1. `_download_year` must write batches STRAIGHT into an on-disk memmap.
   Buffering a whole year costs ~19 GiB at 20 channels.
2. It must read ONE CHANNEL AT A TIME. Concatenating 20 variables and calling
   `.values` once makes dask materialize all 20 variables' chunks together,
   which peaked at 3.46 GiB and was OOM-killed (exit 137) even at --batch 4.

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


class _FakeChannel:
    """Stand-in for one variable's DataArray: (time, latitude, longitude)."""

    def __init__(self, arr, ci, reads, fail=None, nan=None):
        self._arr = arr            # full (T, C, H, W)
        self._ci = ci
        self._reads = reads        # shared log of (channel, start, stop)
        self._fail = fail if fail is not None else {}
        self._nan = nan or set()
        T, _, H, W = arr.shape
        self.sizes = {"time": T, "latitude": H, "longitude": W}

    def isel(self, time):
        key = (self._ci, time.start, time.stop)
        self._reads.append(key)
        if self._fail.get(key, 0) > 0:
            self._fail[key] -= 1
            raise OSError("simulated GCS stall")
        chunk = self._arr[time, self._ci].copy()   # (t, H, W) — one channel only
        if key in self._nan:
            chunk[0, 0, 0] = np.nan
        return _FakeSel(chunk)


class _Fixture:
    """Serves per-channel fakes through the two patched entry points."""

    def __init__(self, arr, fail=None, nan=None):
        self.arr = arr
        self.reads = []
        self.opens = 0
        self._fail = dict(fail or {})
        self._nan = set(nan or ())
        self.lat = np.linspace(-60, 60, arr.shape[2]).astype("float32")
        self.lon = np.linspace(0, 359, arr.shape[3]).astype("float32")

    def open_da(self, *a, **k):
        self.opens += 1
        das = [_FakeChannel(self.arr, ci, self.reads, self._fail, self._nan)
               for ci in range(self.arr.shape[1])]
        return das, self.lat, self.lon

    def patches(self):
        return (mock.patch.object(dl, "_open_da", self.open_da),
                mock.patch.object(dl, "_year_sub", lambda da, *a, **k: da))


class DownloadStreamTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.arr = rng.standard_normal((7, 3, 5, 6)).astype("float32")  # (T,C,H,W)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out = Path(self.tmpdir.name) / "train_2007.npy.tmp"

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self, fx, batch=3, max_retries=3):
        p_open, p_year = fx.patches()
        with p_open, p_year:
            return dl._download_year({}, [-60, 60], 2007, 1, batch, 0, 8,
                                     max_retries, self.out)

    def test_streams_exact_contents(self):
        fx = _Fixture(self.arr)
        shape, lat, lon = self._run(fx, batch=3)
        self.assertEqual(shape, self.arr.shape)
        np.testing.assert_array_equal(np.load(self.out), self.arr)
        self.assertEqual(len(lat), self.arr.shape[2])
        self.assertEqual(len(lon), self.arr.shape[3])

    def test_reads_one_channel_at_a_time(self):
        """The OOM fix: never a read spanning all channels at once."""
        fx = _Fixture(self.arr)
        captured = []
        real = _FakeChannel.isel

        def spy(self_, time):
            sel = real(self_, time)
            captured.append(sel.values.shape)
            return sel

        with mock.patch.object(_FakeChannel, "isel", spy):
            self._run(fx, batch=3)
        # every read is (t, H, W) — 3-D, one channel — never (t, C, H, W)
        self.assertTrue(all(len(s) == 3 for s in captured), captured)
        self.assertTrue(all(s[1:] == self.arr.shape[2:] for s in captured), captured)

    def test_every_channel_and_timestep_covered_exactly_once(self):
        fx = _Fixture(self.arr)
        self._run(fx, batch=3)
        expected = [(ci, s, e) for s, e in [(0, 3), (3, 6), (6, 7)]
                    for ci in range(self.arr.shape[1])]
        # batch-major, channel-minor
        self.assertEqual(sorted(fx.reads), sorted(expected))
        self.assertEqual(len(fx.reads), len(set(fx.reads)))

    def test_never_allocates_a_full_year_in_ram(self):
        fx = _Fixture(self.arr)
        real_empty = np.empty
        big = []

        def spy(shape, *a, **k):
            if isinstance(shape, tuple) and len(shape) == 4:
                big.append(shape)
            return real_empty(shape, *a, **k)

        with mock.patch.object(np, "empty", spy):
            self._run(fx, batch=3)
        self.assertEqual(big, [], f"full-year buffer reintroduced: {big}")

    def test_batch_size_bounds_peak_memory(self):
        fx = _Fixture(self.arr)
        self._run(fx, batch=2)
        self.assertTrue(all(stop - start <= 2 for _, start, stop in fx.reads),
                        fx.reads)

    def test_retries_then_succeeds(self):
        fx = _Fixture(self.arr, fail={(1, 3, 6): 2})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            shape, _, _ = self._run(fx, batch=3, max_retries=3)
        self.assertEqual(shape, self.arr.shape)
        np.testing.assert_array_equal(np.load(self.out), self.arr)
        self.assertGreater(fx.opens, 1, "a retry must reopen a fresh session")

    def test_gives_up_after_max_retries(self):
        fx = _Fixture(self.arr, fail={(0, 0, 3): 99})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            with self.assertRaises(RuntimeError):
                self._run(fx, batch=3, max_retries=2)

    def test_nan_fails_fast_and_is_not_retried(self):
        """A NaN field is bad data, not a stalled connection."""
        fx = _Fixture(self.arr, nan={(1, 0, 3)})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            with self.assertRaises(dl.BadFieldData):
                self._run(fx, batch=3, max_retries=4)
        # ch0 then ch1 of the first batch, and no retry storm on the bad one
        self.assertEqual(fx.reads, [(0, 0, 3), (1, 0, 3)])


if __name__ == "__main__":
    unittest.main()
