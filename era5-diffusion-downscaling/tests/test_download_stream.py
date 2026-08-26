"""Tests for the low-memory per-year download path.

Three constraints, all measured on a BriCS login node against its 4 GiB cgroup
cap (`MemoryMax=4294967296`):

1. `_download_year` writes batches STRAIGHT into an on-disk memmap. Buffering a
   whole year costs ~19 GiB at 20 channels.
2. It reads ONE VARIABLE GROUP AT A TIME. Concatenating 20 variables and
   calling `.values` once made dask materialize every variable's chunks
   together: peak 3.46 GiB, OOM-killed (exit 137) even at --batch 4.
3. All levels of a variable come from ONE read. WB2 chunks pressure-level
   variables as (1, 13, 721, 1440) — every level in a single 51.5 MiB chunk —
   so z500/z700/z850 as separate channel reads refetched that same chunk three
   times, ~772 MiB pulled per timestep to keep ~59 MiB.

xarray is not needed: `_open_da` / `_year_sub` are patched with array-backed
fakes, so these run anywhere the rest of the suite runs.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from data import download_era5 as dl

# Channel layout under test: ch0 is a surface variable, ch1/ch2 are two levels
# of one pressure variable (so they must arrive in a single grouped read).
SURFACE_CH = 0
LEVEL_CHS = (1, 2)


class _FakeSel:
    def __init__(self, arr):
        self.values = arr


class _FakeGroup:
    """One variable: (time, latitude, longitude), or (time, level, lat, lon)."""

    def __init__(self, arr, gi, chans, reads, fail, nan):
        self._arr = arr                # full (T, C, H, W)
        self._gi = gi
        self._chans = chans            # channel indices this variable supplies
        self._reads = reads
        self._fail = fail
        self._nan = nan
        T, _, H, W = arr.shape
        self.sizes = {"time": T, "latitude": H, "longitude": W}

    def isel(self, time):
        key = (self._gi, time.start, time.stop)
        self._reads.append(key)
        if self._fail.get(key, 0) > 0:
            self._fail[key] -= 1
            raise OSError("simulated GCS stall")
        if len(self._chans) == 1:                       # surface: (t, H, W)
            chunk = self._arr[time, self._chans[0]].copy()
        else:                                           # levels: (t, L, H, W)
            chunk = np.stack([self._arr[time, c] for c in self._chans], axis=1)
        if key in self._nan:
            chunk.reshape(-1)[0] = np.nan
        return _FakeSel(chunk)


class _Fixture:
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
        groups = [
            (_FakeGroup(self.arr, 0, [SURFACE_CH], self.reads, self._fail, self._nan),
             [(SURFACE_CH, None)]),
            (_FakeGroup(self.arr, 1, list(LEVEL_CHS), self.reads, self._fail, self._nan),
             [(ci, pos) for pos, ci in enumerate(LEVEL_CHS)]),
        ]
        return groups, self.lat, self.lon

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
        """Every channel lands in its own slot, levels scattered correctly."""
        fx = _Fixture(self.arr)
        shape, lat, lon = self._run(fx, batch=3)
        self.assertEqual(shape, self.arr.shape)
        np.testing.assert_array_equal(np.load(self.out), self.arr)
        self.assertEqual(len(lat), self.arr.shape[2])
        self.assertEqual(len(lon), self.arr.shape[3])

    def test_one_read_per_variable_not_per_channel(self):
        """The chunk-refetch fix: 2 variables => 2 reads per batch, not 3."""
        fx = _Fixture(self.arr)
        self._run(fx, batch=3)
        per_batch = {}
        for gi, s, e in fx.reads:
            per_batch.setdefault((s, e), []).append(gi)
        for span, gis in per_batch.items():
            self.assertEqual(sorted(gis), [0, 1],
                             f"batch {span} did not read exactly one read per variable")

    def test_never_reads_all_channels_at_once(self):
        """No single read may span the whole channel axis."""
        fx = _Fixture(self.arr)
        captured = []
        real = _FakeGroup.isel

        def spy(self_, time):
            sel = real(self_, time)
            captured.append(sel.values.shape)
            return sel

        with mock.patch.object(_FakeGroup, "isel", spy):
            self._run(fx, batch=3)
        C = self.arr.shape[1]
        for s in captured:
            width = s[1] if len(s) == 4 else 1
            self.assertLess(width, C, f"read spanned all {C} channels: {s}")

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

    def test_every_timestep_covered_exactly_once(self):
        fx = _Fixture(self.arr)
        self._run(fx, batch=3)
        expected = [(gi, s, e) for s, e in [(0, 3), (3, 6), (6, 7)] for gi in (0, 1)]
        self.assertEqual(sorted(fx.reads), sorted(expected))
        self.assertEqual(len(fx.reads), len(set(fx.reads)))

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
        fx = _Fixture(self.arr, nan={(1, 0, 3)})
        with mock.patch.object(dl.time, "sleep", lambda _s: None):
            with self.assertRaises(dl.BadFieldData):
                self._run(fx, batch=3, max_retries=4)
        self.assertEqual(fx.reads, [(0, 0, 3), (1, 0, 3)])


if __name__ == "__main__":
    unittest.main()
