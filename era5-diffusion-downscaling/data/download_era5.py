"""Stream ERA5 Z500 from the WeatherBench 2 public GCS store — resumable.

Downloads geopotential @ 500 hPa at 0.25 deg (721 x 1440), cropped to a
mid-latitude band, for the configured train/test year ranges, and caches each
split as a float32 .npy of shape (T, H, W). No credentials required.

Robustness (to survive flaky links to GCS):
  * Fetches ONE YEAR AT A TIME and writes each to datasets/raw/_years/ as it
    completes, so a stalled connection costs at most one year — rerun and it
    skips every year already on disk (resume).
  * Per-request read timeout (--timeout) so a stalled read raises instead of
    hanging forever; each year is retried up to --max-retries times with backoff.
  * Per-year arrays are merged into train.npy / test.npy at the end via a
    memmap, so the merge never holds more than one year in RAM.

Note: striding (data.time_stride) is applied within each year, so the exact
timesteps chosen differ negligibly from striding the whole range at once; this
does not affect the downstream random-crop patches.

Run:
    python -m data.download_era5 --config config/default.yaml
    python -m data.download_era5 --config config/default.yaml --timeout 90 --max-retries 8
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config  # noqa: E402


def _open_da(dcfg, lat_range, timeout):
    """Open the WB2 zarr and return the Z500 DataArray cropped to lat_range.

    A fresh handle (and thus a fresh network session) is opened on every call, so
    a retry after a stall is never stuck reusing a broken connection.
    """
    import xarray as xr  # local import so non-download code has no hard dep

    storage = dict(token=dcfg["gcs_token"])
    if timeout:
        storage["requests_timeout"] = timeout
    try:
        ds = xr.open_zarr(dcfg["era5_zarr"], chunks={"time": 50}, storage_options=storage)
    except TypeError:
        # gcsfs build without requests_timeout: fall back (no per-request timeout).
        storage.pop("requests_timeout", None)
        print("  (warning: gcsfs ignored requests_timeout — stalls may not time out)")
        ds = xr.open_zarr(dcfg["era5_zarr"], chunks={"time": 50}, storage_options=storage)

    da = ds[dcfg["variable"]].sel(level=dcfg["level"])
    da = da.transpose("time", "latitude", "longitude")
    lat = da["latitude"].values
    lo, hi = lat_range
    da = da.isel(latitude=np.where((lat >= lo) & (lat <= hi))[0])
    return da


def _download_year(dcfg, lat_range, year, stride, timeout):
    """Fetch a single year -> (arr (T, H, W) float32, lat, lon)."""
    da = _open_da(dcfg, lat_range, timeout)
    sub = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if stride and stride > 1:
        sub = sub.isel(time=slice(None, None, stride))
    n = int(sub.sizes["time"])
    print(f"  computing {n} fields for {year} ...", flush=True)
    arr = sub.compute().values.astype(np.float32)
    return arr, da["latitude"].values, da["longitude"].values


def _download_year_retry(dcfg, lat_range, year, stride, timeout, max_retries):
    """_download_year with bounded retries + linear backoff on any failure."""
    for attempt in range(1, max_retries + 1):
        try:
            return _download_year(dcfg, lat_range, year, stride, timeout)
        except Exception as e:  # noqa: BLE001 - surface any GCS/dask error and retry
            if attempt == max_retries:
                raise RuntimeError(
                    f"Year {year} failed after {max_retries} attempts: "
                    f"{type(e).__name__}: {e}"
                ) from e
            wait = min(60, 5 * attempt)
            print(f"  [retry {attempt}/{max_retries}] {year} failed "
                  f"({type(e).__name__}: {e}); retrying in {wait}s ...", flush=True)
            time.sleep(wait)


def _merge(cache_dir, split, years, out_path):
    """Concatenate the per-year caches into out_path (.npy) via a memmap.

    Never holds more than one year in RAM; writes to a .tmp then renames so an
    interrupted merge can't leave a half-written train.npy.
    """
    from numpy.lib.format import open_memmap

    files = [cache_dir / f"{split}_{y}.npy" for y in years]
    shapes = [np.load(f, mmap_mode="r").shape for f in files]
    total = sum(s[0] for s in shapes)
    h, w = shapes[0][1], shapes[0][2]

    tmp = out_path.with_suffix(".npy.tmp")
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(total, h, w))
    i = 0
    for f, s in zip(files, shapes):
        a = np.load(f, mmap_mode="r")
        out[i:i + s[0]] = a[:]
        i += s[0]
        del a
    out.flush()
    del out
    tmp.replace(out_path)
    return (total, h, w)


def _save_coords(dcfg, lat_range, timeout, max_retries, coords_path):
    """Save lat/lon once (with retries) — used only if no year populated it."""
    for attempt in range(1, max_retries + 1):
        try:
            da = _open_da(dcfg, lat_range, timeout)
            np.savez(coords_path, lat=da["latitude"].values, lon=da["longitude"].values)
            return
        except Exception:  # noqa: BLE001
            if attempt == max_retries:
                raise
            time.sleep(min(60, 5 * attempt))


def main():
    ap = argparse.ArgumentParser(description="Download ERA5 Z500 from WB2 GCS (resumable).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-request GCS read timeout in seconds (0 disables).")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="retries per year before giving up.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the per-year datasets/raw/_years/ files after merging.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    stride = dcfg.get("time_stride", 1)
    lat_range = cfg["patches"]["lat_range"]

    raw_dir = ensure_dir(cfg["paths"]["raw_dir"])
    cache_dir = ensure_dir(Path(raw_dir) / "_years")
    coords_path = Path(raw_dir) / "coords.npz"

    print(f"Opening {dcfg['era5_zarr']} (per-year, timeout={args.timeout}s, "
          f"retries={args.max_retries})", flush=True)

    splits = {
        "train": list(range(dcfg["train_years"][0], dcfg["train_years"][1] + 1)),
        "test": list(range(dcfg["test_years"][0], dcfg["test_years"][1] + 1)),
    }

    # ── Per-year download (resumable) ─────────────────────────────────────
    for split, years in splits.items():
        for year in years:
            ypath = cache_dir / f"{split}_{year}.npy"
            if ypath.exists():
                print(f"[skip] {split} {year} already cached", flush=True)
                continue
            arr, lat, lon = _download_year_retry(
                dcfg, lat_range, year, stride, args.timeout, args.max_retries
            )
            if not np.isfinite(arr).all():
                raise ValueError(f"{split} {year} contains NaN/Inf.")
            tmp = ypath.with_suffix(".npy.tmp")
            np.save(tmp, arr)
            tmp.replace(ypath)
            print(f"[done] {split} {year}: {arr.shape} -> {ypath.name}", flush=True)
            if not coords_path.exists():
                np.savez(coords_path, lat=lat, lon=lon)
            del arr

    if not coords_path.exists():  # e.g. resumed run where every year was cached
        _save_coords(dcfg, lat_range, args.timeout, args.max_retries, coords_path)

    # ── Merge per-year caches into train.npy / test.npy ───────────────────
    merged = {}
    for split, years in splits.items():
        merged[split] = _merge(cache_dir, split, years, Path(raw_dir) / f"{split}.npy")
        print(f"Merged {split}: {merged[split]} -> {raw_dir}/{split}.npy", flush=True)

    # ── Cleanup per-year caches ───────────────────────────────────────────
    if not args.keep_cache:
        for f in cache_dir.glob("*.npy"):
            f.unlink()
        try:
            cache_dir.rmdir()
        except OSError:
            pass

    h, w = merged["train"][1], merged["train"][2]
    print(f"Done. grid {h}x{w} -> {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
