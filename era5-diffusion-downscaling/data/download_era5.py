"""Stream ERA5 Z500 from the WeatherBench 2 public GCS store — resumable.

Downloads geopotential @ 500 hPa at 0.25 deg (721 x 1440), cropped to a
mid-latitude band, for the configured train/test year ranges, and caches each
split as a float32 .npy of shape (T, H, W). No credentials required.

This store is 0.25 deg (1440x721) — ~36x more data per field than the 1.5 deg
store the baselines use — so the download is large. Built to survive a slow /
flaky link to GCS:
  * ONE YEAR AT A TIME, each written to datasets/raw/_years/ as it finishes —
    a rerun skips every year already on disk (resume).
  * Each year is read in SMALL TIME-BATCHES so every HTTP request is small and
    completes well within --timeout; a stalled batch is retried (with a fresh
    connection) up to --max-retries, costing one batch rather than a whole year.
  * Small dask chunks (--chunk-time) keep individual reads tiny — an oversized
    chunk that can't be read before the timeout fires is the usual cause of
    "times out on the first year".
  * Per-year arrays are merged into train.npy / test.npy via a memmap, so the
    merge never holds more than one year in RAM.

Note: striding (data.time_stride) is applied within each year, so the exact
timesteps chosen differ negligibly from striding the whole range at once; this
does not affect the downstream random-crop patches.

Run:
    python -m data.download_era5 --config config/default.yaml
    python -m data.download_era5 --config config/default.yaml --timeout 120 --chunk-time 8 --batch 48
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config  # noqa: E402


def _open_da(dcfg, lat_range, timeout, chunk_time):
    """Open the WB2 zarr and return the Z500 DataArray cropped to lat_range.

    A fresh handle (fresh network session) is opened on every call, so a retry
    after a stall never reuses a broken connection. Small time-chunks keep each
    underlying HTTP read small.
    """
    import xarray as xr  # local import so non-download code has no hard dep

    storage = dict(token=dcfg["gcs_token"])
    if timeout:
        storage["requests_timeout"] = timeout
    try:
        ds = xr.open_zarr(dcfg["era5_zarr"], chunks={"time": chunk_time},
                          storage_options=storage)
    except TypeError:
        storage.pop("requests_timeout", None)
        print("  (warning: gcsfs ignored requests_timeout — stalls may not time out)")
        ds = xr.open_zarr(dcfg["era5_zarr"], chunks={"time": chunk_time},
                          storage_options=storage)

    da = ds[dcfg["variable"]]
    if dcfg.get("level") is not None:  # surface variables (e.g. 2m_temperature) have no level dim
        da = da.sel(level=dcfg["level"])
    da = da.transpose("time", "latitude", "longitude")
    lat = da["latitude"].values
    lo, hi = lat_range
    da = da.isel(latitude=np.where((lat >= lo) & (lat <= hi))[0])
    return da


def _year_sub(da, year, stride):
    """Strided time-slice of a single calendar year."""
    sub = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if stride and stride > 1:
        sub = sub.isel(time=slice(None, None, stride))
    return sub


def _download_year(dcfg, lat_range, year, stride, batch, timeout, chunk_time, max_retries):
    """Fetch one year in small batches -> (arr (T, H, W) float32, lat, lon).

    Each batch of `batch` (strided) fields is read independently and retried on
    failure with a fresh connection, so a stall costs at most one batch.
    """
    da = _open_da(dcfg, lat_range, timeout, chunk_time)
    sub = _year_sub(da, year, stride)
    T = int(sub.sizes["time"])
    H, W = int(sub.sizes["latitude"]), int(sub.sizes["longitude"])
    lat, lon = da["latitude"].values, da["longitude"].values

    out = np.empty((T, H, W), dtype=np.float32)
    n_batches = (T + batch - 1) // batch
    print(f"  {year}: {T} fields in {n_batches} batches of {batch} ...", flush=True)

    start = 0
    while start < T:
        stop = min(start + batch, T)
        for attempt in range(1, max_retries + 1):
            try:
                out[start:stop] = sub.isel(time=slice(start, stop)).values.astype(np.float32)
                break
            except Exception as e:  # noqa: BLE001 - retry any GCS/dask read error
                if attempt == max_retries:
                    raise RuntimeError(
                        f"{year} fields [{start}:{stop}] failed after {max_retries} "
                        f"attempts: {type(e).__name__}: {e}"
                    ) from e
                wait = min(30, 3 * attempt)
                print(f"    [retry {attempt}/{max_retries}] {year} [{start}:{stop}] "
                      f"{type(e).__name__}; reconnecting in {wait}s ...", flush=True)
                time.sleep(wait)
                da = _open_da(dcfg, lat_range, timeout, chunk_time)  # fresh session
                sub = _year_sub(da, year, stride)
        start = stop
        print(f"    {year}: {stop}/{T}", flush=True)

    return out, lat, lon


def _merge(cache_dir, split, years, out_path):
    """Concatenate per-year caches into out_path (.npy) via a memmap (low RAM).

    Writes to a .tmp then renames so an interrupted merge can't leave a
    half-written train.npy.
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


def main():
    ap = argparse.ArgumentParser(description="Download ERA5 Z500 from WB2 GCS (resumable).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-request GCS read timeout in seconds (0 disables).")
    ap.add_argument("--chunk-time", type=int, default=8,
                    help="dask time-chunk size = size of each HTTP read. Smaller = "
                         "more resilient on a slow link.")
    ap.add_argument("--batch", type=int, default=48,
                    help="fields fetched (and retried) per batch within a year.")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="retries per batch before giving up.")
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

    print(f"Opening {dcfg['era5_zarr']}\n  per-year | timeout={args.timeout}s | "
          f"chunk_time={args.chunk_time} | batch={args.batch} | retries={args.max_retries}",
          flush=True)

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
            arr, lat, lon = _download_year(
                dcfg, lat_range, year, stride, args.batch,
                args.timeout, args.chunk_time, args.max_retries,
            )
            if not np.isfinite(arr).all():
                raise ValueError(f"{split} {year} contains NaN/Inf.")
            tmp = ypath.with_suffix(".npy.tmp")
            with open(tmp, "wb") as fh:  # file object: np.save won't append .npy
                np.save(fh, arr)
            tmp.replace(ypath)
            print(f"[done] {split} {year}: {arr.shape} -> {ypath.name}", flush=True)
            if not coords_path.exists():
                np.savez(coords_path, lat=lat, lon=lon)
            del arr

    if not coords_path.exists():  # e.g. resumed run where every year was cached
        da = _open_da(dcfg, lat_range, args.timeout, args.chunk_time)
        np.savez(coords_path, lat=da["latitude"].values, lon=da["longitude"].values)

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
