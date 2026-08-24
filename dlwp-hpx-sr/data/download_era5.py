"""Stream an ERA5 variable from the WeatherBench 2 public GCS store onto HEALPix.

Downloads the configured variable (surface, e.g. 2m_temperature, or a
pressure-level one like geopotential @ data.level) for the train/test year
ranges and remaps every field to the 12 HEALPix faces (config hpx.nside) as
it arrives, so only the face arrays are cached on disk:

    <hpx_dir>/train.npy        (T, 12, nside, nside) float32
    <hpx_dir>/test.npy
    <hpx_dir>/{train,test}_times.npy   (T,) datetime64
    <hpx_dir>/norm_stats.npz   scalar mean/std of the train split (physical units)
    <hpx_dir>/coords.npz       source lat/lon grid (for eval remap-back)

Built to survive a slow / flaky link to GCS (same scheme as the sibling
era5-diffusion-downscaling downloader — and with the default config the SAME
store / variable / level / years / stride as that project, so the two are
directly comparable):
  * ONE YEAR AT A TIME, each written to <hpx_dir>/_years/ as it finishes —
    a rerun skips every year already on disk (resume).
  * Each year is read in SMALL TIME-BATCHES so every HTTP request completes
    well within --timeout; a stalled batch is retried (with a fresh
    connection) up to --max-retries, costing one batch rather than a year.

Run:
    python -m data.download_era5 --config config/default.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpx.remap import LatLonToHPX  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402


def _open_da(dcfg, timeout, chunk_time):
    """Open the WB2 zarr and return the t2m DataArray as (time, lat, lon).

    A fresh handle (fresh network session) is opened on every call, so a retry
    after a stall never reuses a broken connection.
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
    if "level" in da.dims:
        if dcfg.get("level") is None:
            raise ValueError(f"{dcfg['variable']} has a level dim; set "
                             "data.level (e.g. 500 for Z500)")
        da = da.sel(level=dcfg["level"])
    elif dcfg.get("level") is not None:
        raise ValueError(f"data.level set but {dcfg['variable']} has no level dim")
    da = da.transpose("time", "latitude", "longitude")
    lat = da["latitude"].values
    if lat[0] > lat[-1]:  # some stores order latitude north->south
        da = da.isel(latitude=slice(None, None, -1))
    return da


def _year_sub(da, year, stride):
    sub = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if stride and stride > 1:
        sub = sub.isel(time=slice(None, None, stride))
    return sub


def _download_year(dcfg, remap, year, stride, batch, timeout, chunk_time,
                   max_retries):
    """Fetch one year in small batches, remapped -> ((T, 12, F, F), times)."""
    da = _open_da(dcfg, timeout, chunk_time)
    sub = _year_sub(da, year, stride)
    T = int(sub.sizes["time"])
    times = sub["time"].values

    nside = remap.nside
    out = np.empty((T, 12, nside, nside), dtype=np.float32)
    n_batches = (T + batch - 1) // batch
    print(f"  {year}: {T} fields in {n_batches} batches of {batch} ...", flush=True)

    start = 0
    while start < T:
        stop = min(start + batch, T)
        for attempt in range(1, max_retries + 1):
            try:
                fields = sub.isel(time=slice(start, stop)).values.astype(np.float32)
                break
            except Exception as e:  # noqa: BLE001 - retry any GCS/dask read error
                if attempt == max_retries:
                    raise RuntimeError(
                        f"{year} fields [{start}:{stop}] failed after "
                        f"{max_retries} attempts: {type(e).__name__}: {e}"
                    ) from e
                wait = min(30, 3 * attempt)
                print(f"    [retry {attempt}/{max_retries}] {year} [{start}:{stop}] "
                      f"{type(e).__name__}; reconnecting in {wait}s ...", flush=True)
                time.sleep(wait)
                da = _open_da(dcfg, timeout, chunk_time)  # fresh session
                sub = _year_sub(da, year, stride)
        out[start:stop] = remap(fields)
        start = stop
        print(f"    {year}: {stop}/{T}", flush=True)

    return out, times


def _merge(cache_dir, split, years, out_path):
    """Concatenate per-year caches into out_path (.npy) via a memmap (low RAM)."""
    from numpy.lib.format import open_memmap

    files = [cache_dir / f"{split}_{y}.npy" for y in years]
    shapes = [np.load(f, mmap_mode="r").shape for f in files]
    total = sum(s[0] for s in shapes)

    tmp = out_path.with_suffix(".npy.tmp")
    out = open_memmap(tmp, mode="w+", dtype=np.float32,
                      shape=(total,) + shapes[0][1:])
    i = 0
    for f, s in zip(files, shapes):
        a = np.load(f, mmap_mode="r")
        out[i:i + s[0]] = a[:]
        i += s[0]
        del a
    out.flush()
    del out
    tmp.replace(out_path)

    times = np.concatenate([np.load(cache_dir / f"{split}_{y}_times.npy")
                            for y in years])
    np.save(out_path.parent / f"{split}_times.npy", times)
    return (total,) + shapes[0][1:]


def main():
    ap = argparse.ArgumentParser(
        description="Download ERA5 t2m from WB2 GCS onto HEALPix (resumable).")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--timeout", type=int, default=120,
                    help="per-request GCS read timeout in seconds (0 disables).")
    ap.add_argument("--chunk-time", type=int, default=8,
                    help="dask time-chunk size = size of each HTTP read.")
    ap.add_argument("--batch", type=int, default=64,
                    help="fields fetched (and retried) per batch within a year.")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="retries per batch before giving up.")
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the per-year datasets/hpx/_years/ files after merging.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    nside = int(cfg["hpx"]["nside"])
    stride = dcfg.get("time_stride", 1)

    hpx_dir = ensure_dir(cfg["paths"]["hpx_dir"])
    cache_dir = ensure_dir(Path(hpx_dir) / "_years")
    coords_path = Path(hpx_dir) / "coords.npz"

    print(f"Opening {dcfg['era5_zarr']}\n  nside={nside} | per-year | "
          f"timeout={args.timeout}s | chunk_time={args.chunk_time} | "
          f"batch={args.batch} | retries={args.max_retries}", flush=True)

    da = _open_da(dcfg, args.timeout, args.chunk_time)
    lat, lon = da["latitude"].values, da["longitude"].values
    np.savez(coords_path, lat=lat, lon=lon)
    remap = LatLonToHPX(lat, lon, nside)
    print(f"  source grid {len(lat)}x{len(lon)} -> 12x{nside}x{nside} faces", flush=True)

    splits = {
        "train": list(range(dcfg["train_years"][0], dcfg["train_years"][1] + 1)),
        "test": list(range(dcfg["test_years"][0], dcfg["test_years"][1] + 1)),
    }

    # ── Per-year download+remap (resumable) ───────────────────────────────
    for split, years in splits.items():
        for year in years:
            ypath = cache_dir / f"{split}_{year}.npy"
            if ypath.exists():
                print(f"[skip] {split} {year} already cached", flush=True)
                continue
            arr, times = _download_year(
                dcfg, remap, year, stride, args.batch,
                args.timeout, args.chunk_time, args.max_retries,
            )
            if not np.isfinite(arr).all():
                raise ValueError(f"{split} {year} contains NaN/Inf.")
            tmp = ypath.with_suffix(".npy.tmp")
            with open(tmp, "wb") as fh:  # file object: np.save won't append .npy
                np.save(fh, arr)
            tmp.replace(ypath)
            np.save(cache_dir / f"{split}_{year}_times.npy", times)
            print(f"[done] {split} {year}: {arr.shape} -> {ypath.name}", flush=True)
            del arr

    # ── Merge per-year caches ─────────────────────────────────────────────
    merged = {}
    for split, years in splits.items():
        merged[split] = _merge(cache_dir, split, years, Path(hpx_dir) / f"{split}.npy")
        print(f"Merged {split}: {merged[split]} -> {hpx_dir}/{split}.npy", flush=True)

    # ── Normalization stats from the train split (memmap-friendly) ────────
    train = np.load(Path(hpx_dir) / "train.npy", mmap_mode="r")
    s = ss = n = 0.0
    for i in range(0, len(train), 256):
        chunk = np.asarray(train[i:i + 256], dtype=np.float64)
        s += chunk.sum()
        ss += (chunk ** 2).sum()
        n += chunk.size
    mean = s / n
    std = float(np.sqrt(max(ss / n - mean ** 2, 1e-12)))
    np.savez(Path(hpx_dir) / "norm_stats.npz", mean=mean, std=std)
    units = dcfg.get("units", "")
    print(f"norm stats: mean={mean:.3f} {units}  std={std:.3f} {units}", flush=True)

    # ── Cleanup per-year caches ───────────────────────────────────────────
    if not args.keep_cache:
        for f in cache_dir.glob("*.npy"):
            f.unlink()
        try:
            cache_dir.rmdir()
        except OSError:
            pass

    print(f"Done -> {hpx_dir}", flush=True)


if __name__ == "__main__":
    main()
