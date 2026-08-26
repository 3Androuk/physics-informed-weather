"""Stream ERA5 variables from the WeatherBench 2 public GCS store — resumable.

Downloads the configured variables (data.variables list, or the legacy single
data.variable/level) at 0.25 deg (721 x 1440), cropped to a mid-latitude band,
for the configured train/test year ranges, and caches each split as a float32
.npy of shape (T, C, H, W) — one channel per variable, in config order. No
credentials required.

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
  * Every write goes through an on-disk memmap — each year is STREAMED into its
    cache file batch by batch, and the caches are merged into train.npy /
    test.npy the same way. Peak RAM is one batch (~0.8 GiB at --batch 16),
    never a whole year (~19 GiB at 20 channels), so the download runs on a
    login node instead of costing GPU node-hours.

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
from utils import channel_labels, channel_specs, ensure_dir, load_config  # noqa: E402


class BadFieldData(Exception):
    """NaN/Inf in a fetched batch — a data fault, never a transport fault.

    Distinct from the transient errors the retry loop swallows: reconnecting
    would refetch the same bad numbers, so this propagates immediately.
    """


def _open_da(dcfg, lat_range, timeout, chunk_time):
    """Open the WB2 zarr and return ONE DataArray PER CHANNEL, cropped to
    lat_range, plus the retained lat/lon vectors:
        ([da_0, ..., da_C-1], lat, lon)   each da: (time, latitude, longitude)

    Deliberately NOT concatenated along a 'channel' dim. A single `.values` on
    a 20-variable concat makes dask materialize all 20 variables' chunks at
    once; measured on a BriCS login node that peaked at 3.46 GiB against a
    4 GiB cgroup cap and was OOM-killed (exit 137) even at --batch 4. Reading
    channel by channel keeps the working set to one variable's chunks, and the
    caller writes each straight into its slot in the on-disk memmap.

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

    lat_all = ds["latitude"].values
    lo, hi = lat_range
    keep = np.where((lat_all >= lo) & (lat_all <= hi))[0]

    # Group the requested channels BY VARIABLE and take all of that variable's
    # levels in one selection. WB2 stores pressure-level variables with every
    # level in a single chunk — geopotential is (1, 13, 721, 1440) = 51.5 MiB —
    # so `.sel(level=500)` still fetches all 13 levels. Asking for z500/z700/z850
    # as three separate channels therefore refetched the identical chunk three
    # times: ~772 MiB pulled per timestep to keep ~59 MiB. One read per variable
    # cuts both the traffic and the peak working set by ~3x here.
    by_var = {}
    for ci, spec in enumerate(channel_specs(dcfg)):
        by_var.setdefault(spec["name"], []).append((ci, spec["level"]))

    groups = []
    for name, entries in by_var.items():
        da = ds[name]
        levels = [lv for _, lv in entries if lv is not None]
        if levels:
            da = da.sel(level=levels)
            slots = [(ci, levels.index(lv)) for ci, lv in entries]
        else:  # surface variable (e.g. 2m_temperature) — no level dim
            slots = [(ci, None) for ci, _ in entries]
        da = da.reset_coords(drop=True).isel(latitude=keep)
        groups.append((da, slots))
    return groups, lat_all[keep], ds["longitude"].values


def _year_sub(da, year, stride):
    """Strided time-slice of a single calendar year."""
    sub = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
    if stride and stride > 1:
        sub = sub.isel(time=slice(None, None, stride))
    return sub


def _download_year(dcfg, lat_range, year, stride, batch, timeout, chunk_time,
                   max_retries, out_path):
    """Stream one year into `out_path` (.npy) -> (shape (T, C, H, W), lat, lon).

    Each batch of `batch` (strided) fields is read independently and retried on
    failure with a fresh connection, so a stall costs at most one batch.

    Batches are written STRAIGHT INTO an on-disk memmap rather than buffered in
    a full-year array: at 20 channels a year is ~19 GiB, which does not fit the
    4 GiB cap on a BriCS login node (and running the download on a GPU compute
    node instead would burn node-hours with the GPUs idle). Peak RAM here is one
    batch — ~2.5 GiB at --batch 48, ~0.8 GiB at --batch 16.

    Finiteness is checked per batch, so a bad field fails fast instead of after
    the whole year has been fetched.
    """
    from numpy.lib.format import open_memmap

    groups, lat, lon = _open_da(dcfg, lat_range, timeout, chunk_time)
    subs = [(_year_sub(d, year, stride), slots) for d, slots in groups]
    T = int(subs[0][0].sizes["time"])
    C = sum(len(slots) for _, slots in subs)
    H, W = len(lat), len(lon)

    out = open_memmap(out_path, mode="w+", dtype=np.float32, shape=(T, C, H, W))
    n_batches = (T + batch - 1) // batch
    print(f"  {year}: {T} fields x {C} channels in {n_batches} batches of {batch} ...",
          flush=True)

    try:
        start = 0
        while start < T:
            stop = min(start + batch, T)
            # One VARIABLE at a time: the working set is a single variable's
            # chunks (all its levels, which arrive together anyway), not all C
            # channels at once — see _open_da.
            for gi in range(len(subs)):
                for attempt in range(1, max_retries + 1):
                    try:
                        sub, slots = subs[gi]
                        chunk = sub.isel(time=slice(start, stop)).values.astype(np.float32)
                        if not np.isfinite(chunk).all():
                            raise BadFieldData(
                                f"{year} {[ci for ci, _ in slots]} fields "
                                f"[{start}:{stop}] contain NaN/Inf.")
                        for ci, lpos in slots:
                            out[start:stop, ci] = chunk if lpos is None else chunk[:, lpos]
                        del chunk
                        break
                    except BadFieldData:           # bad data — retrying refetches it
                        raise
                    except Exception as e:  # noqa: BLE001 - retry any GCS/dask read error
                        if attempt == max_retries:
                            raise RuntimeError(
                                f"{year} group {gi} fields [{start}:{stop}] failed after "
                                f"{max_retries} attempts: {type(e).__name__}: {e}"
                            ) from e
                        wait = min(30, 3 * attempt)
                        print(f"    [retry {attempt}/{max_retries}] {year} grp{gi} "
                              f"[{start}:{stop}] {type(e).__name__}; reconnecting in "
                              f"{wait}s ...", flush=True)
                        time.sleep(wait)
                        groups, _, _ = _open_da(dcfg, lat_range, timeout, chunk_time)
                        subs = [(_year_sub(d, year, stride), sl) for d, sl in groups]
            start = stop
            print(f"    {year}: {stop}/{T}", flush=True)
        out.flush()
    finally:
        del out    # close the mapping before the caller renames the file

    return (T, C, H, W), lat, lon


def _merge(cache_dir, split, years, out_path):
    """Concatenate per-year caches into out_path (.npy) via a memmap (low RAM).

    Writes to a .tmp then renames so an interrupted merge can't leave a
    half-written train.npy.
    """
    from numpy.lib.format import open_memmap

    files = [cache_dir / f"{split}_{y}.npy" for y in years]
    shapes = [np.load(f, mmap_mode="r").shape for f in files]
    total = sum(s[0] for s in shapes)
    rest = shapes[0][1:]   # (C, H, W) — or (H, W) for legacy single-channel caches

    tmp = out_path.with_suffix(".npy.tmp")
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(total, *rest))
    i = 0
    for f, s in zip(files, shapes):
        a = np.load(f, mmap_mode="r")
        out[i:i + s[0]] = a[:]
        i += s[0]
        del a
    out.flush()
    del out
    tmp.replace(out_path)
    return (total, *rest)


def main():
    ap = argparse.ArgumentParser(description="Download ERA5 variables from WB2 GCS (resumable).")
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

    labels = channel_labels(dcfg)
    print(f"Opening {dcfg['era5_zarr']}\n  channels ({len(labels)}): {', '.join(labels)}\n"
          f"  per-year | timeout={args.timeout}s | "
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
            # Streamed straight into the .tmp memmap, then renamed: an
            # interrupted year leaves a .tmp that the next run overwrites,
            # never a half-written file that "resume" would trust.
            tmp = ypath.with_suffix(".npy.tmp")
            shape, lat, lon = _download_year(
                dcfg, lat_range, year, stride, args.batch,
                args.timeout, args.chunk_time, args.max_retries, tmp,
            )
            tmp.replace(ypath)
            print(f"[done] {split} {year}: {shape} -> {ypath.name}", flush=True)
            if not coords_path.exists():
                np.savez(coords_path, lat=lat, lon=lon, channels=np.array(labels))

    if not coords_path.exists():  # e.g. resumed run where every year was cached
        _, lat, lon = _open_da(dcfg, lat_range, args.timeout, args.chunk_time)
        np.savez(coords_path, lat=lat, lon=lon, channels=np.array(labels))

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

    h, w = merged["train"][-2], merged["train"][-1]
    print(f"Done. {len(labels)} channel(s), grid {h}x{w} -> {raw_dir}", flush=True)


if __name__ == "__main__":
    main()
