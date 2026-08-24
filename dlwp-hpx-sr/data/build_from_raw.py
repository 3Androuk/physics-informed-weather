"""Build the HEALPix face dataset from an EXISTING era5-diffusion-downscaling
raw download instead of re-streaming everything from GCS.

Points at a sibling-format raw dir (train.npy / test.npy as (T, H, W) float32
plus coords.npz with the lat/lon grid) and remaps those fields to HPX faces,
producing exactly the outputs of data.download_era5 (<hpx_dir>/train.npy,
test.npy, times, norm_stats.npz, coords.npz).

The HEALPix mesh covers the whole sphere, so if the raw fields are a latitude
band (the sibling default crops to +-60 deg) the missing polar rows are
fetched from the configured WB2 store — a fraction of the full transfer —
and stitched onto the existing rows before remapping. If the raw fields
already span the globe, nothing but the time axis is read from the store.

Alignment with the raw file is verified: the per-year strided timestep counts
derived from the store's time axis (using config years/stride, which must
match the settings the raw file was downloaded with) must sum to the raw
file's length, and the raw lat/lon rows must be a subset of the store grid.

Run:
    python -m data.build_from_raw --config config/default.yaml \
        --raw-dir ../era5-diffusion-downscaling/datasets/raw
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.download_era5 import _merge, _open_da, _year_sub  # noqa: E402
from hpx.remap import LatLonToHPX  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402


def _match_rows(store_vals: np.ndarray, raw_vals: np.ndarray, name: str):
    """Indices of raw_vals inside store_vals (exact grid match required)."""
    idx = np.searchsorted(store_vals, raw_vals)
    idx = np.clip(idx, 0, len(store_vals) - 1)
    if not np.allclose(store_vals[idx], raw_vals, atol=1e-4):
        raise ValueError(
            f"raw {name} grid is not a subset of the store grid — was the raw "
            f"download made from {name}-compatible store resolution?")
    return idx


def _fetch_rows_year(da, year, stride, row_idx, batch, max_retries, reopen):
    """Fetch (T_y, len(row_idx), W) for one year, batched with retries."""
    sub = _year_sub(da, year, stride).isel(latitude=row_idx)
    T = int(sub.sizes["time"])
    H, W = len(row_idx), int(sub.sizes["longitude"])
    out = np.empty((T, H, W), dtype=np.float32)
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
                        f"{year} rows [{start}:{stop}] failed after "
                        f"{max_retries} attempts: {type(e).__name__}: {e}") from e
                wait = min(30, 3 * attempt)
                print(f"    [retry {attempt}/{max_retries}] {year} [{start}:{stop}] "
                      f"{type(e).__name__}; reconnecting in {wait}s ...", flush=True)
                time.sleep(wait)
                da = reopen()
                sub = _year_sub(da, year, stride).isel(latitude=row_idx)
        start = stop
        print(f"    {year}: {stop}/{T}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Remap an existing raw lat-lon download to HEALPix faces, "
                    "topping up missing polar rows from the WB2 store.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--raw-dir", required=True,
                    help="sibling-format dir with train.npy, test.npy, coords.npz")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--chunk-time", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--keep-cache", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    nside = int(cfg["hpx"]["nside"])
    stride = dcfg.get("time_stride", 1)
    raw_dir = Path(args.raw_dir)

    coords = np.load(raw_dir / "coords.npz")
    raw_lat, raw_lon = coords["lat"].astype(np.float64), coords["lon"].astype(np.float64)
    lat_flip = raw_lat[0] > raw_lat[-1]
    if lat_flip:
        raw_lat = raw_lat[::-1]

    hpx_dir = ensure_dir(cfg["paths"]["hpx_dir"])
    cache_dir = ensure_dir(Path(hpx_dir) / "_years")

    # ── Store grid + per-year time axis (metadata-sized reads only) ───────
    def reopen():
        return _open_da(dcfg, args.timeout, args.chunk_time)

    da = reopen()  # (time, lat ascending, lon)
    st_lat, st_lon = da["latitude"].values.astype(np.float64), da["longitude"].values.astype(np.float64)
    if len(raw_lon) != len(st_lon) or not np.allclose(raw_lon, st_lon, atol=1e-4):
        raise ValueError("raw lon grid differs from the store — check data.era5_zarr")
    band_idx = _match_rows(st_lat, raw_lat, "lat")
    cap_idx = np.setdiff1d(np.arange(len(st_lat)), band_idx)
    order = np.argsort(np.concatenate([st_lat[cap_idx], st_lat[band_idx]]))
    full_lat = np.concatenate([st_lat[cap_idx], st_lat[band_idx]])[order]
    if not np.allclose(full_lat, st_lat):
        raise ValueError("stitched latitude grid does not reproduce the store grid")
    print(f"raw grid {len(raw_lat)}x{len(raw_lon)}"
          + (f" -> fetching {len(cap_idx)} missing polar rows/field from the store"
             if len(cap_idx) else " already covers the store grid — no field download"),
          flush=True)

    np.savez(Path(hpx_dir) / "coords.npz", lat=st_lat, lon=st_lon)
    remap = LatLonToHPX(st_lat, st_lon, nside)

    splits = {
        "train": list(range(dcfg["train_years"][0], dcfg["train_years"][1] + 1)),
        "test": list(range(dcfg["test_years"][0], dcfg["test_years"][1] + 1)),
    }

    # ── Verify the raw time axes match config years/stride ────────────────
    year_counts, year_times = {}, {}
    for years in splits.values():
        for y in years:
            sub = _year_sub(da, y, stride)
            year_counts[y] = int(sub.sizes["time"])
            year_times[y] = sub["time"].values
    for split, years in splits.items():
        raw = np.load(raw_dir / f"{split}.npy", mmap_mode="r")
        expect = sum(year_counts[y] for y in years)
        if len(raw) != expect:
            raise ValueError(
                f"{split}.npy has {len(raw)} fields but config years {years} "
                f"@ stride {stride} give {expect} — the raw download was made "
                "with different data.train_years/test_years/time_stride; set "
                "the config to the values used for the raw download.")
        if raw.shape[1:] != (len(raw_lat), len(raw_lon)):
            raise ValueError(f"{split}.npy shape {raw.shape[1:]} does not match "
                             f"coords.npz ({len(raw_lat)}, {len(raw_lon)})")

    # ── Per-year: (fetch caps ->) stitch -> remap -> cache (resumable) ─────
    for split, years in splits.items():
        raw = np.load(raw_dir / f"{split}.npy", mmap_mode="r")
        offset = 0
        for year in years:
            T = year_counts[year]
            ypath = cache_dir / f"{split}_{year}.npy"
            if ypath.exists():
                print(f"[skip] {split} {year} already cached", flush=True)
                offset += T
                continue
            band = np.asarray(raw[offset:offset + T], dtype=np.float32)
            if lat_flip:
                band = band[:, ::-1]
            full = np.empty((T, len(st_lat), len(st_lon)), dtype=np.float32)
            full[:, band_idx] = band
            if len(cap_idx):
                print(f"  {year}: fetching polar rows ...", flush=True)
                full[:, cap_idx] = _fetch_rows_year(
                    da, year, stride, cap_idx, args.batch, args.max_retries, reopen)
            if not np.isfinite(full).all():
                raise ValueError(f"{split} {year} contains NaN/Inf.")
            faces = remap(full)
            tmp = ypath.with_suffix(".npy.tmp")
            with open(tmp, "wb") as fh:
                np.save(fh, faces)
            tmp.replace(ypath)
            np.save(cache_dir / f"{split}_{year}_times.npy", year_times[year])
            print(f"[done] {split} {year}: {faces.shape} -> {ypath.name}", flush=True)
            offset += T

    # ── Merge + normalization stats (same as download_era5) ───────────────
    for split, years in splits.items():
        shape = _merge(cache_dir, split, years, Path(hpx_dir) / f"{split}.npy")
        print(f"Merged {split}: {shape} -> {hpx_dir}/{split}.npy", flush=True)

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
