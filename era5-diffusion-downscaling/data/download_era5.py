"""Stream ERA5 Z500 from the WeatherBench 2 public GCS store.

Downloads geopotential @ 500 hPa at 0.25 deg (721 x 1440), cropped to a
mid-latitude band, for the configured train/test year ranges, and caches each
split as a float32 .npy of shape (T, H, W). No credentials required.

Run:
    python -m data.download_era5 --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config  # noqa: E402


def _fetch_split(da, years, stride):
    import xarray as xr  # local import so non-download code has no hard dep

    sub = da.sel(time=slice(f"{years[0]}-01-01", f"{years[1]}-12-31"))
    if stride and stride > 1:
        sub = sub.isel(time=slice(None, None, stride))
    print(f"  computing {sub.sizes['time']} fields for {years[0]}-{years[1]} ...")
    arr = sub.compute().values.astype(np.float32)  # (T, H, W)
    return arr


def main():
    ap = argparse.ArgumentParser(description="Download ERA5 Z500 from WB2 GCS.")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import xarray as xr

    dcfg = cfg["data"]
    raw_dir = ensure_dir(cfg["paths"]["raw_dir"])

    print(f"Opening {dcfg['era5_zarr']} ...")
    ds = xr.open_zarr(
        dcfg["era5_zarr"], chunks={"time": 50},
        storage_options=dict(token=dcfg["gcs_token"]),
    )
    da = ds[dcfg["variable"]].sel(level=dcfg["level"])
    da = da.transpose("time", "latitude", "longitude")

    # Mid-latitude band crop (avoid the worst lat-lon distortion near the poles).
    lat = da["latitude"].values
    lo, hi = cfg["patches"]["lat_range"]
    lat_mask = (lat >= lo) & (lat <= hi)
    da = da.isel(latitude=np.where(lat_mask)[0])
    lat = da["latitude"].values
    lon = da["longitude"].values

    stride = dcfg.get("time_stride", 1)
    train = _fetch_split(da, dcfg["train_years"], stride)
    test = _fetch_split(da, dcfg["test_years"], stride)

    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise ValueError("ERA5 fields contain NaN/Inf.")

    np.save(Path(raw_dir) / "train.npy", train)
    np.save(Path(raw_dir) / "test.npy", test)
    np.savez(Path(raw_dir) / "coords.npz", lat=lat, lon=lon)
    print(
        f"Saved: train {train.shape}, test {test.shape}, "
        f"grid {train.shape[1]}x{train.shape[2]} -> {raw_dir}"
    )


if __name__ == "__main__":
    main()
