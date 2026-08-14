"""Precompute normalized static physiographic conditioning fields.

Streams the time-invariant fields listed in cfg geo.static_fields (default:
geopotential_at_surface [orography], land_sea_mask,
slope_of_sub_gridscale_orography) from the WB2 ERA5 zarr, crops them to the
configured latitude band (same crop as download_era5), z-score normalizes each
field over the band, and writes patch_dir/static_fields.npz. PatchDataset
crops these per patch for the 'static' geo encoder — the literature-standard
baseline for the learned location embeddings.

One tiny download (a few MB); run once per patch_dir:

    python -m data.make_static_fields --config config/t2m.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.geo_encoding import DEFAULT_STATIC_FIELDS  # noqa: E402
from utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["data"]
    names = cfg.get("geo", {}).get("static_fields", DEFAULT_STATIC_FIELDS)
    lo, hi = cfg["patches"]["lat_range"]
    patch_dir = Path(cfg["paths"]["patch_dir"])

    import xarray as xr  # local import: only this script needs it
    ds = xr.open_zarr(dcfg["era5_zarr"], storage_options=dict(token=dcfg["gcs_token"]))

    # The patch grid: origins index into these coords, so shapes must match.
    cf = np.load(patch_dir / "coords_full.npz")
    lat_grid, lon_grid = cf["lat"], cf["lon"]

    fields, means, stds = [], [], []
    for name in names:
        da = ds[name]
        if "time" in da.dims:  # some stores keep a redundant time dim on statics
            da = da.isel(time=0)
        da = da.transpose("latitude", "longitude")
        lat = da["latitude"].values
        da = da.isel(latitude=np.where((lat >= lo) & (lat <= hi))[0])
        arr = np.nan_to_num(da.values.astype(np.float32))  # e.g. masked sea points
        assert arr.shape == (lat_grid.size, lon_grid.size), (
            f"{name}: static grid {arr.shape} does not match the patch grid "
            f"{(lat_grid.size, lon_grid.size)} — check era5_zarr and lat_range")
        mean, std = float(arr.mean()), float(arr.std())
        std = std if std > 1e-8 else 1.0
        fields.append((arr - mean) / std)
        means.append(mean)
        stds.append(std)
        print(f"  {name}: mean={mean:.3f} std={std:.3f}")

    out = patch_dir / "static_fields.npz"
    np.savez(out, fields=np.stack(fields), names=np.array(names),
             mean=np.array(means), std=np.array(stds))
    print(f"-> {out} | {len(names)} field(s), grid {fields[0].shape}")


if __name__ == "__main__":
    main()
