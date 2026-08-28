"""How much does the deployment path differ from the training path?

Training builds the coarse input ON the mesh:

    ERA5 lat-lon  --remap-->  HPX(nside)  --coarsen-->  HPX(nside/r)      (A)

so coarsen(truth) == input holds exactly, which is what makes the
data-consistency projection exact. But in deployment the coarse field arrives
already coarse and on a lat-lon grid (a coarse forecast model, or CMIP6), so it
must be remapped at the coarse level instead:

    ERA5 lat-lon  --coarsen-->  1 deg lat-lon  --remap-->  HPX(nside/r)   (B)

This script measures RMSE(A, B): the size of that train/deploy mismatch, i.e.
how wrong the projection's constraint is when the input did not come from our
own mesh coarsening.

The expectation is that it is much smaller than the FINE-level remap floor
(0.124 K at HPX256, eval/measure_remap_floor.py), because a coarse field is
smooth and the target mesh samples it comfortably. This turns that expectation
into a number.

Note this measures only the *operator* mismatch, with both paths starting from
the same ERA5 truth. Real coarse data additionally carries model bias, which is
a separate and larger problem — see the discussion of soft vs exact projection.

Run:
    python -m eval.deploy_mismatch --config config/default.yaml --ratios 4 8 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen_faces  # noqa: E402
from data.download_era5 import _open_da  # noqa: E402
from hpx.remap import LatLonToHPX  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402


def coarsen_latlon(field, lat, lon, r):
    """Average-pool a lat-lon field by r, returning (field, lat, lon).

    The 0.25 deg grid has 721 rows (poles included); 721 is not divisible, so
    the last row is dropped first — the standard way to get a cell-centered
    1 deg grid from it.
    """
    nlat = (len(lat) // r) * r
    f = torch.from_numpy(np.ascontiguousarray(field[:, :nlat]))[:, None]
    pooled = torch.nn.functional.avg_pool2d(f, r)[:, 0].numpy()
    la = lat[:nlat].reshape(-1, r).mean(axis=1)
    lo = lon.reshape(-1, r).mean(axis=1)
    return pooled, la, lo


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--ratios", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n-fields", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config(args.config)
    nside = int(cfg["hpx"]["nside"])
    units = cfg["data"].get("units", "phys")
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    times = np.load(hpx_dir / "test_times.npy")
    idxs = np.linspace(0, len(times) - 1, args.n_fields).astype(int)
    da = _open_da(cfg["data"], args.timeout, chunk_time=1)
    lat = da["latitude"].values.astype(np.float64)
    lon = da["longitude"].values.astype(np.float64)
    truth = da.sel(time=times[idxs]).values.astype(np.float64)
    print(f"{args.n_fields} fields {truth.shape[1:]} | mesh HPX{nside}")

    # Path A: remap once at full resolution, then coarsen on the mesh
    fine_faces = LatLonToHPX(lat, lon, nside)(truth)

    out = {"nside": nside, "n_fields": args.n_fields, "units": units,
           "fine_remap_floor_reference": 0.124, "ratios": {}}
    for r in args.ratios:
        if nside % r:
            print(f"  skip {r}x"); continue
        a = coarsen_faces(torch.from_numpy(fine_faces)[:, :, None], r)[:, :, 0].numpy()

        # Path B: coarsen in lat-lon first, then remap at the coarse level
        cf, cla, clo = coarsen_latlon(truth, lat, lon, r)
        b = LatLonToHPX(cla, clo, nside // r)(cf)

        rmse = float(np.sqrt(((a - b) ** 2).mean()))
        mx = float(np.abs(a - b).max())
        # scale for context: spread of the coarse field itself
        spread = float(a.std())
        out["ratios"][f"{r}x"] = {"rmse": rmse, "max_abs": mx,
                                  "coarse_field_std": spread,
                                  "nside_coarse": nside // r,
                                  "latlon_coarse": list(cf.shape[1:])}
        print(f"  {r}x -> HPX{nside // r} from {cf.shape[1]}x{cf.shape[2]} lat-lon: "
              f"mismatch {rmse:.4f} {units} (max {mx:.3f}, field std {spread:.2f})")

    path = Path(results_dir) / "deploy_mismatch.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
