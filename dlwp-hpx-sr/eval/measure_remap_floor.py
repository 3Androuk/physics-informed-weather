"""Measure the lat-lon -> mesh -> lat-lon round-trip floor.

The floor is the error a PERFECT model would still be charged when scored on
the lat-lon grid (the sibling project's protocol), because our prediction lives
on the mesh. It bounds how much of eval/compare_full_field.py's score is the
representation rather than the model.

Measured on the true ERA5 fields themselves — no model involved:

    truth (lat-lon)  --forward-->  HPX faces  --backward-->  lat-lon  vs truth

Backward uses exact spherical-harmonic synthesis straight onto the target grid
(ducc0, Clenshaw-Curtis geometry) rather than synthesising onto a finer mesh
and interpolating there — same result, far cheaper at large nside.

The `lmax` of the backward analysis matters more than the interpolation method:
capping at 2*nside discards content the mesh does carry. Default is
min(3*nside - 1, nlat - 1), i.e. as much as the mesh and the target grid can
jointly support.

Run:
    python -m eval.measure_remap_floor --nside 256 512 --n-fields 4
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.download_era5 import _open_da  # noqa: E402
from hpx.mesh import faces_to_nest  # noqa: E402
from hpx.remap import (LatLonToHPX, LatLonToHPXSHT,  # noqa: E402
                       hpx_to_latlon)
from utils import load_config  # noqa: E402


def back_sht(faces, nlat, nlon, lmax=None, nthreads=8):
    """(N,12,F,F) mesh -> (N,nlat,nlon) by exact CC synthesis."""
    import healpy as hp
    from ducc0.sht.experimental import synthesis_2d

    nside = faces.shape[-1]
    lmax = int(lmax or min(3 * nside - 1, nlat - 1))
    flat = faces_to_nest(np.asarray(faces), nside).reshape(-1, 12 * nside * nside)
    out = np.empty((len(flat), nlat, nlon))
    for i, m in enumerate(flat):
        ring = hp.reorder(m.astype(np.float64), n2r=True)
        alm = hp.map2alm(ring, lmax=lmax, iter=1)
        g = synthesis_2d(alm=np.ascontiguousarray(alm)[None], ntheta=nlat,
                         nphi=nlon, lmax=lmax, spin=0, geometry="CC",
                         nthreads=nthreads)[0]
        out[i] = g[::-1]                      # north-first -> ascending latitude
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--nside", type=int, nargs="+", default=[256])
    ap.add_argument("--n-fields", type=int, default=4)
    ap.add_argument("--forward", nargs="+", default=["bilinear", "sht"],
                    choices=["bilinear", "sht"])
    ap.add_argument("--backward", nargs="+", default=["bilinear", "sht"],
                    choices=["bilinear", "sht"])
    ap.add_argument("--lmax-back", type=int, default=None,
                    help="backward analysis band limit (default: as high as the "
                         "mesh and target grid jointly allow)")
    ap.add_argument("--band", type=float, nargs=2, default=[-60.0, 60.0],
                    help="latitude band the floor is reported over (the scoring region)")
    ap.add_argument("--out", default=None, help="write results as JSON here")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    times = np.load(hpx_dir / "test_times.npy")
    idxs = np.linspace(0, len(times) - 1, args.n_fields).astype(int)

    da = _open_da(cfg["data"], args.timeout, chunk_time=1)
    lat = da["latitude"].values.astype(np.float64)
    lon = da["longitude"].values.astype(np.float64)
    truth = da.sel(time=times[idxs]).values.astype(np.float64)
    lo, hi = args.band
    band = (lat >= lo) & (lat <= hi)
    units = cfg["data"].get("units", "phys")
    print(f"{args.n_fields} fields {truth.shape[1:]} | band [{lo}, {hi}] "
          f"({int(band.sum())} rows) | grid band limit l ~ {len(lat) - 1}")

    def score(rec):
        e = rec[:, band] - truth[:, band]
        return float(np.sqrt((e ** 2).mean()))

    results = {}
    for nside in args.nside:
        lmax_b = args.lmax_back or min(3 * nside - 1, len(lat) - 1)
        print(f"\n--- HPX{nside} | mesh limit l ~ {2 * nside} | "
              f"backward lmax {lmax_b} ---")
        for fname in args.forward:
            t0 = time.time()
            fwd = (LatLonToHPX(lat, lon, nside) if fname == "bilinear"
                   else LatLonToHPXSHT(lat, lon, nside))
            faces = fwd(truth)
            t_fwd = time.time() - t0
            for bname in args.backward:
                t1 = time.time()
                rec = (hpx_to_latlon(faces, lat, lon) if bname == "bilinear"
                       else back_sht(faces, len(lat), len(lon), lmax_b))
                s = score(rec)
                results[f"hpx{nside}/{fname}->{bname}"] = s
                print(f"  {fname:8s} -> {bname:8s} {s:8.4f} {units}   "
                      f"(fwd {t_fwd:.0f}s, back {time.time() - t1:.0f}s)")
    best = min(results, key=results.get)
    print(f"\nbest: {best} = {results[best]:.4f} {units}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"n_fields": args.n_fields, "band": args.band,
                       "units": units, "floors": results}, f, indent=2)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
