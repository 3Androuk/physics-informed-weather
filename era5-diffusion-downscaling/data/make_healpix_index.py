"""Precompute HEALPix interpolation indices/weights for the healpix geo encoder.

For every pixel of the full lat/lon grid (patch_dir/coords_full.npz) and every
level of the Nside ladder, store the 4 interpolation neighbor cell indices and
weights (healpy.get_interp_weights). PatchDataset crops these per patch, so the
encoder never needs healpy at train/eval time.

NOTE: this is the ONLY place healpy is imported, and healpy ships Linux/macOS
wheels only — run this once on the CLUSTER, not on Windows:

    python -m data.make_healpix_index --config config/t2m.yaml

Output: patch_dir/healpix_index.npz with
    idx    (L, H, W, 4) int64   neighbor cell index per level/pixel
    w      (L, H, W, 4) float32 interp weights (sum to 1 per pixel)
    nsides (L,)         int64   the Nside ladder
    scheme str                  healpy indexing scheme used ("ring")
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.geo_encoding import healpix_nside_ladder  # noqa: E402
from utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="Precompute HEALPix interp indices/weights (cluster-only: healpy).")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    try:
        import healpy as hp
    except ImportError as e:  # no Windows wheels — see module docstring
        raise SystemExit(
            "healpy is required for this precompute step (Linux/macOS only): "
            "pip install healpy") from e

    cfg = load_config(args.config)
    g = cfg["geo"]
    nsides = healpix_nside_ladder(
        g["n_levels"], g.get("healpix_nside_min", 1), g.get("healpix_nside_max", 128))
    patch_dir = Path(cfg["paths"]["patch_dir"])
    cf = np.load(patch_dir / "coords_full.npz")
    lat, lon = cf["lat"], cf["lon"]
    h, w_ = lat.size, lon.size
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    theta = np.deg2rad(90.0 - lat2).ravel()          # colatitude
    phi = np.deg2rad(np.mod(lon2, 360.0)).ravel()

    n_lv = len(nsides)
    idx = np.empty((n_lv, h, w_, 4), dtype=np.int64)
    wts = np.empty((n_lv, h, w_, 4), dtype=np.float32)
    for lv, ns in enumerate(nsides):
        pix, wt = hp.get_interp_weights(ns, theta, phi)   # ring scheme, (4, N) each
        wt = wt / wt.sum(axis=0, keepdims=True)           # exact normalization
        assert np.allclose(wt.sum(axis=0), 1.0, atol=1e-5), "interp weights must sum to 1"
        idx[lv] = pix.T.reshape(h, w_, 4)
        wts[lv] = wt.T.reshape(h, w_, 4).astype(np.float32)
        print(f"level {lv}: Nside {ns:4d} | cells {12 * ns * ns:>9,}")

    out = patch_dir / "healpix_index.npz"
    np.savez(out, idx=idx, w=wts, nsides=np.array(nsides, dtype=np.int64), scheme="ring")
    n_params = sum(12 * ns * ns for ns in nsides) * g["n_features_per_level"]
    print(f"-> {out} | grid {h}x{w_} | encoder table params: {n_params:,}")


if __name__ == "__main__":
    main()
