"""GATE for any HEALPix-native backbone: measure the remap round-trip floor.

Before training anything on a HEALPix representation, measure what the
lat-lon -> HPX -> lat-lon round trip alone costs on real fields, with the
project's own metrics. The result is a FLOOR under any HPX-internal model
(its own error adds on top). Decision rule: if the round-trip spectral error
at the candidate Nside is comparable to the encoder-ladder spread
(~0.002-0.005 spec-logL1), an HPX backbone starts underwater and the idea is
dead on measurement; if it is negligible, the pilot is worth training.

Forward remap: cell-mean aggregation (bincount), the same operator a
training-data remap would use. Backward: healpy bilinear interpolation.
Scoring crops `--margin` degrees off the band edges so cells straddling the
boundary (partially outside the lat band) don't contaminate the number.

Run (in a worktree whose raw test.npy still exists, e.g. wb2-20var):
    python -m data.healpix_roundtrip_check --config config/wb2_20var.yaml \
        --channel 0 --nsides 128 256 512
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import l2_norm, spectrum_log_l1  # noqa: E402
from utils import load_config  # noqa: E402


def roundtrip(fields, lat, lon, nside):
    """fields: (N, H, W) physical units -> (N, H, W) after HPX round trip."""
    import healpy as hp
    theta = np.deg2rad(90.0 - lat)[:, None] * np.ones_like(lon)[None, :]
    phi = np.deg2rad(lon % 360.0)[None, :] * np.ones_like(lat)[:, None]
    pix = hp.ang2pix(nside, theta.ravel(), phi.ravel())
    npix = hp.nside2npix(nside)
    counts = np.bincount(pix, minlength=npix)
    out = np.empty_like(fields)
    for i, f in enumerate(fields):
        sums = np.bincount(pix, weights=f.ravel().astype(np.float64),
                           minlength=npix)
        hpx = np.where(counts > 0, sums / np.maximum(counts, 1), hp.UNSEEN)
        out[i] = hp.get_interp_val(hpx, theta.ravel(), phi.ravel()
                                   ).reshape(f.shape)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/t2m.yaml")
    ap.add_argument("--channel", type=int, default=0,
                    help="Channel of raw test.npy to score (0 = t2m on the "
                         "20-var layout; single-var raw has one channel).")
    ap.add_argument("--n-fields", type=int, default=8)
    ap.add_argument("--nsides", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--margin", type=float, default=5.0,
                    help="Degrees cropped off each band edge before scoring.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw = np.load(raw_dir / "test.npy", mmap_mode="r")
    if raw.ndim == 3:
        raw = raw[:, None]
    fields = np.asarray(raw[: args.n_fields, args.channel], dtype=np.float64)
    z = np.load(raw_dir / "coords.npz")
    lat, lon = z["lat"], z["lon"]

    keep = np.abs(lat) <= (np.abs(lat).max() - args.margin)
    ref = fields[:, keep]
    print(f"{len(fields)} test fields | grid {fields.shape[-2:]} | scoring "
          f"|lat| <= {np.abs(lat[keep]).max():.1f} ({keep.sum()} rows)\n"
          f"{'nside':>6} {'cell':>7} {'L2 (phys)':>10} {'spec-logL1':>11}")
    for nside in args.nsides:
        back = roundtrip(fields, lat, lon, nside)[:, keep]
        cell_deg = np.degrees(np.sqrt(4 * np.pi / (12 * nside * nside)))
        print(f"{nside:>6} {cell_deg:>6.2f}° {l2_norm(back, ref):>10.4f} "
              f"{spectrum_log_l1(back, ref):>11.4f}")
    print("\nCompare against the ladder: encoder arms differ by ~0.005-0.01 "
          "L2 and ~0.002-0.005 spec-logL1. A floor at or above that scale "
          "sinks any HPX-internal backbone before it trains.")


if __name__ == "__main__":
    main()
