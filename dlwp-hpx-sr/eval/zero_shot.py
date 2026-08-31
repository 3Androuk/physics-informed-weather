"""Zero-shot downscaling baseline: trained ratio + parameter-free interpolation.

Reproduces the protocol of Ekström et al. (2025), "On the effectiveness of
neural operators at zero-shot weather downscaling" (arXiv:2409.13955), on the
mesh. Rather than feeding a ratio-4 model an 8x-degraded field — a genuine
input-distribution shift, under which our regressors collapse — the model is run
at the ratio it was trained for on a correspondingly coarser mesh, and
parameter-free interpolation covers the remaining factor. In that paper this
kept SwinIR degrading gracefully (MSE 0.36 -> 0.51 going 4x -> 8x) rather than
collapsing, so it is the honest strong baseline for any zero-shot claim, and a
far tougher bar than plain bicubic.

The HEALPix ladder makes each stage exact: coarsening is average pooling and
every level is a power-of-two sub-mesh, so no arbitrary regridding is involved.

Run:
    python -m eval.zero_shot --config config/default.yaml --ratios 4 8 16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import HPXDataset, Normalizer  # noqa: E402
from data.degrade import (coarsen_faces, upsample_bilinear_faces,  # noqa: E402
                          upsample_nearest_faces)
from eval.metrics import faces_as_images, spectrum_log_l1  # noqa: E402
from hpx.padding import HEALPixPadding  # noqa: E402
from models.hpx_unet import load_at_nside  # noqa: E402
from utils import ensure_dir, get_device, load_config, set_seed  # noqa: E402


@torch.no_grad()
def cascade(ckpt, lf, ratio, native_ratio, device, cache=None):
    """Run the model at its native ratio, then interpolate the rest of the way.

    Args:
        lf: (B,12,C,f,f) coarse observation, f = nside_target / ratio.
        ratio: the requested (possibly unseen) downscaling factor.
        native_ratio: the factor the checkpoint was trained at.
    Returns:
        (B,12,C,F,F) reconstruction on the target mesh.
    """
    nside_target = int(ckpt["config"]["hpx"]["nside"])
    nside_lf = lf.shape[-1]
    nside_mid = nside_lf * native_ratio
    if nside_mid > nside_target:
        raise ValueError(f"ratio {ratio} is finer than the model's native "
                         f"{native_ratio}: nothing to cascade")
    key = nside_mid
    if cache is None or key not in cache:
        model = load_at_nside(ckpt, nside_mid, device=device)
        if cache is not None:
            cache[key] = model
    else:
        model = cache[key]

    x = upsample_nearest_faces(lf, native_ratio)   # block structure as trained
    y_mid = model(x)
    rest = nside_target // nside_mid
    if rest == 1:
        return y_mid
    pad = HEALPixPadding(nside_mid, 1).to(device)
    return upsample_bilinear_faces(y_mid, rest, pad)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--checkpoint", default=None, help="override eval.checkpoint")
    ap.add_argument("--ratios", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--n-test-samples", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()
    nside = int(cfg["hpx"]["nside"])
    native = int(cfg["sr"]["ratio"])
    units = cfg["data"].get("units", "phys")
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / (args.checkpoint
                                                 or cfg["eval"]["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    normalizer = Normalizer(ckpt["norm_mean"], ckpt["norm_std"])
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, native ratio {native}x)")

    ds = HPXDataset(hpx_dir / "test.npy", normalizer)
    n = min(args.n_test_samples, len(ds))
    # spread over the whole test period: the first N fields are one contiguous
    # winter month, which biases absolute scores by several percent
    # .round(), not truncation: evaluate_guided / evaluate_diffusion round,
    # and at n=16 truncation picks 6 different fields out of 16.
    sel = np.unique(np.linspace(0, len(ds) - 1, n).round().astype(int))
    n = len(sel)
    print(f"{n}/{len(ds)} test fields (spread over the test period) | "
          f"ratios {args.ratios} | nside {nside}")

    cache = {}
    out = {"checkpoint": str(ckpt_path), "native_ratio": native,
           "n_samples": n, "units": units, "ratios": {}}
    for ratio in args.ratios:
        if nside % ratio:
            print(f"  skip {ratio}x: nside {nside} not divisible"); continue
        # cascade needs nside_lf * native <= nside_target, so it is undefined
        # for a ratio FINER than the model's native one (nothing to cascade).
        methods = ((("cascade",) if ratio >= native else ())
                   + ("naive", "bilinear"))
        sq = {m: 0.0 for m in methods}
        spec = {m: [] for m in methods}
        n_px = 0
        pad1 = HEALPixPadding(nside // ratio, 1).to(device)
        naive_model = cache.setdefault(
            nside, load_at_nside(ckpt, nside, device=device))
        with torch.no_grad():
            for start in range(0, n, args.batch_size):
                idx = sel[start:start + args.batch_size]
                y = torch.stack([ds[int(i)] for i in idx]).to(device)
                lf = coarsen_faces(y, ratio)
                preds = {
                    # what we did before: feed the OOD-degraded field directly
                    "naive": naive_model(upsample_nearest_faces(lf, ratio)),
                    "bilinear": upsample_bilinear_faces(lf, ratio, pad1),
                }
                if ratio >= native:   # Ekström-style: native ratio, then interpolate
                    preds["cascade"] = cascade(ckpt, lf, ratio, native, device, cache)
                truth = normalizer.decode(y).cpu().numpy()[:, :, 0]
                truth_imgs = faces_as_images(truth)
                n_px += truth.size
                for m in methods:
                    pk = normalizer.decode(preds[m]).cpu().numpy()[:, :, 0]
                    sq[m] += float(((pk - truth) ** 2).sum())
                    spec[m].append(spectrum_log_l1(faces_as_images(pk), truth_imgs))
        entry = {m: {"rmse": float(np.sqrt(sq[m] / n_px)),
                     "spectrum_log_l1": float(np.mean(spec[m]))} for m in methods}
        out["ratios"][f"{ratio}x"] = entry
        tag = "  (native)" if ratio == native else ""
        # `cascade` is absent for a ratio finer than native (see methods above).
        print(f"  {ratio}x{tag}: "
              + "".join(f"{m} {entry[m]['rmse']:.4f} | " for m in methods)
              + units)

    path = Path(results_dir) / "zero_shot.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
