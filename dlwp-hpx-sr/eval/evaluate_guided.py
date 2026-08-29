"""Guided reconstruction from the unconditional mesh prior, at every ratio.

The prior (train/train_prior.py) never saw a degradation ratio. This script
reconstructs the test fields at each ratio in `sample.reconstructions` from the
SAME checkpoint, using noise-mixing + intermediate-start DDIM with the exact
mesh data-consistency projection, and scores it against seam-aware bilinear and
the degraded input.

The ratios beyond the ones the sibling project tuned (16x) are the point of the
exercise: a ratio-agnostic prior should degrade gracefully where the
ratio-specific regressors collapse (measured at 8x: patch direct map 1.23, mesh
regressor 1.26, vs bicubic 0.96, vs patch guided diffusion 0.77).

All metrics are computed on the mesh, where pixels are equal-area, so nothing
is latitude-weighted. `data_consistency_max_abs` reports
max |coarsen(pred) - lf|, which the exact projection should hold at ~0.

Run:
    python -m eval.evaluate_guided --config config/prior.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import HPXDataset, Normalizer  # noqa: E402
from data.degrade import (coarsen_faces, degrade_faces,  # noqa: E402
                          upsample_bilinear_faces)
from eval.metrics import faces_as_images, spectrum_log_l1  # noqa: E402
from hpx.mesh import nest_to_faces, pixel_lonlat_deg  # noqa: E402
from hpx.padding import HEALPixPadding  # noqa: E402
from models.hpx_diffusion import build_diffusion  # noqa: E402
from train.train_prior import build_prior_model  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/prior.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--ratios", type=int, nargs="+", default=None,
                    help="subset of the configured ratios to run")
    ap.add_argument("--n-test-samples", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None,
                    help="DDIM timestep stride; >1 trades accuracy for speed")
    ap.add_argument("--no-project", action="store_true",
                    help="ablate the exact mesh projection (expect a large loss:"
                         " with an unconditional prior it is the ONLY link to"
                         " the observation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    sc, ec = cfg["sample"], cfg["eval"]
    nside = int(cfg["hpx"]["nside"])
    units = cfg["data"].get("units", "phys")
    stride = args.stride or sc.get("stride", 1)
    project = (not args.no_project) and sc.get("project", True)
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / ec["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = Normalizer(ckpt["norm_mean"], ckpt["norm_std"])
    model = build_prior_model(ckpt["config"]).to(device)
    use_ema = bool(ec.get("use_ema", True) and ckpt.get("ema"))
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    diffusion = build_diffusion(ckpt["config"]).to(device)
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val eps mse "
          f"{ckpt['val_loss']:.5f}, {'EMA' if use_ema else 'raw'} weights)")

    ds = HPXDataset(hpx_dir / "test.npy", normalizer)
    n = min(args.n_test_samples or ec["n_test_samples"] or len(ds), len(ds))
    recons = [r for r in sc["reconstructions"]
              if args.ratios is None or int(r["ratio"]) in args.ratios]
    # spread over the test period: the first N fields are one contiguous
    # winter month, which biases absolute scores by several percent
    sel = np.linspace(0, len(ds) - 1, n).astype(int)
    print(f"{n}/{len(ds)} test fields (spread over the test period) | ratios "
          f"{[r['ratio'] for r in recons]} | project {project} | stride {stride}")

    _, plat = pixel_lonlat_deg(nside)
    lat_faces = nest_to_faces(plat, nside)
    band_masks = {f"{lo}..{hi}": (lat_faces >= lo) & (lat_faces < hi)
                  for lo, hi in ec["lat_bands"]}

    out = {"checkpoint": str(ckpt_path), "epoch": int(ckpt["epoch"]),
           "n_samples": n, "nside": nside, "units": units,
           "project": project, "stride": stride, "ratios": {}}
    batch = int(ec.get("batch_size", 2))

    for rc in recons:
        ratio, K, t_steps = int(rc["ratio"]), int(rc["K"]), list(rc["t_steps"])
        padder1 = HEALPixPadding(nside // ratio, 1).to(device)
        methods = ("guided", "bilinear", "nearest")
        sq = {m: 0.0 for m in methods}
        ab = {m: 0.0 for m in methods}
        bias = {m: 0.0 for m in methods}
        band_sq = {m: {b: 0.0 for b in band_masks} for m in methods}
        band_n = {b: 0.0 for b in band_masks}
        spec = {m: [] for m in methods}
        n_px = 0
        dc_max = 0.0
        t_sample = 0.0
        print(f"\n=== ratio {ratio}x | K={K} t_steps={t_steps} "
              f"({K * sum(t_steps) // max(stride,1) // K if K else 0} steps/loop avg) ===")

        with torch.no_grad():
            for start in range(0, n, batch):
                idx = sel[start:start + batch]
                y = torch.stack([ds[int(i)] for i in idx]).to(device)
                lf = coarsen_faces(y, ratio)
                x_g = degrade_faces(y, ratio)      # nearest-upsampled guidance

                t0 = time.time()
                g = torch.Generator(device=device).manual_seed(args.seed + start)
                pred = diffusion.guided_reconstruct(
                    model, x_g, t_steps=t_steps, K=K, eta=sc.get("eta", 0.0),
                    stride=stride, project=project, lf=lf, ratio=ratio,
                    generator=g)
                t_sample += time.time() - t0

                dc_max = max(dc_max,
                             float((coarsen_faces(pred, ratio) - lf).abs().max()))
                preds = {"guided": pred,
                         "bilinear": upsample_bilinear_faces(lf, ratio, padder1),
                         "nearest": x_g}
                truth_K = normalizer.decode(y).cpu().numpy()[:, :, 0]
                truth_imgs = faces_as_images(truth_K)
                n_px += truth_K.size
                for b, mask in band_masks.items():
                    band_n[b] += truth_K.shape[0] * int(mask.sum())
                for m in methods:
                    pk = normalizer.decode(preds[m]).cpu().numpy()[:, :, 0]
                    err = pk - truth_K
                    sq[m] += float((err ** 2).sum())
                    ab[m] += float(np.abs(err).sum())
                    bias[m] += float(err.sum())
                    for b, mask in band_masks.items():
                        band_sq[m][b] += float((err[..., mask] ** 2).sum())
                    spec[m].append(spectrum_log_l1(faces_as_images(pk), truth_imgs))
                print(f"  {min(start + batch, n)}/{n} fields "
                      f"({t_sample / max(min(start + batch, n), 1):.1f}s/field)",
                      flush=True)

        entry = {"K": K, "t_steps": t_steps,
                 "seconds_per_field": t_sample / max(n, 1),
                 "data_consistency_max_abs": dc_max}
        for m in methods:
            entry[m] = {
                "rmse": float(np.sqrt(sq[m] / n_px)),
                "mae": ab[m] / n_px,
                "bias": bias[m] / n_px,
                "spectrum_log_l1": float(np.mean(spec[m])),
                "rmse_by_band": {b: float(np.sqrt(band_sq[m][b] / band_n[b]))
                                 for b in band_masks},
            }
        out["ratios"][f"{ratio}x"] = entry
        print(f"  guided   rmse {entry['guided']['rmse']:.4f} {units} | "
              f"spec {entry['guided']['spectrum_log_l1']:.4f}")
        print(f"  bilinear rmse {entry['bilinear']['rmse']:.4f} {units} | "
              f"spec {entry['bilinear']['spectrum_log_l1']:.4f}")
        print(f"  consistency max|coarsen(pred)-lf| = {dc_max:.2e}")

    path = Path(results_dir) / "metrics_guided.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")

    wb_run, _ = init_wandb(cfg, job_type="eval_guided",
                           extra_config={"n_eval": n, "ckpt": str(ckpt_path)})
    if wb_run is not None:
        wb_run.log({f"{r}/{m}/rmse": out["ratios"][r][m]["rmse"]
                    for r in out["ratios"] for m in ("guided", "bilinear", "nearest")})
        wb_run.finish()


if __name__ == "__main__":
    main()
