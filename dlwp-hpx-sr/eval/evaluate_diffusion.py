"""Evaluate the sphere-native residual diffusion model on the mesh.

Scores, on the test split, in physical units and on the equal-area mesh (so no
latitude weighting anywhere):

    diffusion  — the sampled reconstruction (mean field + residual chain)
    mean       — the frozen deterministic mean alone (the regressor's own score)
    bilinear   — seam-aware bilinear upsampling
    nearest    — the degraded input

RMSE / MAE / bias / per-latitude-band RMSE, plus the per-face radial power
spectrum log-L1 (eval/metrics.py, verbatim from the sibling project) — the
metric the deterministic model loses on, and the reason this model exists.

Also reports the data-consistency residual max |coarsen(pred) - lf|: with
eval.project it should be ~0 to floating-point, the exactness the mesh buys.

Run:
    python -m eval.evaluate_diffusion --config config/diffusion.yaml
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
from models.hpx_residual import build_residual_model, load_mean_field  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/diffusion.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--n-steps", type=int, default=None, help="override eval.n_steps")
    ap.add_argument("--n-test-samples", type=int, default=None)
    ap.add_argument("--ensemble", type=int, default=None)
    ap.add_argument("--no-project", action="store_true",
                    help="ablate the exact mesh data-consistency projection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    ec = cfg["eval"]
    ratio = int(cfg["sr"]["ratio"])
    nside = int(cfg["hpx"]["nside"])
    units = cfg["data"].get("units", "phys")
    n_steps = args.n_steps or ec["n_steps"]
    n_members = args.ensemble or ec.get("ensemble", 1)
    project = (not args.no_project) and ec.get("project", True)
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / ec["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = Normalizer(ckpt["norm_mean"], ckpt["norm_std"])
    model = build_residual_model(ckpt["config"]).to(device)
    use_ema = bool(ec.get("use_ema", True) and ckpt.get("ema"))
    model.load_state_dict(ckpt["ema"] if use_ema else ckpt["model"])
    model.eval()
    diffusion = build_diffusion(ckpt["config"]).to(device)
    mean_field = load_mean_field(ckpt["mean_kind"], ratio, nside,
                                 ckpt.get("mean_ckpt") or None, device)
    # must match training exactly, or the composed field is off by this factor
    res_scale = float(ckpt.get("residual_scale", 1.0))
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, val eps mse "
          f"{ckpt['val_loss']:.5f}, {'EMA' if use_ema else 'raw'} weights) "
          f"| mean: {ckpt['mean_kind']}")

    ds = HPXDataset(hpx_dir / "test.npy", normalizer)
    n = len(ds) if ec["n_test_samples"] is None else min(ec["n_test_samples"], len(ds))
    if args.n_test_samples:
        n = min(args.n_test_samples, len(ds))
    print(f"Sampling {n}/{len(ds)} test fields | {n_steps} DDIM steps | "
          f"eta {ec['eta']} | project {project} | ensemble {n_members} | "
          f"residual scale {res_scale:.5f}")

    _, plat = pixel_lonlat_deg(nside)
    lat_faces = nest_to_faces(plat, nside)
    band_masks = {f"{lo}..{hi}": (lat_faces >= lo) & (lat_faces < hi)
                  for lo, hi in ec["lat_bands"]}

    padder1 = HEALPixPadding(nside // ratio, 1).to(device)
    methods = ("diffusion", "mean", "bilinear", "nearest")
    sq = {m: 0.0 for m in methods}
    ab = {m: 0.0 for m in methods}
    bias = {m: 0.0 for m in methods}
    band_sq = {m: {b: 0.0 for b in band_masks} for m in methods}
    band_n = {b: 0.0 for b in band_masks}
    spec = {m: [] for m in methods}
    n_px = 0
    dc_max = 0.0
    spread_sum = 0.0
    t_sample = 0.0

    batch = int(ec.get("batch_size", 2))
    with torch.no_grad():
        for start in range(0, n, batch):
            idx = range(start, min(start + batch, n))
            y = torch.stack([ds[i] for i in idx]).to(device)   # (B,12,1,F,F)
            lf = coarsen_faces(y, ratio)
            x_up = degrade_faces(y, ratio)
            mean = mean_field(x_up)

            t0 = time.time()
            members = []
            for m_i in range(n_members):
                g = torch.Generator(device=device).manual_seed(
                    args.seed + 1000 * m_i + start)
                members.append(diffusion.sample(
                    model, mean, lf, ratio, n_steps=n_steps, eta=ec["eta"],
                    project=project, generator=g, residual_scale=res_scale))
            t_sample += time.time() - t0
            stack = torch.stack(members)                        # (M,B,12,1,F,F)
            pred = stack.mean(0)
            if n_members > 1:
                spread_sum += float(stack.std(0).mean()) * len(idx) * normalizer.std

            dc_max = max(dc_max, float((coarsen_faces(pred, ratio) - lf).abs().max()))

            preds = {"diffusion": pred, "mean": mean,
                     "bilinear": upsample_bilinear_faces(lf, ratio, padder1),
                     "nearest": x_up}
            truth_K = normalizer.decode(y).cpu().numpy()[:, :, 0]
            n_px += truth_K.size
            for b, mask in band_masks.items():
                band_n[b] += truth_K.shape[0] * int(mask.sum())
            truth_imgs = faces_as_images(truth_K)
            for m in methods:
                pk = normalizer.decode(preds[m]).cpu().numpy()[:, :, 0]
                err = pk - truth_K
                sq[m] += float((err ** 2).sum())
                ab[m] += float(np.abs(err).sum())
                bias[m] += float(err.sum())
                for b, mask in band_masks.items():
                    band_sq[m][b] += float((err[..., mask] ** 2).sum())
                spec[m].append(spectrum_log_l1(faces_as_images(pk), truth_imgs))
            print(f"  {min(start + batch, n)}/{n} fields", flush=True)

    metrics = {"n_samples": n, "ratio": ratio, "nside": nside,
               "units": units, "n_steps": n_steps, "eta": ec["eta"],
               "project": project, "ensemble": n_members,
               "mean_kind": ckpt["mean_kind"], "residual_scale": res_scale,
               "seconds_per_field": t_sample / max(n, 1),
               "data_consistency_max_abs": dc_max}
    if n_members > 1:
        metrics["ensemble_spread"] = spread_sum / max(n, 1)
    for m in methods:
        metrics[m] = {
            "rmse": float(np.sqrt(sq[m] / n_px)),
            "mae": ab[m] / n_px,
            "bias": bias[m] / n_px,
            "spectrum_log_l1": float(np.mean(spec[m])),
            "rmse_by_band": {b: float(np.sqrt(band_sq[m][b] / band_n[b]))
                             for b in band_masks},
        }
    with open(Path(results_dir) / "metrics_diffusion.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k in (*methods, "seconds_per_field",
                               "data_consistency_max_abs")}, indent=2))

    wb_run, _ = init_wandb(cfg, job_type="eval_diffusion",
                           extra_config={"n_eval": n, "ckpt": str(ckpt_path)})
    if wb_run is not None:
        wb_run.log({f"{m}/{k}": v for m in methods
                    for k, v in metrics[m].items() if not isinstance(v, dict)})
        wb_run.finish()
    print(f"-> {results_dir}/metrics_diffusion.json")


if __name__ == "__main__":
    main()
