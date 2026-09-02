"""Deployment demo: downscale coarse FORECAST fields, score against ERA5.

Consumes data/download_forecast.py output. For each init at one lead time it
tiles the full band field through the guided sampler (per-step projection on
OBSERVED channels only; unobserved channels are generated — channel
inpainting), and reports, per channel:

  * RMSE (physical units) + cos(lat)-weighted RMSE, vs bicubic baseline
  * CRPS + spread when --ensemble > 1 (members from sampler noise, plus
    forecast realizations when the file carries them)
  * spectrum log-L1 on the display channel

--control additionally downscales COARSENED TRUTH with the same seeds — the
gap between the control and the forecast run isolates what forecast error
costs on top of downscaling error (the number the "cheaper and still
accurate" claim needs).

Run:
    python -m eval.downscale_forecast --config config/wb2_20var.yaml \
        --ckpt diffusion_geo_static.pt --data-dir datasets/forecast_hres \
        --lead 120 --limit 20 --ensemble 4 --control
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import load_norm_stats  # noqa: E402
from data.degrade import coarsen, upsample_nearest  # noqa: E402
from eval.metrics import crps_ensemble, spectrum_log_l1  # noqa: E402
from sample.full_field import reconstruct_full_tiled_diffusion  # noqa: E402
from sample.reconstruct import load_diffusion  # noqa: E402
from utils import ensure_dir, get_device, load_config  # noqa: E402


def rmse_per_channel(pred: torch.Tensor, truth: torch.Tensor,
                     lat: np.ndarray = None) -> list:
    """Per-channel RMSE over (N, C, H, W); optionally cos(lat)-row-weighted."""
    sq = (pred - truth).double().pow(2).mean(dim=-1)          # (N, C, H)
    if lat is not None:
        w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
        w = torch.as_tensor(w / w.mean(), device=sq.device)
        sq = sq * w
    return [float(sq[:, c].mean(dim=-1).sqrt().mean()) for c in range(sq.shape[1])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--ckpt", default="diffusion.pt")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lead", type=int, required=True, help="lead in hours")
    ap.add_argument("--limit", type=int, default=None, help="max inits")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="sampler members per input (eta>0 or distinct init "
                         "noise diversifies them)")
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--batch-tiles", type=int, default=8)
    ap.add_argument("--obs-noise", default="0",
                    help="DDNM+ observation noise sigma_y in NORMALIZED "
                         "units; 0 = plain DDNM (exact consistency). "
                         "'auto' measures RMS(coarsen(truth) - fcst) "
                         "at this lead — the coarse-scale forecast error.")
    ap.add_argument("--no-project", action="store_true",
                    help="drop DDNM entirely: the coarse field steers the\n"
                         "chain via noise-mixing but is never enforced. On a\n"
                         "NOISY observation this stops forecast error being\n"
                         "reproduced exactly (lambda = 0 in the DDNM+ family).")
    ap.add_argument("--control", action="store_true",
                    help="also downscale coarsened truth with the same seeds "
                         "(isolates forecast error from downscaling error)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    data_dir = Path(args.data_dir)
    meta = np.load(data_dir / f"meta_{args.lead}h.npz")
    obs = torch.as_tensor(meta["observed"], dtype=torch.bool)
    ratio = int(meta["ratio"][0])
    labels = [str(x) for x in meta["channels"]]
    lat_fine = meta["lat_fine"]
    eta = cfg["sample"]["ddim_eta"] if args.eta is None else args.eta
    rc = next((r for r in cfg["sample"]["reconstructions"]
               if r["ratio"] == ratio),
              dict(cfg["sample"]["reconstructions"][-1], ratio=ratio))

    fc = np.load(data_dir / f"fcst_{args.lead}h.npy", mmap_mode="r")
    tr = np.load(data_dir / f"truth_{args.lead}h.npy", mmap_mode="r")
    n = len(fc) if args.limit is None else min(args.limit, len(fc))
    n_real = fc.shape[1]

    patch_dir = Path(cfg["paths"]["patch_dir"])
    normalizer = load_norm_stats(patch_dir)
    model, diffusion, cfg_ck = load_diffusion(
        Path(cfg["paths"]["ckpt_dir"]) / args.ckpt, device)
    geo_full = None
    if cfg_ck.get("geo", {}).get("enabled", False):
        from eval.full_field import _geo_full
        geo_full = _geo_full(cfg_ck, patch_dir, tuple(tr.shape[-2:]), device)

    print(f"{args.lead}h: {n} inits x {n_real} forecast member(s) x "
          f"{args.ensemble} sampler member(s); ratio {ratio}; "
          f"observed {int(obs.sum())}/{len(obs)}; eta {eta:g}"
          f"{'; control ON' if args.control else ''}")

    # ---- DDNM+ observation noise -------------------------------------
    # y here is a FORECAST, not a noiseless block-average of the truth, so
    # exact consistency would imprint forecast error. sigma_y is measured in
    # NORMALIZED units as the coarse-scale forecast error at this lead.
    if str(args.obs_noise).lower() == "auto":
        k = min(n, 16)
        errs = []
        for j in range(k):
            t_n = normalizer.encode(torch.from_numpy(np.array(tr[j:j + 1])).float())
            c_n = normalizer.encode(torch.from_numpy(np.array(fc[j, 0][None])).float())
            errs.append((coarsen(t_n, ratio) - c_n).pow(2).mean().item())
        obs_noise = float(np.sqrt(np.mean(errs)))
        print(f"  DDNM+ sigma_y (auto, {k} inits) = {obs_noise:.4f} normalized units")
    else:
        obs_noise = float(args.obs_noise)
        if obs_noise > 0:
            print(f"  DDNM+ sigma_y = {obs_noise:.4f} normalized units")

    def run(coarse_norm, seed):
        lf = upsample_nearest(coarse_norm, tuple(tr.shape[-2:]))
        # CPU generator: sample/full_field._global_noise draws the shared
        # global noise on CPU so a seed is device-independent, and torch
        # requires the generator's device to match the creation device.
        gen = torch.Generator().manual_seed(seed)
        return reconstruct_full_tiled_diffusion(
            diffusion, model, lf, coarse_norm, ratio, rc, eta=eta,
            tile=args.tile, overlap=args.overlap, batch=args.batch_tiles,
            geo_full=geo_full, project_steps=not args.no_project,
            project_final=not args.no_project, generator=gen,
            obs_noise=obs_noise,
            observed=obs).cpu()

    sums = {k: None for k in ("fcst", "bicubic", "control")}
    counts = 0
    crps_members, crps_truth = [], []
    spec_pred, spec_truth = [], []
    disp = 0
    for i in range(n):
        truth = torch.from_numpy(np.array(tr[i:i + 1])).float()
        truth_n = normalizer.encode(truth)
        members = []
        for r in range(n_real):
            coarse = torch.from_numpy(np.array(fc[i, r][None])).float()
            coarse_n = normalizer.encode(coarse).to(device)
            for e in range(args.ensemble):
                out_n = run(coarse_n, seed=i * 1000 + r * 100 + e)
                members.append(normalizer.decode(out_n))
        stack = torch.stack(members)                       # (M, 1, C, H, W)
        pred = stack.mean(0) if len(members) > 1 else stack[0]

        bic = normalizer.decode(torch.nn.functional.interpolate(
            normalizer.encode(torch.from_numpy(np.array(fc[i, 0][None])).float()),
            size=tuple(tr.shape[-2:]), mode="bicubic", align_corners=False))

        rows = {"fcst": rmse_per_channel(pred, truth, lat_fine),
                "bicubic": rmse_per_channel(bic, truth, lat_fine)}
        if args.control:
            ctrl_coarse = coarsen(truth_n, ratio).to(device)
            ctrl = normalizer.decode(run(ctrl_coarse, seed=i * 1000))
            rows["control"] = rmse_per_channel(ctrl, truth, lat_fine)
        for k, v in rows.items():
            sums[k] = np.array(v) if sums[k] is None else sums[k] + np.array(v)
        counts += 1
        if len(members) > 1:
            crps_members.append([m[:, disp] for m in members])
            crps_truth.append(truth[:, disp])
        spec_pred.append(pred[:, disp]); spec_truth.append(truth[:, disp])
        print(f"  init {i + 1}/{n} done", flush=True)

    out = {"lead_h": args.lead, "n_inits": counts, "ckpt": args.ckpt,
           "ratio": ratio, "eta": eta, "ensemble": args.ensemble,
           "observed": {l: bool(o) for l, o in zip(labels, obs)},
           "rmse_latweighted": {}}
    for k, s in sums.items():
        if s is not None:
            out["rmse_latweighted"][k] = dict(zip(labels, (s / counts).tolist()))
    out["spectrum_log_l1_display"] = spectrum_log_l1(
        torch.cat(spec_pred), torch.cat(spec_truth))
    if crps_members:
        out["crps_display"] = float(np.mean([
            crps_ensemble(ms, t) for ms, t in zip(crps_members, crps_truth)]))

    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    # encode the consistency arm, or arms overwrite each other
    arm = ("noproj" if args.no_project
           else "ddnm" if obs_noise <= 0 else f"ddnmplus{obs_noise:.3f}")
    stem = f"forecast_{args.lead}h_{Path(args.ckpt).stem}_{arm}"
    with open(results_dir / f"{stem}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["rmse_latweighted"], indent=2)[:2000])
    print(f"saved -> {results_dir / stem}.json")


if __name__ == "__main__":
    main()
