"""THE GATE EXPERIMENT: ensemble calibration versus Langevin-corrector steps.

Tests the branch's falsifiable question (docs/spectral_posterior_corrector.md):
does the spectrally preconditioned null-space corrector reach the same
calibration as the isotropic one in substantially fewer steps — and does
either fix the measured ~20% ensemble underdispersion?

Design: for each ratio, M base reconstructions per patch are computed ONCE
(the expensive part — the guided chain), then each corrector mode evolves its
own copy of those members step by step (1 network eval per step), snapshotting
ensemble metrics at the requested step checkpoints. Step 0 is the uncorrected
baseline, so the eta-sweep comparison point is built in.

Reads the spectral covariance artifact of data.estimate_spectral_covariance
for the spectral mode (weather_ddnm.covariance_file, as for Weather-DDNM).

Run:
    python -m eval.corrector_calibration --config config/t2m.yaml \
        --ckpt diffusion.pt --project --wandb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import coarsen  # noqa: E402
from eval.compare_geo import _payload  # noqa: E402
from eval.metrics import crps_ensemble, l2_norm  # noqa: E402
from sample.langevin_corrector import (check_consistency,  # noqa: E402
                                       langevin_correct, load_spectral_power)
from sample.reconstruct import load_diffusion, reconstruct_diffusion  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, run_name  # noqa: E402


def _ensemble_metrics(members_phys, hf_phys, n_members):
    stack = torch.stack(members_phys)
    ens_mean_l2 = l2_norm(stack.mean(0), hf_phys)
    return {
        "single_l2": float(np.mean([l2_norm(m, hf_phys) for m in members_phys])),
        "ens_mean_l2": ens_mean_l2,
        "crps": crps_ensemble(members_phys, hf_phys),
        "spread": float(stack.std(0).mean()),
        # A reliable M-member ensemble has spread ~= ens-mean RMSE / sqrt(1+1/M).
        "reliable_spread": ens_mean_l2 / float(np.sqrt(1.0 + 1.0 / n_members)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/t2m.yaml")
    ap.add_argument("--ckpt", default="diffusion.pt")
    ap.add_argument("--ensemble", type=int, default=4)
    ap.add_argument("--patches", type=int, default=64)
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[0, 1, 2, 4, 8, 16, 32],
                    help="Step checkpoints at which metrics are recorded; the "
                         "chain runs to max(steps).")
    ap.add_argument("--modes", nargs="+", default=["isotropic", "spectral"],
                    choices=["isotropic", "spectral"])
    ap.add_argument("--t-eps", type=int, default=50,
                    help="DDPM timestep of the corrector's score estimate.")
    ap.add_argument("--snr", type=float, default=0.16,
                    help="Song-style adaptive step-size signal-to-noise.")
    ap.add_argument("--delta", type=float, default=None,
                    help="Fixed step size overriding the snr rule.")
    ap.add_argument("--covariance", default=None,
                    help="Spectrum artifact for spectral mode; default "
                         "<patch_dir>/<weather_ddnm.covariance_file>.")
    ap.add_argument("--project", action="store_true",
                    help="Per-step DDNM projection in the BASE reconstruction "
                         "(match the setting of the numbers you compare to).")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()
    eta = cfg["sample"]["ddim_eta"]
    patch_dir = Path(cfg["paths"]["patch_dir"])
    results_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "corrector")
    normalizer = load_norm_stats(patch_dir)

    model, diffusion, cfg_ck = load_diffusion(
        Path(cfg["paths"]["ckpt_dir"]) / args.ckpt, device)
    geo_on = cfg_ck.get("geo", {}).get("enabled", False)

    ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    n = min(args.patches, len(ds))
    hf = torch.stack([ds[i] for i in range(n)]).to(device)
    hf_phys = normalizer.decode(hf.cpu())
    coords = (_payload(patch_dir, normalizer, n, cfg_ck["geo"], device)
              if geo_on else None)

    power = None
    if "spectral" in args.modes:
        cov_path = Path(args.covariance) if args.covariance else (
            patch_dir / cfg.get("weather_ddnm", {}).get(
                "covariance_file", "spectral_covariance.npz"))
        if not cov_path.exists():
            raise SystemExit(
                f"spectral mode needs {cov_path} — run "
                "data.estimate_spectral_covariance first (or drop 'spectral' "
                "from --modes)")
        power = load_spectral_power(cov_path, device)

    checkpoints = sorted(set(int(s) for s in args.steps))
    report = {"ckpt": args.ckpt, "modes": args.modes, "t_eps": args.t_eps,
              "snr": args.snr, "delta": args.delta, "ensemble": args.ensemble,
              "patches": n, "project": args.project, "results": {}}

    for rc in cfg["sample"]["reconstructions"]:
        ratio = rc["ratio"]
        tag = f"{ratio}x"
        coarse = coarsen(hf, ratio)

        # ── The expensive part, once per member, shared by every mode ──────
        base = []
        for m in range(args.ensemble):
            print(f"{tag}: base reconstruction member {m + 1}/{args.ensemble}")
            outs = []
            for i in range(0, n, args.batch):
                outs.append(reconstruct_diffusion(
                    diffusion, model, hf[i:i + args.batch], ratio, rc, eta=eta,
                    coords=None if coords is None else coords[i:i + args.batch],
                    project=args.project).cpu())
            base.append(torch.cat(outs))

        rows = {}
        for mode in args.modes:
            mode_power = power if mode == "spectral" else None
            states = [b.clone().to(device) for b in base]
            curve = {}
            done = 0
            for ck_step in checkpoints:
                for m in range(args.ensemble):
                    outs = []
                    for i in range(0, n, args.batch):
                        c = None if coords is None else coords[i:i + args.batch]
                        outs.append(langevin_correct(
                            model, diffusion, states[m][i:i + args.batch].to(device),
                            coarse[i:i + args.batch], ratio,
                            steps=ck_step - done, t_eps=args.t_eps,
                            snr=args.snr, delta=args.delta, cond=c,
                            power=mode_power).cpu())
                    states[m] = torch.cat(outs)
                done = ck_step
                members_phys = [normalizer.decode(s) for s in states]
                curve[ck_step] = _ensemble_metrics(members_phys, hf_phys,
                                                   args.ensemble)
                viol = max(check_consistency(s.to(device), coarse, ratio)
                           for s in states)
                curve[ck_step]["max_coarse_violation"] = viol
                r = curve[ck_step]
                print(f"  {tag} {mode:9s} step {ck_step:3d} | "
                      f"single L2 {r['single_l2']:.4f} | "
                      f"ens L2 {r['ens_mean_l2']:.4f} | "
                      f"CRPS {r['crps']:.4f} | spread {r['spread']:.4f} "
                      f"(reliable {r['reliable_spread']:.4f}) | "
                      f"coarse {viol:.1e}")
            rows[mode] = curve
        report["results"][tag] = rows

        # ── Figure: spread & CRPS versus corrector steps ───────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for mode in args.modes:
            steps = sorted(rows[mode])
            axes[0].plot(steps, [rows[mode][s]["spread"] for s in steps],
                         marker="o", label=f"{mode} spread")
            axes[0].plot(steps, [rows[mode][s]["reliable_spread"] for s in steps],
                         ls="--", alpha=0.6, label=f"{mode} reliable target")
            axes[1].plot(steps, [rows[mode][s]["crps"] for s in steps],
                         marker="o", label=mode)
        axes[0].set(xlabel="corrector steps", ylabel="ensemble spread (K)",
                    title=f"{tag} spread vs reliable target")
        axes[1].set(xlabel="corrector steps", ylabel="CRPS (K)",
                    title=f"{tag} CRPS")
        for ax in axes:
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig_path = results_dir / f"calibration_{tag}.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {fig_path}")

    out = results_dir / "corrector_calibration.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {out}")

    wb_run, wandb = init_wandb(
        cfg, job_type="corrector_calibration",
        extra_config={k: report[k] for k in
                      ("ckpt", "modes", "t_eps", "snr", "ensemble", "patches",
                       "project")},
        name=run_name(cfg, "corrector", Path(args.ckpt).stem,
                      *args.modes, f"t{args.t_eps}"))
    if wb_run is not None:
        for tag, rows in report["results"].items():
            for mode, curve in rows.items():
                for s, r in curve.items():
                    for k, v in r.items():
                        wb_run.summary[f"{tag}/{mode}/step{s}/{k}"] = v
            fig_path = results_dir / f"calibration_{tag}.png"
            if fig_path.exists():
                wb_run.log({f"corrector/{tag}": wandb.Image(str(fig_path))})
        wb_run.finish()


if __name__ == "__main__":
    main()
