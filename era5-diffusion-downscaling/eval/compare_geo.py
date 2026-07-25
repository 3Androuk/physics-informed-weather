"""Head-to-head comparison of two guided-diffusion checkpoints.

Originally geo-vs-baseline; now generic: EITHER checkpoint may be
geo-conditioned (hash or healpix). Each model receives the geo payload its own
config requires (hash coords vs healpix indices), built from the same test
origins, so hash-geo vs healpix-geo, geo vs plain, or seed-replicate pairs all
compare on identical test patches at every ratio, alongside bicubic. Reports
L2 (RMSE) and the power-spectrum metric; optional per-step DDNM projection and
ensemble metrics (ensemble-mean L2, CRPS, spread).

Run:
    python -m eval.compare_geo --config config/t2m.yaml --project \
        --geo-ckpt diffusion_geo_hpx.pt --base-ckpt diffusion_geo.pt
"""

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
from eval.metrics import radial_power_spectrum, spectrum_log_l1, l2_norm  # noqa: E402
from sample.reconstruct import (load_diffusion, reconstruct_bicubic,  # noqa: E402
                                reconstruct_diffusion)
from utils import ensure_dir, get_device, init_wandb, load_config, run_name  # noqa: E402


def _recon(diffusion, model, hf, ratio, rc, eta, coords, batch, label="recon",
           project=False):
    it = range(0, len(hf), batch)
    try:
        from tqdm import tqdm
        it = tqdm(it, desc=label)
    except ImportError:
        pass
    outs = []
    for i in it:
        c = None if coords is None else coords[i:i + batch]
        outs.append(reconstruct_diffusion(diffusion, model, hf[i:i + batch], ratio, rc,
                                          eta=eta, coords=c, project=project).cpu())
    return torch.cat(outs, dim=0)


def _payload(patch_dir, normalizer, n, geo_cfg, device):
    """Geo payload stack for the first n test patches, per one model's config."""
    ds = PatchDataset(
        patch_dir / "test_patches.npy", normalizer,
        origins_path=patch_dir / "test_origins.npy",
        coords_full_path=patch_dir / "coords_full.npz",
        geo_input_dim=geo_cfg["input_dim"], altitude=geo_cfg["altitude"],
        geo_encoder=geo_cfg.get("encoder", "hash"),
    )
    return torch.stack([ds[i][1] for i in range(n)]).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--geo-ckpt", default="diffusion_geo.pt",
                    help="model A checkpoint (may be geo or plain)")
    ap.add_argument("--base-ckpt", default="diffusion.pt",
                    help="model B checkpoint (may be geo or plain)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--project", action="store_true",
                    help="Per-step data-consistency projection: coarsen(x0) == LF "
                         "enforced at every DDIM step.")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="Ensemble members per patch (>1 adds ensemble-mean L2, "
                         "CRPS, and spread on a subset of patches).")
    ap.add_argument("--ensemble-patches", type=int, default=64,
                    help="How many test patches the ensemble metrics use.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()
    eta = cfg["sample"]["ddim_eta"]

    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)
    n = min(cfg["eval"]["n_test_patches"],
            len(PatchDataset(patch_dir / "test_patches.npy", normalizer)))

    name_a = Path(args.geo_ckpt).stem
    name_b = Path(args.base_ckpt).stem
    model_a, diff_a, cfg_a = load_diffusion(ckpt_dir / args.geo_ckpt, device)
    model_b, diff_b, cfg_b = load_diffusion(ckpt_dir / args.base_ckpt, device)
    geo_a = cfg_a.get("geo", {}).get("enabled", False)
    geo_b = cfg_b.get("geo", {}).get("enabled", False)
    print(f"Comparing {name_a} (geo={geo_a}) vs {name_b} (geo={geo_b}) on {n} patches"
          f"{' | projection ON' if args.project else ''}")

    ds_plain = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    hf = torch.stack([ds_plain[i] for i in range(n)]).to(device)
    hf_phys = normalizer.decode(hf.cpu())
    coords_a = _payload(patch_dir, normalizer, n, cfg_a["geo"], device) if geo_a else None
    coords_b = _payload(patch_dir, normalizer, n, cfg_b["geo"], device) if geo_b else None

    table, spectra = {}, {"Reference": radial_power_spectrum(hf_phys)}
    for rc in cfg["sample"]["reconstructions"]:
        ratio = rc["ratio"]; tag = f"{ratio}x"
        preds = {
            name_a: _recon(diff_a, model_a, hf, ratio, rc, eta, coords_a, args.batch,
                           label=f"{tag} {name_a}", project=args.project),
            name_b: _recon(diff_b, model_b, hf, ratio, rc, eta, coords_b, args.batch,
                           label=f"{tag} {name_b}", project=args.project),
            "Bicubic": torch.cat([reconstruct_bicubic(hf[i:i + args.batch], ratio).cpu()
                                  for i in range(0, len(hf), args.batch)]),
        }
        row = {}
        for name, p in preds.items():
            pp = normalizer.decode(p)
            row[name] = {"l2": l2_norm(pp, hf_phys), "spectrum_log_l1": spectrum_log_l1(pp, hf_phys)}
            spectra[f"{name} {tag}"] = radial_power_spectrum(pp)
            print(f"  {tag} {name:28s} | L2 {row[name]['l2']:.4f} | "
                  f"spec-logL1 {row[name]['spectrum_log_l1']:.4f}")
        table[tag] = row
        _qualitative(normalizer, hf, preds, ratio, rc,
                     results_dir / f"geo_ablation_qualitative_{tag}.png")

    # ── Ensemble metrics (subset of patches; diffusion methods only) ──────
    if args.ensemble > 1:
        from eval.metrics import crps_ensemble
        n_e = min(args.ensemble_patches, len(hf))
        hf_e, hf_e_phys = hf[:n_e], hf_phys[:n_e]
        print(f"\nEnsemble metrics: {args.ensemble} members x {n_e} patches")
        ens = {}
        for rc in cfg["sample"]["reconstructions"]:
            ratio = rc["ratio"]; tag = f"{ratio}x"
            for name, (dif, mod, c) in {
                    name_a: (diff_a, model_a, None if coords_a is None else coords_a[:n_e]),
                    name_b: (diff_b, model_b, None if coords_b is None else coords_b[:n_e])}.items():
                members = [normalizer.decode(
                    _recon(dif, mod, hf_e, ratio, rc, eta, c, args.batch,
                           label=f"{tag} {name} member {m + 1}/{args.ensemble}",
                           project=args.project))
                    for m in range(args.ensemble)]
                stack = torch.stack(members)
                row = {
                    "single_l2": float(np.mean([l2_norm(p, hf_e_phys) for p in members])),
                    "ensemble_mean_l2": l2_norm(stack.mean(0), hf_e_phys),
                    "crps": crps_ensemble(members, hf_e_phys),
                    "spread": float(stack.std(0).mean()),
                }
                ens.setdefault(tag, {})[name] = row
                print(f"  {tag} {name:28s} | single L2 {row['single_l2']:.4f} | "
                      f"ens-mean L2 {row['ensemble_mean_l2']:.4f} | "
                      f"CRPS {row['crps']:.4f} | spread {row['spread']:.4f}")
        table["ensemble"] = ens

    with open(results_dir / "geo_ablation.json", "w") as f:
        json.dump(table, f, indent=2)
    _plot(spectra, results_dir / "geo_ablation_spectrum.png")
    print(f"\nSaved -> {results_dir / 'geo_ablation.json'}, geo_ablation_spectrum.png, "
          f"and geo_ablation_qualitative_*.png")

    wb_run, wandb = init_wandb(cfg, job_type="compare_geo",
                               extra_config={"n_test_patches": n,
                                             "ckpt_a": args.geo_ckpt,
                                             "ckpt_b": args.base_ckpt,
                                             "projection": args.project,
                                             "ensemble": args.ensemble},
                               name=run_name(cfg, "ablation", name_a, "vs", name_b,
                                             "proj" if args.project else "",
                                             f"ens{args.ensemble}" if args.ensemble > 1 else ""))
    if wb_run is not None:
        # Scalars go to the run SUMMARY (columns in the runs table), not log():
        # a one-shot eval otherwise creates one single-point chart per metric.
        tbl = wandb.Table(columns=["ratio", "method", "l2", "spectrum_log_l1"])
        log = {}
        for tag, row in table.items():
            if tag == "ensemble":
                for etag, erow in row.items():
                    for method, v in erow.items():
                        for mk, mv in v.items():
                            wb_run.summary[f"ensemble/{etag}/{method}/{mk}"] = mv
                continue
            for method, v in row.items():
                tbl.add_data(tag, method, v["l2"], v["spectrum_log_l1"])
                wb_run.summary[f"{tag}/{method}/l2"] = v["l2"]
                wb_run.summary[f"{tag}/{method}/spectrum_log_l1"] = v["spectrum_log_l1"]
        log["ablation/table"] = tbl
        log["ablation/spectrum"] = wandb.Image(str(results_dir / "geo_ablation_spectrum.png"))
        for rc in cfg["sample"]["reconstructions"]:
            q = results_dir / f"geo_ablation_qualitative_{rc['ratio']}x.png"
            if q.exists():
                log[f"ablation/qualitative_{rc['ratio']}x"] = wandb.Image(str(q))
        wb_run.log(log)
        wb_run.finish()
        print("wandb: ablation run logged")


def _qualitative(normalizer, hf, preds, ratio, rc, path, idx=0):
    """Side-by-side panels on a SHARED color scale (taken from the reference),
    so residual noise or bias shows as a visible difference instead of being
    hidden by per-panel autoscaling."""
    from data.degrade import degrade
    lf = degrade(hf[idx:idx + 1].cpu(), ratio, rc.get("smooth_sigma", 0.0))
    panels = [("Input (LF)", lf)]
    for name, p in preds.items():
        panels.append((name, p[idx:idx + 1]))
    panels.append(("Reference", hf[idx:idx + 1].cpu()))
    ref = normalizer.decode(hf[idx:idx + 1].cpu())[0, 0].numpy()
    vmin, vmax = float(ref.min()), float(ref.max())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (title, t) in zip(axes, panels):
        ax.imshow(normalizer.decode(t.cpu())[0, 0].numpy(), cmap="RdBu_r",
                  vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{ratio}x reconstruction (shared color scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot(spectra, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, e) in spectra.items():
        ax.loglog(k[1:], e[1:], "-" if label == "Reference" else "--",
                  lw=2.2 if label == "Reference" else 1.4, label=label)
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)")
    ax.set_title("Checkpoint comparison: power spectrum")
    ax.legend(fontsize=7); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
