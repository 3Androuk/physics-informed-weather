"""Geo-on vs geo-off ablation for the diffusion downscaler.

Loads two diffusion checkpoints — geo-conditioned (hash-grid location embedding)
and the plain baseline — and evaluates both on the SAME test patches at every
ratio, alongside bicubic. Reports L2 (RMSE) and a power-spectrum metric so you
can see whether the geographic conditioning helps, and where (in-distribution 4x
vs out-of-distribution 8x).

Run (after training both models):
    python -m train.train_diffusion  --config config/default.yaml        # diffusion.pt
    # set geo.enabled: true in the config, then:
    python -m train.train_diffusion  --config config/default.yaml        # diffusion_geo.pt
    python -m eval.compare_geo        --config config/default.yaml
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
from utils import ensure_dir, get_device, init_wandb, load_config  # noqa: E402


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


def _load_test(patch_dir, normalizer, n, geo_cfg, device):
    """Return (hf_norm, hf_phys, coords-or-None). coords built if geo_cfg given."""
    if geo_cfg is not None:
        ds = PatchDataset(
            patch_dir / "test_patches.npy", normalizer,
            origins_path=patch_dir / "test_origins.npy",
            coords_full_path=patch_dir / "coords_full.npz",
            geo_input_dim=geo_cfg["input_dim"], altitude=geo_cfg["altitude"],
        )
        items = [ds[i] for i in range(n)]
        hf = torch.stack([it[0] for it in items]).to(device)
        coords = torch.stack([it[1] for it in items]).to(device)
        return hf, normalizer.decode(hf.cpu()), coords
    ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    hf = torch.stack([ds[i] for i in range(n)]).to(device)
    return hf, normalizer.decode(hf.cpu()), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--geo-ckpt", default="diffusion_geo.pt")
    ap.add_argument("--base-ckpt", default="diffusion.pt")
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

    geo_model, geo_diff, geo_cfg = load_diffusion(ckpt_dir / args.geo_ckpt, device)
    base_model, base_diff, _ = load_diffusion(ckpt_dir / args.base_ckpt, device)
    assert geo_cfg["geo"]["enabled"], f"{args.geo_ckpt} is not a geo checkpoint"
    print(f"Comparing geo ({args.geo_ckpt}) vs baseline ({args.base_ckpt}) on {n} patches"
          f"{' | projection ON' if args.project else ''}")

    hf, hf_phys, coords = _load_test(patch_dir, normalizer, n, geo_cfg["geo"], device)

    table, spectra = {}, {"Reference": radial_power_spectrum(hf_phys)}
    for rc in cfg["sample"]["reconstructions"]:
        ratio = rc["ratio"]; tag = f"{ratio}x"
        preds = {
            "Geo": _recon(geo_diff, geo_model, hf, ratio, rc, eta, coords, args.batch,
                          label=f"{tag} Geo", project=args.project),
            "No-geo": _recon(base_diff, base_model, hf, ratio, rc, eta, None, args.batch,
                             label=f"{tag} No-geo", project=args.project),
            "Bicubic": torch.cat([reconstruct_bicubic(hf[i:i + args.batch], ratio).cpu()
                                  for i in range(0, len(hf), args.batch)]),
        }
        row = {}
        for name, p in preds.items():
            pp = normalizer.decode(p)
            row[name] = {"l2": l2_norm(pp, hf_phys), "spectrum_log_l1": spectrum_log_l1(pp, hf_phys)}
            spectra[f"{name} {tag}"] = radial_power_spectrum(pp)
            print(f"  {tag} {name:8s} | L2 {row[name]['l2']:.4f} | spec-logL1 {row[name]['spectrum_log_l1']:.4f}")
        table[tag] = row
        _qualitative(normalizer, hf, preds, ratio, rc,
                     results_dir / f"geo_ablation_qualitative_{tag}.png")

    # ── Ensemble metrics (subset of patches; diffusion methods only) ──────
    if args.ensemble > 1:
        from eval.metrics import crps_ensemble
        n_e = min(args.ensemble_patches, len(hf))
        hf_e, hf_e_phys = hf[:n_e], hf_phys[:n_e]
        coords_e = None if coords is None else coords[:n_e]
        print(f"\nEnsemble metrics: {args.ensemble} members x {n_e} patches")
        ens = {}
        for rc in cfg["sample"]["reconstructions"]:
            ratio = rc["ratio"]; tag = f"{ratio}x"
            for name, (dif, mod, c) in {"Geo": (geo_diff, geo_model, coords_e),
                                        "No-geo": (base_diff, base_model, None)}.items():
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
                print(f"  {tag} {name:8s} | single L2 {row['single_l2']:.4f} | "
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
                                             "geo_ckpt": args.geo_ckpt,
                                             "base_ckpt": args.base_ckpt,
                                             "projection": args.project,
                                             "ensemble": args.ensemble})
    if wb_run is not None:
        tbl = wandb.Table(columns=["ratio", "method", "l2", "spectrum_log_l1"])
        log = {}
        for tag, row in table.items():
            if tag == "ensemble":
                for etag, erow in row.items():
                    for method, v in erow.items():
                        key = method.lower().replace("-", "_")
                        for mk, mv in v.items():
                            log[f"ablation/ensemble/{etag}/{key}/{mk}"] = mv
                continue
            for method, v in row.items():
                tbl.add_data(tag, method, v["l2"], v["spectrum_log_l1"])
                key = method.lower().replace("-", "_")
                log[f"ablation/{tag}/{key}/l2"] = v["l2"]
                log[f"ablation/{tag}/{key}/spectrum_log_l1"] = v["spectrum_log_l1"]
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
    panels = [("Input (LF)", lf),
              ("Bicubic", preds["Bicubic"][idx:idx + 1]),
              ("No-geo", preds["No-geo"][idx:idx + 1]),
              ("Geo", preds["Geo"][idx:idx + 1]),
              ("Reference", hf[idx:idx + 1].cpu())]
    ref = normalizer.decode(hf[idx:idx + 1].cpu())[0, 0].numpy()
    vmin, vmax = float(ref.min()), float(ref.max())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (title, t) in zip(axes, panels):
        ax.imshow(normalizer.decode(t.cpu())[0, 0].numpy(), cmap="RdBu_r",
                  vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{ratio}x reconstruction: geo vs no-geo (shared color scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot(spectra, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, e) in spectra.items():
        ax.loglog(k[1:], e[1:], "-" if label == "Reference" else "--",
                  lw=2.2 if label == "Reference" else 1.4, label=label)
    ax.set_xlabel("wavenumber k"); ax.set_ylabel("E(k)")
    ax.set_title("Geo vs no-geo: power spectrum")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
