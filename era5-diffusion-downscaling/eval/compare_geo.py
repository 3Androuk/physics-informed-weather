"""Comparison of N guided-diffusion checkpoints on identical test patches.

Originally a geo-vs-baseline pair; now a generic ladder: pass any number of
checkpoints via --ckpts (names in paths.ckpt_dir or absolute paths) and each
model receives the geo payload its own config requires (hash coords, healpix
indices, raw xyz, sinusoidal coords, or static fields), built from the same
test origins — so the full geo-encoder ladder (no-geo, xyz, sinusoidal,
static, hash, healpix) compares on identical patches at every ratio, alongside
bicubic. Reports L2 (RMSE) and the power-spectrum metric; optional per-step
DDNM projection, --shuffle-geo permutation control, and ensemble metrics
(ensemble-mean L2, CRPS, spread).

Run (ladder):
    python -m eval.compare_geo --config config/t2m.yaml --wandb \
        --ckpts diffusion.pt diffusion_geo.pt diffusion_geo_hpx.pt \
                diffusion_geo_xyz.pt diffusion_geo_sin.pt diffusion_geo_static.pt

Run (legacy pair):
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
from eval.metrics import (radial_power_spectrum, spectrum_log_l1,  # noqa: E402
                          l2_norm, l2_norm_weighted, patch_latitudes)
from sample.reconstruct import (load_diffusion, reconstruct_bicubic,  # noqa: E402
                                reconstruct_diffusion)
from utils import ensure_dir, get_device, init_wandb, load_config, run_name  # noqa: E402


def _recon(diffusion, model, hf, ratio, rc, eta, coords, batch, label="recon",
           project=False, dps_scale=0.0):
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
                                          eta=eta, coords=c, project=project,
                                          dps_scale=dps_scale).cpu())
    return torch.cat(outs, dim=0)


def _payload(patch_dir, normalizer, n, geo_cfg, device):
    """Geo payload stack for the first n test patches, per one model's config."""
    ds = PatchDataset(
        patch_dir / "test_patches.npy", normalizer,
        origins_path=patch_dir / "test_origins.npy",
        coords_full_path=patch_dir / "coords_full.npz",
        geo_input_dim=geo_cfg["input_dim"], altitude=geo_cfg["altitude"],
        geo_encoder=geo_cfg.get("encoder", "hash"),
        healpix_index_path=((patch_dir / geo_cfg["healpix_index"])
                            if geo_cfg.get("healpix_index") else None),
    )
    return torch.stack([ds[i][1] for i in range(n)]).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="N checkpoints to compare on identical patches (names in "
                         "paths.ckpt_dir or absolute paths). Overrides "
                         "--geo-ckpt/--base-ckpt.")
    ap.add_argument("--geo-ckpt", default="diffusion_geo.pt",
                    help="legacy pair mode: model A (may be geo or plain)")
    ap.add_argument("--base-ckpt", default="diffusion.pt",
                    help="legacy pair mode: model B (may be geo or plain)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--project", action="store_true",
                    help="Per-step data-consistency projection: coarsen(x0) == LF "
                         "enforced at every DDIM step.")
    ap.add_argument("--shuffle-geo", action="store_true",
                    help="Permutation control: every geo-conditioned model gets "
                         "ANOTHER patch's geo payload (same permutation for all "
                         "models). Genuinely geographic gains should collapse "
                         "toward no-geo; gains that hold were capacity.")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="Ensemble members per patch (>1 adds ensemble-mean L2, "
                         "CRPS, and spread on a subset of patches).")
    ap.add_argument("--ensemble-patches", type=int, default=64,
                    help="How many test patches the ensemble metrics use.")
    ap.add_argument("--eta", type=float, default=None,
                    help="Override sample.ddim_eta (DDIM stochasticity) for "
                         "this eval only. >0 diversifies ensemble members "
                         "beyond the noise-mixing initialization — the fix for "
                         "underdispersive ensembles; no retraining involved.")
    ap.add_argument("--dps", type=float, default=0.0,
                    help="DPS likelihood-guidance step size (Chung et al. "
                         "2023): each DDIM step also descends the gradient of "
                         "||y - A x0_hat|| THROUGH the denoiser. Soft "
                         "data consistency — an alternative or complement to "
                         "--project. Costs one backward pass per step "
                         "(halve --batch if VRAM is tight). Try 0.3-1.0.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()
    eta = cfg["sample"]["ddim_eta"] if args.eta is None else args.eta

    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)
    n = min(cfg["eval"]["n_test_patches"],
            len(PatchDataset(patch_dir / "test_patches.npy", normalizer)))

    # ── Load every checkpoint; each builds the geo payload ITS config needs ─
    ckpt_names = args.ckpts or [args.geo_ckpt, args.base_ckpt]
    models = []   # (display name, model, diffusion, coords, encoder tag)
    seen = {}
    for name in ckpt_names:
        path = ckpt_dir / name
        model, diff, cfg_ck = load_diffusion(path, device)
        geo_on = cfg_ck.get("geo", {}).get("enabled", False)
        encoder = cfg_ck["geo"].get("encoder", "hash") if geo_on else "-"
        coords = _payload(patch_dir, normalizer, n, cfg_ck["geo"], device) if geo_on else None
        disp = Path(name).stem
        if disp in seen:  # same stem from different directories
            seen[disp] += 1
            disp = f"{disp}#{seen[disp]}"
        else:
            seen[disp] = 1
        models.append((disp, model, diff, coords, encoder))
        print(f"  {disp}: {path} (geo={geo_on}, encoder={encoder})")
    print(f"Comparing {len(models)} checkpoint(s) on {n} patches"
          f"{' | projection ON' if args.project else ''}"
          f"{f' | DPS scale {args.dps:g}' if args.dps > 0 else ''}"
          f"{' | SHUFFLED geo payloads' if args.shuffle_geo else ''}")

    if args.shuffle_geo:
        # One shared permutation (seeded off cfg seed) so every model sees the
        # SAME location mismatch and the control is reproducible.
        gen = torch.Generator().manual_seed(int(cfg["seed"]))
        perm = torch.randperm(n, generator=gen).to(device)
        models = [(d, m, dif, None if c is None else c[perm], e)
                  for d, m, dif, c, e in models]

    tag_names = "_vs_".join(d for d, *_ in models) if len(models) <= 2 \
        else f"{len(models)}way"
    stem = (f"compare_{tag_names}{'_proj' if args.project else ''}"
            f"{'_shufgeo' if args.shuffle_geo else ''}")
    if args.eta is not None:
        stem += f"_eta{args.eta:g}"
    if args.dps > 0:
        stem += f"_dps{args.dps:g}"

    ds_plain = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    hf = torch.stack([ds_plain[i] for i in range(n)]).to(device)
    hf_phys = normalizer.decode(hf.cpu())
    # Latitude-weighted RMSE (WeatherBench2 convention) alongside plain L2;
    # None on legacy patch dirs without saved origins/coords.
    lat = patch_latitudes(patch_dir, n, hf.shape[-2])
    if lat is None:
        print("  (no origins/coords_full.npz — latitude-weighted RMSE skipped)")

    table, spectra = {}, {"Reference": radial_power_spectrum(hf_phys)}
    for rc in cfg["sample"]["reconstructions"]:
        ratio = rc["ratio"]; tag = f"{ratio}x"
        preds = {disp: _recon(dif, mod, hf, ratio, rc, eta, coords, args.batch,
                              label=f"{tag} {disp}", project=args.project,
                              dps_scale=args.dps)
                 for disp, mod, dif, coords, _ in models}
        preds["Bicubic"] = torch.cat(
            [reconstruct_bicubic(hf[i:i + args.batch], ratio).cpu()
             for i in range(0, len(hf), args.batch)])
        row = {}
        for name, p in preds.items():
            pp = normalizer.decode(p)
            row[name] = {"l2": l2_norm(pp, hf_phys),
                         "spectrum_log_l1": spectrum_log_l1(pp, hf_phys)}
            if lat is not None:
                row[name]["l2_latweighted"] = l2_norm_weighted(pp, hf_phys, lat)
            spectra[f"{name} {tag}"] = radial_power_spectrum(pp)
            lw = (f" | L2(lat-w) {row[name]['l2_latweighted']:.4f}"
                  if lat is not None else "")
            print(f"  {tag} {name:28s} | L2 {row[name]['l2']:.4f}{lw} | "
                  f"spec-logL1 {row[name]['spectrum_log_l1']:.4f}")
        table[tag] = row
        _qualitative(normalizer, hf, preds, ratio, rc,
                     results_dir / f"{stem}_qualitative_{tag}.png")

    # ── Ensemble metrics (subset of patches; diffusion methods only) ──────
    if args.ensemble > 1:
        from eval.metrics import crps_ensemble
        n_e = min(args.ensemble_patches, len(hf))
        hf_e, hf_e_phys = hf[:n_e], hf_phys[:n_e]
        print(f"\nEnsemble metrics: {args.ensemble} members x {n_e} patches")
        ens = {}
        for rc in cfg["sample"]["reconstructions"]:
            ratio = rc["ratio"]; tag = f"{ratio}x"
            for disp, mod, dif, coords, _ in models:
                c = None if coords is None else coords[:n_e]
                members = [normalizer.decode(
                    _recon(dif, mod, hf_e, ratio, rc, eta, c, args.batch,
                           label=f"{tag} {disp} member {m + 1}/{args.ensemble}",
                           project=args.project, dps_scale=args.dps))
                    for m in range(args.ensemble)]
                stack = torch.stack(members)
                row = {
                    "single_l2": float(np.mean([l2_norm(p, hf_e_phys) for p in members])),
                    "ensemble_mean_l2": l2_norm(stack.mean(0), hf_e_phys),
                    "crps": crps_ensemble(members, hf_e_phys),
                    "spread": float(stack.std(0).mean()),
                }
                if lat is not None:
                    row["ensemble_mean_l2_latweighted"] = l2_norm_weighted(
                        stack.mean(0), hf_e_phys, lat[:n_e])
                ens.setdefault(tag, {})[disp] = row
                print(f"  {tag} {disp:28s} | single L2 {row['single_l2']:.4f} | "
                      f"ens-mean L2 {row['ensemble_mean_l2']:.4f} | "
                      f"CRPS {row['crps']:.4f} | spread {row['spread']:.4f}")
        table["ensemble"] = ens

    with open(results_dir / f"{stem}.json", "w") as f:
        json.dump(table, f, indent=2)
    _plot(spectra, results_dir / f"{stem}_spectrum.png")
    print(f"\nSaved -> {results_dir / stem}.json, {stem}_spectrum.png, "
          f"and {stem}_qualitative_*.png")

    wb_run, wandb = init_wandb(cfg, job_type="compare_geo",
                               extra_config={"n_test_patches": n,
                                             "ckpts": ckpt_names,
                                             "projection": args.project,
                                             "dps_scale": args.dps,
                                             "shuffle_geo": args.shuffle_geo,
                                             "ensemble": args.ensemble},
                               name=run_name(cfg, "ladder" if len(models) > 2 else "ablation",
                                             *(d for d, *_ in models),
                                             "proj" if args.project else "",
                                             f"dps{args.dps:g}" if args.dps > 0 else "",
                                             "shufgeo" if args.shuffle_geo else "",
                                             f"ens{args.ensemble}" if args.ensemble > 1 else "",
                                             f"eta{args.eta:g}" if args.eta is not None else ""))
    if wb_run is not None:
        # Scalars go to the run SUMMARY (columns in the runs table), not log():
        # a one-shot eval otherwise creates one single-point chart per metric.
        tbl = wandb.Table(columns=["ratio", "method", "l2", "l2_latweighted",
                                   "spectrum_log_l1"])
        log = {}
        for tag, row in table.items():
            if tag == "ensemble":
                for etag, erow in row.items():
                    for method, v in erow.items():
                        for mk, mv in v.items():
                            wb_run.summary[f"ensemble/{etag}/{method}/{mk}"] = mv
                continue
            for method, v in row.items():
                tbl.add_data(tag, method, v["l2"], v.get("l2_latweighted"),
                             v["spectrum_log_l1"])
                for mk, mv in v.items():
                    wb_run.summary[f"{tag}/{method}/{mk}"] = mv
        log["ablation/table"] = tbl
        log["ablation/spectrum"] = wandb.Image(str(results_dir / f"{stem}_spectrum.png"))
        for rc in cfg["sample"]["reconstructions"]:
            q = results_dir / f"{stem}_qualitative_{rc['ratio']}x.png"
            if q.exists():
                log[f"ablation/qualitative_{rc['ratio']}x"] = wandb.Image(str(q))
            # per-model Input|model|Reference figures
            for pm in sorted(results_dir.glob(f"{stem}_qualitative_{rc['ratio']}x_*.png")):
                key = pm.stem.replace(f"{stem}_", "")
                log[f"ablation/{key}"] = wandb.Image(str(pm))
        wb_run.log(log)
        wb_run.finish()
        print("wandb: ablation run logged")


def _qualitative(normalizer, hf, preds, ratio, rc, path, idx=0):
    """Side-by-side panels on a SHARED color scale (taken from the reference),
    so residual noise or bias shows as a visible difference instead of being
    hidden by per-panel autoscaling. Also writes one small Input|model|Reference
    figure PER model next to the combined panel."""
    from data.degrade import degrade
    lf = degrade(hf[idx:idx + 1].cpu(), ratio, rc.get("smooth_sigma", 0.0))
    ref = normalizer.decode(hf[idx:idx + 1].cpu())[0, 0].numpy()
    vmin, vmax = float(ref.min()), float(ref.max())

    def _panel_fig(panels, out_path, suptitle):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
        for ax, (title, t) in zip(np.atleast_1d(axes), panels):
            ax.imshow(normalizer.decode(t.cpu())[0, 0].numpy(), cmap="RdBu_r",
                      vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        fig.suptitle(suptitle)
        fig.tight_layout()
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    panels = [("Input (LF)", lf)]
    for name, p in preds.items():
        panels.append((name, p[idx:idx + 1]))
    panels.append(("Reference", hf[idx:idx + 1].cpu()))
    _panel_fig(panels, path, f"{ratio}x reconstruction (shared color scale)")

    # One figure per model, each with the reference for direct comparison.
    for name, p in preds.items():
        safe = name.replace(" ", "_").replace("#", "")
        _panel_fig([("Input (LF)", lf), (name, p[idx:idx + 1]),
                    ("Reference", hf[idx:idx + 1].cpu())],
                   path.with_name(f"{path.stem}_{safe}{path.suffix}"),
                   f"{ratio}x — {name} vs reference (shared color scale)")


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
