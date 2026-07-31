"""Robustness experiment + headline table and figures.

Runs the SAME trained diffusion model on every ratio (4x in-distribution, 8x
out-of-distribution), alongside the direct-mapping baseline (trained on 4x only)
and bicubic. Produces:
  - results/headline_table.{json,txt}: rows {ratio} x cols {method} x {L2, spectrum}
  - results/spectrum.png:  E(k) per method/ratio vs reference
  - results/value_dist.png: Z500 value histograms vs reference
  - results/qualitative_{ratio}x.png: Input/Bicubic/DirectMap/Diffusion/Reference

The story is the 8x row: direct-map degrades, diffusion holds.

Run:
    python -m eval.make_tables_figures --config config/default.yaml
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
from data.degrade import degrade  # noqa: E402
from eval.metrics import (l2_norm, radial_power_spectrum,  # noqa: E402
                          spectrum_log_l1, value_histogram)
from sample.reconstruct import (load_diffusion, load_directmap,  # noqa: E402
                                load_residual, reconstruct_bicubic,
                                reconstruct_diffusion, reconstruct_directmap,
                                reconstruct_residual)
from utils import ensure_dir, get_device, init_wandb, load_config, run_name  # noqa: E402


@torch.no_grad()
def _batched(fn, hf, batch=16, extra=None, label=None):
    """Apply a per-batch reconstruction fn over all test patches.

    If `extra` is given (e.g. per-pixel geo coords), it is sliced in lockstep
    and passed as the second argument to `fn`. `label` shows a tqdm bar.
    """
    it = range(0, len(hf), batch)
    if label is not None:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=label)
        except ImportError:
            pass
    outs = []
    for i in it:
        if extra is None:
            outs.append(fn(hf[i:i + batch]).cpu())
        else:
            outs.append(fn(hf[i:i + batch], extra[i:i + batch]).cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--ckpt", default="diffusion.pt",
                    help="diffusion checkpoint name (e.g. diffusion_geo.pt)")
    ap.add_argument("--project", action="store_true",
                    help="Per-step data-consistency projection during sampling.")
    ap.add_argument("--dm-ckpt", default="directmap.pt",
                    help="direct-map checkpoint name (e.g. directmap_geo.pt)")
    ap.add_argument("--res-ckpt", default=None,
                    help="residual-diffusion checkpoint name (e.g. residual_geo.pt); "
                         "adds a 'Residual' method column when given.")
    ap.add_argument("--shuffle-geo", action="store_true",
                    help="Permutation control: shuffle the per-patch geo payload "
                         "across patches, so each patch is reconstructed with "
                         "ANOTHER patch's location. If the geo gain is genuinely "
                         "geographic the scores should fall back toward the "
                         "no-geo baseline; if they hold, the gain was capacity.")
    ap.add_argument("--ensemble", type=int, default=1,
                    help="Ensemble members per patch for the STOCHASTIC methods "
                         "(Diffusion, Residual). >1 adds ensemble-mean L2, CRPS "
                         "and spread on a subset of patches.")
    ap.add_argument("--ensemble-patches", type=int, default=64,
                    help="How many test patches the ensemble metrics use.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()

    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)

    n = min(cfg["eval"]["n_test_patches"], len(PatchDataset(patch_dir / "test_patches.npy", normalizer)))

    model, diffusion, cfg_ck = load_diffusion(ckpt_dir / args.ckpt, device)
    geo_on = cfg_ck.get("geo", {}).get("enabled", False)

    dm_model, dm_geo = None, False
    if (ckpt_dir / args.dm_ckpt).exists():
        dm_model, dm_cfg = load_directmap(ckpt_dir / args.dm_ckpt, device)
        dm_geo = dm_cfg.get("geo", {}).get("enabled", False)

    res_model, res_geo, res_mean_geo = None, False, False
    if args.res_ckpt is not None:
        (res_model, res_diff, res_cfg, res_std,
         res_mean, res_mean_geo) = load_residual(ckpt_dir / args.res_ckpt, device)
        res_geo = res_cfg.get("geo", {}).get("enabled", False)
        res_steps = res_cfg.get("residual", {}).get("n_steps", 100)

    # Test patches (+ per-pixel coords if any model is geo-conditioned).
    coords = None
    if geo_on or dm_geo or res_geo or res_mean_geo:
        g = (cfg_ck if geo_on else (dm_cfg if dm_geo else res_cfg))["geo"]
        ds = PatchDataset(
            patch_dir / "test_patches.npy", normalizer,
            origins_path=patch_dir / "test_origins.npy",
            coords_full_path=patch_dir / "coords_full.npz",
            geo_input_dim=g["input_dim"], altitude=g["altitude"],
            geo_encoder=g.get("encoder", "hash"),
        )
        items = [ds[i] for i in range(n)]
        hf = torch.stack([it[0] for it in items]).to(device)
        coords = torch.stack([it[1] for it in items]).to(device)
        if args.shuffle_geo:
            # Permutation control. Seeded off cfg['seed'] so the shuffled run is
            # reproducible and every model in this run sees the SAME mismatch.
            gen = torch.Generator().manual_seed(int(cfg["seed"]))
            perm = torch.randperm(coords.shape[0], generator=gen)
            fixed = int((perm == torch.arange(len(perm))).sum())
            coords = coords[perm.to(device)]
            print(f"[shuffle-geo] permuted geo payload across {len(perm)} patches "
                  f"({fixed} patch(es) kept their own location by chance)")
    else:
        ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
        hf = torch.stack([ds[i] for i in range(n)]).to(device)
    hf_phys = normalizer.decode(hf.cpu())                          # physical units
    print(f"Evaluating on {n} test patches | geo={geo_on} | dm_geo={dm_geo}")

    eta = cfg["sample"]["ddim_eta"]
    table = {}
    ens_table = {}    # tag -> method -> ensemble metrics (only with --ensemble > 1)
    spectra = {}      # label -> (k, E)
    hists = {}        # label -> (centers, density)
    vrange = (float(hf_phys.min()), float(hf_phys.max()))

    # Reference spectrum / distribution.
    k_ref, e_ref = radial_power_spectrum(hf_phys)
    spectra["Reference"] = (k_ref, e_ref)
    hists["Reference"] = value_histogram(hf_phys, cfg["eval"]["hist_bins"], vrange)

    for rc in cfg["sample"]["reconstructions"]:
        ratio = rc["ratio"]
        tag = f"{ratio}x"
        print(f"\n=== ratio {tag} (K={rc['K']}, t_steps={rc['t_steps']}, "
              f"smooth_sigma={rc.get('smooth_sigma', 0.0)}) ===")

        if geo_on:
            diff = _batched(
                lambda b, c: reconstruct_diffusion(diffusion, model, b, ratio, rc, eta=eta,
                                                   coords=c, project=args.project),
                hf, args.batch, extra=coords, label=f"{tag} Diffusion")
        else:
            diff = _batched(lambda b: reconstruct_diffusion(diffusion, model, b, ratio, rc,
                                                            eta=eta, project=args.project),
                            hf, args.batch, label=f"{tag} Diffusion")
        bic = _batched(lambda b: reconstruct_bicubic(b, ratio), hf, args.batch)
        preds = {"Diffusion": diff, "Bicubic": bic}
        if res_model is not None:
            if res_geo or res_mean_geo:
                preds["Residual"] = _batched(
                    lambda b, c: reconstruct_residual(res_diff, res_model, b, ratio,
                                                      res_std, n_steps=res_steps,
                                                      coords=c, project=args.project,
                                                      mean_model=res_mean,
                                                      mean_geo=res_mean_geo),
                    hf, args.batch, extra=coords, label=f"{tag} Residual")
            else:
                preds["Residual"] = _batched(
                    lambda b: reconstruct_residual(res_diff, res_model, b, ratio,
                                                   res_std, n_steps=res_steps,
                                                   project=args.project,
                                                   mean_model=res_mean),
                    hf, args.batch, label=f"{tag} Residual")
        if dm_model is not None:
            if dm_geo:
                preds["Direct map"] = _batched(
                    lambda b, c: reconstruct_directmap(
                        dm_model, b, ratio, rc.get("smooth_sigma", 0.0), coords=c),
                    hf, args.batch, extra=coords, label=f"{tag} Direct map")
            else:
                preds["Direct map"] = _batched(
                    lambda b: reconstruct_directmap(dm_model, b, ratio, rc.get("smooth_sigma", 0.0)),
                    hf, args.batch, label=f"{tag} Direct map")

        row = {}
        for name, p in preds.items():
            p_phys = normalizer.decode(p)
            row[name] = {
                "l2": l2_norm(p_phys, hf_phys),
                "spectrum_log_l1": spectrum_log_l1(p_phys, hf_phys),
            }
            spectra[f"{name} {tag}"] = radial_power_spectrum(p_phys)
            hists[f"{name} {tag}"] = value_histogram(p_phys, cfg["eval"]["hist_bins"], vrange)
            print(f"  {name:11s} | L2 {row[name]['l2']:.4f} | "
                  f"spec-logL1 {row[name]['spectrum_log_l1']:.4f}")
        table[tag] = row

        # ── Ensemble metrics for the STOCHASTIC methods only ───────────────
        # Bicubic and the direct map are deterministic, so repeated draws are
        # identical and CRPS/spread are meaningless for them.
        if args.ensemble > 1:
            from eval.metrics import crps_ensemble  # noqa: PLC0415
            n_e = min(args.ensemble_patches, len(hf))
            hf_e, hf_e_phys = hf[:n_e], hf_phys[:n_e]
            c_e = None if coords is None else coords[:n_e]

            def _diff_draw(m):
                if geo_on:
                    return _batched(
                        lambda b, c: reconstruct_diffusion(diffusion, model, b, ratio, rc,
                                                           eta=eta, coords=c,
                                                           project=args.project),
                        hf_e, args.batch, extra=c_e,
                        label=f"{tag} Diffusion ens {m}/{args.ensemble}")
                return _batched(
                    lambda b: reconstruct_diffusion(diffusion, model, b, ratio, rc,
                                                    eta=eta, project=args.project),
                    hf_e, args.batch, label=f"{tag} Diffusion ens {m}/{args.ensemble}")

            def _res_draw(m):
                if res_geo or res_mean_geo:
                    return _batched(
                        lambda b, c: reconstruct_residual(res_diff, res_model, b, ratio,
                                                          res_std, n_steps=res_steps,
                                                          coords=c, project=args.project,
                                                          mean_model=res_mean,
                                                          mean_geo=res_mean_geo),
                        hf_e, args.batch, extra=c_e,
                        label=f"{tag} Residual ens {m}/{args.ensemble}")
                return _batched(
                    lambda b: reconstruct_residual(res_diff, res_model, b, ratio,
                                                   res_std, n_steps=res_steps,
                                                   project=args.project,
                                                   mean_model=res_mean),
                    hf_e, args.batch, label=f"{tag} Residual ens {m}/{args.ensemble}")

            draws = {"Diffusion": _diff_draw}
            if res_model is not None:
                draws["Residual"] = _res_draw
            print(f"  ensemble: {args.ensemble} members x {n_e} patches")
            for name, draw in draws.items():
                members = [normalizer.decode(draw(m + 1)) for m in range(args.ensemble)]
                stack = torch.stack(members)
                e = {
                    "single_l2": float(np.mean([l2_norm(m_, hf_e_phys) for m_ in members])),
                    "ensemble_mean_l2": l2_norm(stack.mean(0), hf_e_phys),
                    "crps": crps_ensemble(members, hf_e_phys),
                    "spread": float(stack.std(0).mean()),
                }
                ens_table.setdefault(tag, {})[name] = e
                print(f"    {name:11s} | single {e['single_l2']:.4f} | "
                      f"ens-mean {e['ensemble_mean_l2']:.4f} | "
                      f"CRPS {e['crps']:.4f} | spread {e['spread']:.4f}")

        _qualitative(normalizer, hf, preds, ratio, rc, results_dir)

    _save_table(table, results_dir, ens_table)
    _plot_spectra(spectra, results_dir)
    _plot_hists(hists, results_dir)

    wb_run, wandb = init_wandb(cfg, job_type="eval",
                               extra_config={"n_test_patches": n,
                                             "has_directmap": dm_model is not None},
                               name=run_name(cfg, "table", Path(args.ckpt).stem,
                                             "proj" if args.project else "",
                                             "shufgeo" if args.shuffle_geo else "",
                                             f"ens{args.ensemble}" if args.ensemble > 1 else "",
                                             f"res-{Path(args.res_ckpt).stem}" if args.res_ckpt else "",
                                             f"dm-{Path(args.dm_ckpt).stem}" if dm_model is not None else ""))
    if wb_run is not None:
        # Scalars go to the run SUMMARY (columns in the runs table), not log():
        # a one-shot eval otherwise creates one single-point chart per metric.
        tbl = wandb.Table(columns=["ratio", "method", "l2", "spectrum_log_l1"])
        log = {}
        for tag, row in table.items():
            for method, v in row.items():
                tbl.add_data(tag, method, v["l2"], v["spectrum_log_l1"])
                key = method.lower().replace(" ", "_")
                wb_run.summary[f"{tag}/{key}/l2"] = v["l2"]
                wb_run.summary[f"{tag}/{key}/spectrum_log_l1"] = v["spectrum_log_l1"]
        for etag, erow in ens_table.items():
            for method, v in erow.items():
                for mk, mv in v.items():
                    wb_run.summary[f"ensemble/{etag}/{method.lower()}/{mk}"] = mv
        log["eval/headline_table"] = tbl
        log["eval/spectrum"] = wandb.Image(str(results_dir / "spectrum.png"))
        log["eval/value_dist"] = wandb.Image(str(results_dir / "value_dist.png"))
        for rc in cfg["sample"]["reconstructions"]:
            q = results_dir / f"qualitative_{rc['ratio']}x.png"
            if q.exists():
                log[f"eval/qualitative_{rc['ratio']}x"] = wandb.Image(str(q))
        wb_run.log(log)
        wb_run.finish()
        print("wandb: eval run logged")

    print(f"\nAll outputs -> {results_dir}")


def _qualitative(normalizer, hf, preds, ratio, rc, results_dir, idx=0):
    """One panel per method actually evaluated, plus input and reference, all
    on the reference's color scale."""
    lf = degrade(hf[idx:idx + 1].cpu(), ratio, rc.get("smooth_sigma", 0.0))
    panels = [("Input (LF)", lf)]
    for name in ("Bicubic", "Direct map", "Residual", "Diffusion"):
        if name in preds:
            panels.append((name, preds[name][idx:idx + 1]))
    panels.append(("Reference", hf[idx:idx + 1].cpu()))
    ref = normalizer.decode(hf[idx:idx + 1].cpu())[0, 0].numpy()
    vmin, vmax = float(ref.min()), float(ref.max())
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (title, t) in zip(axes, panels):
        ax.imshow(normalizer.decode(t)[0, 0].numpy(), cmap="RdBu_r",
                  vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{ratio}x reconstruction (shared color scale)")
    fig.tight_layout()
    fig.savefig(results_dir / f"qualitative_{ratio}x.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_table(table, results_dir, ens=None):
    """Write the JSON (ratios + optional 'ensemble' block) and the text table.

    The text table stays ratio x method; ensemble metrics have a different shape
    and would not fit its columns, so they go to the JSON only."""
    payload = dict(table)
    if ens:
        payload["ensemble"] = ens
    with open(results_dir / "headline_table.json", "w") as f:
        json.dump(payload, f, indent=2)
    lines = ["Headline comparison (L2 / spectrum-log-L1; lower is better)", ""]
    methods = sorted({m for row in table.values() for m in row})
    header = f"{'ratio':<8}" + "".join(f"{m:>26}" for m in methods)
    lines.append(header)
    lines.append("-" * len(header))
    for tag, row in table.items():
        cells = ""
        for m in methods:
            if m in row:
                cells += f"{row[m]['l2']:>12.4f}/{row[m]['spectrum_log_l1']:<13.4f}"
            else:
                cells += f"{'-':>26}"
        lines.append(f"{tag:<8}{cells}")
    txt = "\n".join(lines)
    with open(results_dir / "headline_table.txt", "w") as f:
        f.write(txt + "\n")
    print("\n" + txt)


def _plot_spectra(spectra, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, e) in spectra.items():
        style = "-" if label == "Reference" else "--"
        lw = 2.2 if label == "Reference" else 1.4
        ax.loglog(k[1:], e[1:], style, lw=lw, label=label)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title("Radial power spectrum")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "spectrum.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_hists(hists, results_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (c, d) in hists.items():
        style = "-" if label == "Reference" else "--"
        lw = 2.2 if label == "Reference" else 1.2
        ax.plot(c, d, style, lw=lw, label=label)
    ax.set_xlabel("Z500 value")
    ax.set_ylabel("density")
    ax.set_title("Value distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "value_dist.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
