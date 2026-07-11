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
                                reconstruct_bicubic, reconstruct_diffusion,
                                reconstruct_directmap)
from utils import ensure_dir, get_device, init_wandb, load_config  # noqa: E402


@torch.no_grad()
def _batched(fn, hf, batch=16):
    """Apply a per-batch reconstruction fn over all test patches."""
    outs = []
    for i in range(0, len(hf), batch):
        outs.append(fn(hf[i:i + batch]).cpu())
    return torch.cat(outs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()

    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)

    ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    n = min(cfg["eval"]["n_test_patches"], len(ds))
    hf = torch.stack([ds[i] for i in range(n)]).to(device)        # normalized
    hf_phys = normalizer.decode(hf.cpu())                          # physical units
    print(f"Evaluating on {n} test patches")

    model, diffusion, _ = load_diffusion(ckpt_dir / "diffusion.pt", device)
    dm_model = None
    if (ckpt_dir / "directmap.pt").exists():
        dm_model, _ = load_directmap(ckpt_dir / "directmap.pt", device)

    eta = cfg["sample"]["ddim_eta"]
    table = {}
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

        diff = _batched(lambda b: reconstruct_diffusion(diffusion, model, b, ratio, rc, eta=eta),
                        hf, args.batch)
        bic = _batched(lambda b: reconstruct_bicubic(b, ratio), hf, args.batch)
        preds = {"Diffusion": diff, "Bicubic": bic}
        if dm_model is not None:
            preds["Direct map"] = _batched(
                lambda b: reconstruct_directmap(dm_model, b, ratio, rc.get("smooth_sigma", 0.0)),
                hf, args.batch)

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

        _qualitative(normalizer, hf, diff, bic,
                     preds.get("Direct map"), ratio, rc, results_dir)

    _save_table(table, results_dir)
    _plot_spectra(spectra, results_dir)
    _plot_hists(hists, results_dir)

    wb_run, wandb = init_wandb(cfg, job_type="eval",
                               extra_config={"n_test_patches": n,
                                             "has_directmap": dm_model is not None})
    if wb_run is not None:
        tbl = wandb.Table(columns=["ratio", "method", "l2", "spectrum_log_l1"])
        log = {}
        for tag, row in table.items():
            for method, v in row.items():
                tbl.add_data(tag, method, v["l2"], v["spectrum_log_l1"])
                key = method.lower().replace(" ", "_")
                log[f"eval/{tag}/{key}/l2"] = v["l2"]
                log[f"eval/{tag}/{key}/spectrum_log_l1"] = v["spectrum_log_l1"]
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


def _qualitative(normalizer, hf, diff, bic, dm, ratio, rc, results_dir, idx=0):
    lf = degrade(hf[idx:idx + 1].cpu(), ratio, rc.get("smooth_sigma", 0.0))
    panels = [("Input (LF)", lf), ("Bicubic", bic[idx:idx + 1])]
    if dm is not None:
        panels.append(("Direct map", dm[idx:idx + 1]))
    panels += [("Diffusion", diff[idx:idx + 1]), ("Reference", hf[idx:idx + 1].cpu())]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (title, t) in zip(axes, panels):
        ax.imshow(normalizer.decode(t)[0, 0].numpy(), cmap="RdBu_r")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{ratio}x reconstruction")
    fig.tight_layout()
    fig.savefig(results_dir / f"qualitative_{ratio}x.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def _save_table(table, results_dir):
    with open(results_dir / "headline_table.json", "w") as f:
        json.dump(table, f, indent=2)
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
