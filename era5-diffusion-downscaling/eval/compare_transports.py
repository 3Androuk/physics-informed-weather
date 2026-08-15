"""Compare flow matching and stochastic interpolants on identical ERA5 patches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from sample.reconstruct import reconstruct_bicubic  # noqa: E402
from sample.transport import load_transport, reconstruct_transport  # noqa: E402
from eval.metrics import l2_norm, radial_power_spectrum, spectrum_log_l1  # noqa: E402
from utils import (channel_labels, display_channel, ensure_dir,  # noqa: E402
                   get_device, load_config)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml",
                    help="Evaluation data and ratio configuration.")
    ap.add_argument("--flow-ckpt", default="flow_matching.pt")
    ap.add_argument("--si-ckpt", default="stochastic_interpolant.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--solver", choices=["euler", "heun"], default=None)
    ap.add_argument("--si-sampler", choices=["ode", "sde"], default=None)
    ap.add_argument("--stochasticity", type=float, default=None)
    ap.add_argument("--projection", choices=["none", "final", "each"], default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"]) / "transport_comparison"
    results_dir.mkdir(parents=True, exist_ok=True)
    normalizer = load_norm_stats(patch_dir)
    plain_ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    n = min(cfg["eval"]["n_test_patches"], len(plain_ds))
    hf = torch.stack([plain_ds[i] for i in range(n)]).to(device)
    hf_phys = normalizer.decode(hf.cpu())
    labels_ch = channel_labels(cfg["data"])
    disp = display_channel(cfg)
    multi = len(labels_ch) > 1
    hf_disp = hf_phys[:, disp:disp + 1]  # figures + headline metrics channel

    specs = [
        ("Flow matching", ckpt_dir / args.flow_ckpt),
        ("Stochastic interpolant", ckpt_dir / args.si_ckpt),
    ]
    loaded = []
    for label, path in specs:
        if not path.exists():
            print(f"[skip] {label}: checkpoint not found at {path}")
            continue
        model, process, model_cfg, method, residual = load_transport(path, device)
        _validate_compatible(cfg, model_cfg, label)
        coords = _payload(patch_dir, normalizer, n, model_cfg, device)
        if residual is not None:
            label += " (residual)"
            # A geo-conditioned mean needs coords even when the transport model
            # itself is not: build them from the mean's own geo config.
            if coords is None and residual["mean_geo"]:
                coords = _payload(patch_dir, normalizer, n,
                                  {"geo": residual["mean_geo_cfg"]}, device)
        loaded.append((label, model, process, model_cfg, method, coords, residual))
    if not loaded:
        raise FileNotFoundError("no transport checkpoints found")

    ratios = cfg.get("transport", {}).get("eval_ratios", [4, 8])
    table = {}
    spectra = {"Reference": radial_power_spectrum(hf_disp)}
    for ratio in ratios:
        preds = {"Bicubic": _batched_bicubic(hf, ratio, args.batch)}
        for label, model, process, model_cfg, method, coords, residual in loaded:
            preds[label] = _batched_transport(
                hf, coords, args.batch, model, process, ratio, model_cfg, method,
                args, residual)
        row = {}
        for label, pred in preds.items():
            physical = normalizer.decode(pred)
            p_disp = physical[:, disp:disp + 1]
            # Headline metrics: display channel in physical units; multi-channel
            # runs additionally get an all-channel L2 in normalized units and
            # the per-channel physical breakdown (see make_tables_figures).
            row[label] = {
                "l2": l2_norm(p_disp, hf_disp),
                "spectrum_log_l1": spectrum_log_l1(p_disp, hf_disp),
            }
            extra = ""
            if multi:
                row[label]["l2_all_norm"] = l2_norm(pred, hf.cpu())
                row[label]["per_channel"] = {
                    lab: l2_norm(physical[:, c:c + 1], hf_phys[:, c:c + 1])
                    for c, lab in enumerate(labels_ch)
                }
                extra = f" | L2-all(norm) {row[label]['l2_all_norm']:.4f}"
            spectra[f"{label} {ratio}x"] = radial_power_spectrum(p_disp)
            print(f"{ratio}x {label:24s} | L2[{labels_ch[disp]}] {row[label]['l2']:.4f} | "
                  f"spectrum {row[label]['spectrum_log_l1']:.4f}{extra}")
        table[f"{ratio}x"] = row
        _qualitative(normalizer, hf, preds, ratio,
                     results_dir / f"qualitative_{ratio}x.png",
                     disp=disp, disp_label=labels_ch[disp])

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(table, f, indent=2)
    _plot_spectra(spectra, results_dir / "spectra.png")
    print(f"Outputs -> {results_dir}")


def _payload(patch_dir, normalizer, n, cfg, device):
    if not cfg.get("geo", {}).get("enabled", False):
        return None
    g = cfg["geo"]
    ds = PatchDataset(
        patch_dir / "test_patches.npy", normalizer,
        origins_path=patch_dir / "test_origins.npy",
        coords_full_path=patch_dir / "coords_full.npz",
        geo_input_dim=g["input_dim"], altitude=g.get("altitude"),
        geo_encoder=g.get("encoder", "hash"),
        healpix_index_path=((patch_dir / g["healpix_index"])
                            if g.get("healpix_index") else None),
    )
    return torch.stack([ds[i][1] for i in range(n)]).to(device)


def _validate_compatible(eval_cfg, model_cfg, label):
    if (eval_cfg.get("patches", {}).get("size")
            != model_cfg.get("patches", {}).get("size")):
        raise ValueError(f"{label} checkpoint patches.size does not match eval config")

    def _specs(cfg):
        d = cfg.get("data", {})
        return channel_labels(d) if (d.get("variables") or d.get("variable")) else None

    if _specs(eval_cfg) != _specs(model_cfg):
        raise ValueError(f"{label} checkpoint data channels do not match eval config")


def _batched_bicubic(hf, ratio, batch):
    return torch.cat([
        reconstruct_bicubic(hf[i:i + batch], ratio).cpu()
        for i in range(0, len(hf), batch)
    ])


def _batched_transport(hf, coords, batch, model, process, ratio, cfg, method,
                       args, residual=None):
    outputs = []
    for i in range(0, len(hf), batch):
        c = None if coords is None else coords[i:i + batch]
        outputs.append(reconstruct_transport(
            model, process, hf[i:i + batch], ratio, cfg, method, coords=c,
            steps=args.steps, solver=args.solver, sampler=args.si_sampler,
            stochasticity=args.stochasticity, projection=args.projection,
            residual=residual,
        ).cpu())
    return torch.cat(outputs)


def _qualitative(normalizer, hf, preds, ratio, path, idx=0, disp=0, disp_label=""):
    panels = [(name, pred[idx:idx + 1]) for name, pred in preds.items()]
    panels.append(("Reference", hf[idx:idx + 1].cpu()))
    ref = normalizer.decode(hf[idx:idx + 1].cpu())[0, disp].numpy()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2))
    for ax, (title, tensor) in zip(axes, panels):
        ax.imshow(normalizer.decode(tensor.cpu())[0, disp].numpy(), cmap="RdBu_r",
                  vmin=float(ref.min()), vmax=float(ref.max()))
        ax.set_title(title)
        ax.axis("off")
    chan = f" ({disp_label})" if disp_label else ""
    fig.suptitle(f"{ratio}x transport super-resolution{chan}")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_spectra(spectra, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (k, energy) in spectra.items():
        ax.loglog(k[1:], energy[1:], "-" if label == "Reference" else "--", label=label)
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
