"""Evaluate the trained DLWP-HPX SR model on the t2m test split.

Compares three reconstructions of the high-res field from the degraded input:
  * model      — the trained HEALPix U-Net,
  * bilinear   — seam-aware bilinear upsampling of the coarse field,
  * nearest    — the input itself (each coarse pixel repeated; lower bound).

Metrics (physical units, K) are computed directly on the mesh — HEALPix
pixels are equal-area, so plain means are already area-fair:
  * global RMSE / MAE / bias,
  * RMSE per latitude band (from pixel-center latitudes),
  * value histograms vs ground truth.

Also renders a few global maps (truth / input / model / error) remapped back
to the source lat-lon grid. Outputs go to results/: metrics.json + PNGs, and
to wandb when enabled.

Run:
    python -m eval.evaluate_sr --config config/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import HPXDataset, Normalizer  # noqa: E402
from data.degrade import coarsen_faces, degrade_faces, upsample_bilinear_faces  # noqa: E402
from hpx.mesh import nest_to_faces, pixel_lonlat_deg  # noqa: E402
from hpx.padding import HEALPixPadding  # noqa: E402
from hpx.remap import hpx_to_latlon  # noqa: E402
from models.hpx_unet import build_model  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def _metrics(pred_K: np.ndarray, truth_K: np.ndarray, band_masks) -> dict:
    err = pred_K - truth_K
    out = {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "bias": float(err.mean()),
        "rmse_by_band": {},
    }
    for name, mask in band_masks.items():
        out["rmse_by_band"][name] = float(np.sqrt((err[..., mask] ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    ec = cfg["eval"]
    ratio = int(cfg["sr"]["ratio"])
    nside = int(cfg["hpx"]["nside"])
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    ckpt_path = Path(cfg["paths"]["ckpt_dir"]) / ec["checkpoint"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = Normalizer(ckpt["norm_mean"], ckpt["norm_std"])
    model = build_model(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path} (epoch {ckpt['epoch']}, "
          f"val rmse {ckpt['val_rmse_norm']:.5f} norm)")

    ds = HPXDataset(hpx_dir / "test.npy", normalizer)
    n = len(ds) if ec["n_test_samples"] is None else min(ec["n_test_samples"], len(ds))
    print(f"Evaluating {n}/{len(ds)} test samples | ratio {ratio}x | nside {nside}")

    wb_run, wandb = init_wandb(cfg, job_type="eval_sr",
                               extra_config={"n_eval": n, "ckpt": str(ckpt_path)})

    # pixel-center latitudes -> per-band masks over the (12, F, F) faces
    _, plat = pixel_lonlat_deg(nside)
    lat_faces = nest_to_faces(plat, nside)
    band_masks = {}
    for lo, hi in ec["lat_bands"]:
        band_masks[f"{lo}..{hi}"] = (lat_faces >= lo) & (lat_faces < hi)

    padder1 = HEALPixPadding(nside // ratio, 1).to(device)

    methods = ("model", "bilinear", "nearest")
    sq_err = {m: 0.0 for m in methods}
    abs_err = {m: 0.0 for m in methods}
    bias = {m: 0.0 for m in methods}
    band_sq = {m: {b: 0.0 for b in band_masks} for m in methods}
    band_n = {b: 0.0 for b in band_masks}
    hists = {m: None for m in (*methods, "truth")}
    hist_range = (normalizer.mean - 5 * normalizer.std,
                  normalizer.mean + 5 * normalizer.std)
    n_px = 0

    batch = 8
    with torch.no_grad():
        for start in range(0, n, batch):
            y = torch.stack([ds[i] for i in range(start, min(start + batch, n))])
            y = y.to(device)                      # (B, 12, 1, F, F) normalized
            lo = coarsen_faces(y, ratio)          # coarse HEALPix field
            x = degrade_faces(y, ratio)           # nearest-upsampled input
            preds = {
                "model": model(x),
                "bilinear": upsample_bilinear_faces(lo, ratio, padder1),
                "nearest": x,
            }
            truth_K = normalizer.decode(y).cpu().numpy()[:, :, 0]
            h, _ = np.histogram(truth_K, bins=ec["hist_bins"], range=hist_range)
            hists["truth"] = h if hists["truth"] is None else hists["truth"] + h
            n_px += truth_K.size
            for b, mask in band_masks.items():
                band_n[b] += truth_K.shape[0] * int(mask.sum())
            for m in methods:
                pred_K = normalizer.decode(preds[m]).cpu().numpy()[:, :, 0]
                err = pred_K - truth_K
                sq_err[m] += float((err ** 2).sum())
                abs_err[m] += float(np.abs(err).sum())
                bias[m] += float(err.sum())
                for b, mask in band_masks.items():
                    band_sq[m][b] += float((err[..., mask] ** 2).sum())
                h, _ = np.histogram(pred_K, bins=ec["hist_bins"], range=hist_range)
                hists[m] = h if hists[m] is None else hists[m] + h

    metrics = {"n_samples": n, "ratio": ratio, "nside": nside}
    for m in methods:
        metrics[m] = {
            "rmse_K": float(np.sqrt(sq_err[m] / n_px)),
            "mae_K": abs_err[m] / n_px,
            "bias_K": bias[m] / n_px,
            "rmse_by_band_K": {b: float(np.sqrt(band_sq[m][b] / band_n[b]))
                               for b in band_masks},
        }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ── Figures ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # value histograms
    edges = np.linspace(*hist_range, ec["hist_bins"] + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, style in (("truth", "k-"), ("model", "C0-"),
                     ("bilinear", "C1--"), ("nearest", "C2:")):
        ax.plot(centers, hists[m] / (n_px * np.diff(edges)), style, label=m)
    ax.set_xlabel("t2m [K]")
    ax.set_ylabel("density")
    ax.set_title(f"t2m value distribution ({ratio}x SR, nside {nside})")
    ax.legend()
    fig.tight_layout()
    hist_path = results_dir / "histograms.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    # global maps for a few samples, remapped back to lat-lon
    coords = np.load(hpx_dir / "coords.npz")
    lat, lon = coords["lat"], coords["lon"]
    times = None
    times_path = hpx_dir / "test_times.npy"
    if times_path.exists():
        times = np.load(times_path)
    map_paths = []
    with torch.no_grad():
        for i in range(min(ec["n_map_samples"], n)):
            y = ds[i].unsqueeze(0).to(device)
            x = degrade_faces(y, ratio)
            pred = model(x)
            fields = {
                "truth": normalizer.decode(y),
                "input (degraded)": normalizer.decode(x),
                "model": normalizer.decode(pred),
            }
            fields = {k: hpx_to_latlon(v.cpu().numpy()[0, :, 0], lat, lon)
                      for k, v in fields.items()}
            fields["model - truth"] = fields["model"] - fields["truth"]

            fig, axes = plt.subplots(2, 2, figsize=(13, 6.5))
            vmin, vmax = fields["truth"].min(), fields["truth"].max()
            for ax, (name, fld) in zip(axes.ravel(), fields.items()):
                if name == "model - truth":
                    lim = max(1.0, float(np.abs(fld).max()))
                    im = ax.pcolormesh(lon, lat, fld, cmap="RdBu_r",
                                       vmin=-lim, vmax=lim, shading="auto")
                else:
                    im = ax.pcolormesh(lon, lat, fld, cmap="RdYlBu_r",
                                       vmin=vmin, vmax=vmax, shading="auto")
                ax.set_title(name)
                fig.colorbar(im, ax=ax, shrink=0.9)
            title = f"sample {i}"
            if times is not None:
                title += f" | {np.datetime_as_string(times[i], unit='h')}"
            fig.suptitle(f"t2m {ratio}x SR on HPX{nside} — {title}")
            fig.tight_layout()
            p = results_dir / f"map_{i:02d}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            map_paths.append(p)

    print(f"Figures -> {hist_path}, {', '.join(str(p) for p in map_paths)}")
    if wb_run is not None:
        log = {"eval/metrics": metrics,
               "eval/histograms": wandb.Image(str(hist_path))}
        for p in map_paths:
            log[f"eval/{p.stem}"] = wandb.Image(str(p))
        for m in methods:
            log[f"eval/rmse_K/{m}"] = metrics[m]["rmse_K"]
        wb_run.log(log)
        wb_run.finish()


if __name__ == "__main__":
    main()
