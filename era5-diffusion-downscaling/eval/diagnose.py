"""Diagnostic battery: WHERE does the remaining reconstruction error live?

Four diagnostics over N guided-diffusion checkpoints (same ladder interface as
eval.compare_geo), each answering a different upgrade question:

  errmap     Per-pixel |error| accumulated over the test patches into a map of
             the full grid, per model (+ difference map between the first two
             models). Error ridges on terrain/coastlines => location signal is
             still being missed (conditioning upgrades apply); diffuse synoptic
             blobs => the residual is weather, conditioning is done.
  memgap     Denoising loss on TRAIN vs TEST patches, binned by noise level
             (x-axis: signal fraction u = 1 - t/T). A train/test gap opening at
             high u (low noise) is the fine-scale memorization signature that
             --gated targets; high absolute loss in a band = that band is
             underfit (capacity/schedule upgrades apply).
  coherence  Radial spectral coherence pred-vs-reference per wavenumber, with
             the coarse-input Nyquist marked. Coherence dying right above the
             cutoff => the sampler doesn't propagate information upward (tune
             K/t_steps/eta); coherent but power-deficient fine scales => the
             generative prior under-produces texture (training-side fix).
  qq         Quantile-quantile of predicted vs reference values (display
             channel, physical units). Compressed tails = residual
             mean-reversion the pixel metrics are blind to.

Run (mirrors compare_geo):
    python -m eval.diagnose --config config/t2m.yaml --project --wandb \
        --ckpts diffusion_geo_combo.pt diffusion_geo_static.pt diffusion_geo.pt
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
from eval.compare_geo import _recon  # noqa: E402
from eval.metrics import radial_coherence  # noqa: E402
from sample.reconstruct import load_diffusion  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, run_name  # noqa: E402

DIAGNOSTICS = ("errmap", "memgap", "coherence", "qq")


def accumulate_error_map(errs, origins, grid_hw):
    """Mean per-pixel error over all patches covering each grid cell.

    errs: (N, s, s) per-patch |error|; origins: (N, 2) row/col grid origins.
    Returns (H, W) with NaN where no test patch ever lands."""
    h, w = grid_hw
    s = errs.shape[-1]
    sums, counts = np.zeros((h, w)), np.zeros((h, w))
    for e, (r0, c0) in zip(errs, origins):
        sums[r0:r0 + s, c0:c0 + s] += e
        counts[r0:r0 + s, c0:c0 + s] += 1
    return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _payload(patch_dir, normalizer, idx, geo_cfg, device, split="test"):
    """Geo payload for the given patch indices of a split, per one model's
    config (same construction as compare_geo, generalized to train/test)."""
    ds = PatchDataset(
        patch_dir / f"{split}_patches.npy", normalizer,
        origins_path=patch_dir / f"{split}_origins.npy",
        coords_full_path=patch_dir / "coords_full.npz",
        geo_input_dim=geo_cfg["input_dim"], altitude=geo_cfg["altitude"],
        geo_encoder=geo_cfg.get("encoder", "hash"),
        healpix_index_path=((patch_dir / geo_cfg["healpix_index"])
                            if geo_cfg.get("healpix_index") else None),
    )
    return torch.stack([ds[i][1] for i in idx]).to(device)


@torch.no_grad()
def _binned_denoise_loss(model, diffusion, x, coords, t_centers, batch):
    """Noise-prediction MSE at fixed timesteps. The injected noise is seeded
    per bin so train and test evaluate against comparable perturbations."""
    out = []
    for b, tc in enumerate(t_centers):
        gen = torch.Generator().manual_seed(10_000 + b)
        tot, cnt = 0.0, 0
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]
            cb = None if coords is None else coords[i:i + batch]
            noise = torch.randn(xb.shape, generator=gen).to(xb.device)
            t = torch.full((len(xb),), int(tc), device=xb.device)
            xt = diffusion.q_sample(xb, t, noise)
            pred = model(xt, t.float()) if cb is None else model(xt, t.float(), cb)
            tot += float((pred - noise).pow(2).mean()) * len(xb)
            cnt += len(xb)
        out.append(tot / cnt)
    return out


def _grid(patch_dir):
    z = np.load(patch_dir / "coords_full.npz")
    lat, lon = z["lat"], z["lon"]
    lsm = None
    sf = patch_dir / "static_fields.npz"
    if sf.exists():
        z = np.load(sf, allow_pickle=True)
        for i, name in enumerate(z["names"]):
            if "land_sea" in str(name):
                lsm = z["fields"][i]
    return lat, lon, lsm


def _map_axes(ax, m, lat, lon, lsm, title, cmap="magma", vmax=None, vmin=None):
    origin = "upper" if lat[0] > lat[-1] else "lower"
    extent = [float(lon.min()), float(lon.max()),
              float(min(lat[0], lat[-1])), float(max(lat[0], lat[-1]))]
    im = ax.imshow(m, cmap=cmap, origin=origin, extent=extent, aspect="auto",
                   vmax=vmax, vmin=vmin)
    if lsm is not None:
        ax.contour(lon, lat, lsm, levels=[0.5], colors="cyan", linewidths=0.4)
    ax.set_title(title, fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="Guided-DDPM checkpoints (names in paths.ckpt_dir or "
                         "absolute paths); each gets the geo payload its own "
                         "config requires.")
    ap.add_argument("--diagnostics", nargs="+", choices=DIAGNOSTICS,
                    default=list(DIAGNOSTICS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--project", action="store_true",
                    help="Per-step data-consistency projection during "
                         "reconstruction (match the setting your tables use).")
    ap.add_argument("--mem-patches", type=int, default=256,
                    help="memgap: patches per split for the binned loss.")
    ap.add_argument("--t-bins", type=int, default=12,
                    help="memgap: number of noise-level bins over [1, T].")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    device = get_device()
    eta = cfg["sample"]["ddim_eta"]
    disp = int(cfg.get("eval", {}).get("display_channel", 0))
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)

    ds_plain = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    n = min(cfg["eval"]["n_test_patches"], len(ds_plain))
    hf = torch.stack([ds_plain[i] for i in range(n)]).to(device)
    hf_phys = normalizer.decode(hf.cpu())
    test_origins = np.load(patch_dir / "test_origins.npy")[:n]
    lat, lon, lsm = _grid(patch_dir)

    models = []  # (name, model, diffusion, test_coords, cfg_ck)
    for name in args.ckpts:
        model, dif, cfg_ck = load_diffusion(ckpt_dir / name, device)
        geo_on = cfg_ck.get("geo", {}).get("enabled", False)
        coords = (_payload(patch_dir, normalizer, range(n), cfg_ck["geo"], device)
                  if geo_on else None)
        models.append((Path(name).stem, model, dif, coords, cfg_ck))
        print(f"  {Path(name).stem}: geo={geo_on}"
              f"{', encoder=' + cfg_ck['geo'].get('encoder', 'hash') if geo_on else ''}")

    recon_needed = any(d in args.diagnostics for d in ("errmap", "coherence", "qq"))
    stem = f"diag_{len(models)}way{'_proj' if args.project else ''}"
    report = {"ckpts": args.ckpts, "projection": args.project}
    figures = []

    def _save(fig, name):
        path = results_dir / name
        if not fig.get_constrained_layout():
            fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        figures.append(path)
        print(f"saved -> {path}")

    if recon_needed:
        for rc in cfg["sample"]["reconstructions"]:
            ratio = rc["ratio"]
            tag = f"{ratio}x"
            preds = {name: normalizer.decode(
                _recon(dif, mod, hf, ratio, rc, eta, coords, args.batch,
                       label=f"{tag} {name}", project=args.project))
                for name, mod, dif, coords, _ in models}

            if "errmap" in args.diagnostics:
                maps = {name: accumulate_error_map(
                    (p[:, disp] - hf_phys[:, disp]).abs().numpy(),
                    test_origins, (len(lat), len(lon)))
                    for name, p in preds.items()}
                vmax = float(np.nanpercentile(np.stack(list(maps.values())), 99))
                ncols = len(maps) + (1 if len(maps) >= 2 else 0)
                fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 3.2),
                                         squeeze=False, constrained_layout=True)
                ims = [_map_axes(axes[0][i], m, lat, lon, lsm,
                                 f"{name} {tag} |err|", vmax=vmax, vmin=0.0)
                       for i, (name, m) in enumerate(maps.items())]
                fig.colorbar(ims[0], ax=list(axes[0][:len(maps)]), shrink=0.85)
                if len(maps) >= 2:
                    (na, ma), (nb, mb) = list(maps.items())[:2]
                    d = ma - mb
                    lim = float(np.nanpercentile(np.abs(d), 99))
                    im = _map_axes(axes[0][-1], d, lat, lon, lsm,
                                   f"{na} - {nb}", cmap="RdBu_r",
                                   vmin=-lim, vmax=lim)
                    fig.colorbar(im, ax=axes[0][-1], shrink=0.85)
                _save(fig, f"{stem}_errmap_{tag}.png")
                report.setdefault("errmap", {})[tag] = {
                    name: {"mean": float(np.nanmean(m)),
                           "p99": float(np.nanpercentile(m, 99))}
                    for name, m in maps.items()}

            if "coherence" in args.diagnostics:
                fig, ax = plt.subplots(figsize=(7, 4))
                coh_rows = {}
                for name, p in preds.items():
                    k, coh = radial_coherence(p[:, disp], hf_phys[:, disp])
                    ax.plot(k, coh, label=name, lw=1.4)
                    coh_rows[name] = {"k": k.tolist(),
                                      "coherence": np.round(coh, 4).tolist()}
                k_cut = hf.shape[-1] // (2 * ratio)
                ax.axvline(k_cut, color="gray", ls="--", lw=1,
                           label=f"coarse-input Nyquist (k={k_cut})")
                ax.set(xlabel="wavenumber k", ylabel="coherence with reference",
                       ylim=(0, 1.02), title=f"{tag} spectral coherence")
                ax.legend(fontsize=8)
                _save(fig, f"{stem}_coherence_{tag}.png")
                report.setdefault("coherence", {})[tag] = coh_rows

            if "qq" in args.diagnostics:
                qs = np.linspace(0.1, 99.9, 249)
                ref_q = np.percentile(hf_phys[:, disp].numpy(), qs)
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                rows = {}
                for name, p in preds.items():
                    pred_q = np.percentile(p[:, disp].numpy(), qs)
                    ax.plot(ref_q, pred_q, lw=1.3, label=name)
                    rows[name] = {"q01_err": float(pred_q[qs <= 1][-1] - ref_q[qs <= 1][-1]),
                                  "q99_err": float(pred_q[qs >= 99][0] - ref_q[qs >= 99][0])}
                lims = [ref_q.min(), ref_q.max()]
                ax.plot(lims, lims, "k--", lw=1, label="perfect calibration")
                ax.set(xlabel="reference quantile", ylabel="predicted quantile",
                       title=f"{tag} value calibration (display channel)")
                ax.legend(fontsize=8)
                _save(fig, f"{stem}_qq_{tag}.png")
                report.setdefault("qq", {})[tag] = rows

    if "memgap" in args.diagnostics:
        ds_train = PatchDataset(patch_dir / "train_patches.npy", normalizer)
        m = min(args.mem_patches, len(ds_train), len(ds_plain))
        x_train = torch.stack([ds_train[i] for i in range(m)]).to(device)
        x_test = hf[:m]
        fig, ax = plt.subplots(figsize=(7, 4))
        rows = {}
        for mi, (name, mod, dif, coords, cfg_ck) in enumerate(models):
            t_centers = np.unique(np.linspace(
                1, dif.timesteps, args.t_bins).round().astype(int))
            u = (1.0 - t_centers / dif.timesteps).tolist()
            geo_on = cfg_ck.get("geo", {}).get("enabled", False)
            c_train = (_payload(patch_dir, normalizer, range(m), cfg_ck["geo"],
                                device, split="train") if geo_on else None)
            c_test = None if coords is None else coords[:m]
            tr = _binned_denoise_loss(mod, dif, x_train, c_train, t_centers, args.batch)
            te = _binned_denoise_loss(mod, dif, x_test, c_test, t_centers, args.batch)
            color = f"C{mi}"
            ax.plot(u, tr, "--", color=color, lw=1.2, label=f"{name} train")
            ax.plot(u, te, "-", color=color, lw=1.6, label=f"{name} test")
            rows[name] = {"u": u, "train": np.round(tr, 5).tolist(),
                          "test": np.round(te, 5).tolist(),
                          "gap": np.round(np.array(te) - np.array(tr), 5).tolist()}
        ax.set(xlabel="signal fraction u = 1 - t/T", ylabel="noise-prediction MSE",
               title=f"denoising loss by noise level ({m} patches/split)")
        ax.legend(fontsize=7)
        _save(fig, f"{stem}_memgap.png")
        report["memgap"] = rows

    with open(results_dir / f"{stem}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {results_dir / stem}.json")

    wb_run, wandb = init_wandb(
        cfg, job_type="diagnose",
        extra_config={"ckpts": args.ckpts, "projection": args.project,
                      "diagnostics": args.diagnostics},
        name=run_name(cfg, "diagnose", *(Path(c).stem for c in args.ckpts),
                      "proj" if args.project else ""))
    if wb_run is not None:
        for p in figures:
            wb_run.log({f"diagnose/{p.stem}": wandb.Image(str(p))})
        wb_run.finish()


if __name__ == "__main__":
    main()
