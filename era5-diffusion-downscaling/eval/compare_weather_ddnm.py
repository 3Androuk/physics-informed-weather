"""T2M ablation for covariance-aware DDNM and coarse-informed initialization.

The primary arms use the same trained diffusion prior and the same
initialization noise on every patch:

1. guided diffusion without a projection;
2. ordinary DDNM;
3. spectral Weather-DDNM at every denoising step;
4. deterministic bicubic interpolation.

By default, two additional initialization ablations compare ordinary and
spectral DDNM with the covariance lift ``K_C y`` at the first outer loop.

Run after estimating the covariance artifact::

    python -m data.estimate_spectral_covariance --config config/t2m.yaml
    python -m eval.compare_weather_ddnm --config config/t2m.yaml
"""

from __future__ import annotations

import argparse
import copy
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
from data.degrade import coarsen, degrade  # noqa: E402
from eval.metrics import l2_norm, radial_power_spectrum, spectrum_log_l1  # noqa: E402
from sample.reconstruct import (load_diffusion, reconstruct_bicubic,  # noqa: E402
                                reconstruct_diffusion)
from sample.weather_ddnm import SpectralCovarianceProjector  # noqa: E402
from utils import (ensure_dir, get_device, init_wandb, load_config,  # noqa: E402
                   run_name)


def _geo_payload(patch_dir, normalizer, n, geo_cfg, device):
    dataset = PatchDataset(
        patch_dir / "test_patches.npy", normalizer,
        origins_path=patch_dir / "test_origins.npy",
        coords_full_path=patch_dir / "coords_full.npz",
        geo_input_dim=geo_cfg["input_dim"], altitude=geo_cfg["altitude"],
        geo_encoder=geo_cfg.get("encoder", "hash"),
        healpix_index_path=((patch_dir / geo_cfg["healpix_index"])
                            if geo_cfg.get("healpix_index") else None),
    )
    return torch.stack([dataset[i][1] for i in range(n)]).to(device)


def _reconstruct(diffusion, model, hf, coords, ratio, recon_cfg, projector,
                 arm, init_noise, batch, eta):
    """Run one arm, slicing the shared CPU noise to keep comparisons paired."""
    settings = {
        "no_projection": {"project": False},
        "ddnm": {"project": True},
        "ddnm_cov_init": {
            "project": True,
            "covariance_init": True,
            "covariance_init_projector": projector,
        },
        "weather_ddnm": {
            "project": True,
            "covariance_projector": projector,
        },
        "weather_ddnm_cov_init": {
            "project": True,
            "covariance_projector": projector,
            "covariance_init": True,
        },
    }[arm]
    iterator = range(0, len(hf), batch)
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=f"{ratio}x {arm}")
    except ImportError:
        pass
    output = []
    for start in iterator:
        stop = min(start + batch, len(hf))
        batch_coords = None if coords is None else coords[start:stop]
        noise = init_noise[:, start:stop].to(hf.device)
        output.append(reconstruct_diffusion(
            diffusion, model, hf[start:stop], ratio, recon_cfg,
            eta=eta, coords=batch_coords, init_noise=noise,
            **settings,
        ).cpu())
    return torch.cat(output)


def _coarse_rmse_kelvin(pred, truth, ratio, std):
    residual = coarsen(pred, ratio) - coarsen(truth, ratio)
    return float(residual.square().mean().sqrt() * std)


def _plot_qualitative(normalizer, hf, predictions, ratio, path, index=0):
    coarse = degrade(hf[index:index + 1], ratio)
    panels = {"Coarse input": coarse.cpu(),
              **{name: value[index:index + 1] for name, value in predictions.items()},
              "Reference": hf[index:index + 1].cpu()}
    reference = normalizer.decode(hf[index:index + 1].cpu())[0, 0].numpy()
    vmin, vmax = np.quantile(reference, [0.01, 0.99])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5))
    image = None
    for ax, (name, value) in zip(axes, panels.items()):
        field = normalizer.decode(value)[0, 0].numpy()
        image = ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(name.replace("_", " "))
        ax.axis("off")
    fig.colorbar(image, ax=axes, shrink=0.78, label="T2M (K)")
    fig.suptitle(f"Weather-DDNM, {ratio}x (test patch {index})")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_spectra(reference, spectra, path):
    k_ref, p_ref = radial_power_spectrum(reference)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(k_ref[1:], p_ref[1:], color="black", linewidth=2, label="Reference")
    for label, fields in spectra.items():
        k, power = radial_power_spectrum(fields)
        ax.loglog(k[1:], power[1:], label=label.replace("_", " "))
    ax.set(xlabel="Wavenumber", ylabel="Power", title="T2M radial power spectrum")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _resolve_covariance_path(argument, patch_dir, configured):
    if argument is None:
        return patch_dir / configured
    path = Path(argument)
    return path if path.is_absolute() else Path.cwd() / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/t2m.yaml")
    parser.add_argument("--ckpt", default="diffusion.pt",
                        help="Checkpoint name in paths.ckpt_dir or an absolute path.")
    parser.add_argument("--covariance", default=None,
                        help="Override the configured covariance artifact path.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-patches", type=int, default=None)
    parser.add_argument("--t0", type=int, default=None,
                        help="Use a single outer loop starting at this DDIM time.")
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--wandb", action="store_true",
                        help="Upload metrics, figures, and covariance artifact to W&B.")
    parser.add_argument("--primary-only", action="store_true",
                        help="Compare no projection, ordinary DDNM, spectral "
                             "Weather-DDNM, and bicubic; skip initialization ablations.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    if cfg["data"]["variable"] != "2m_temperature" or cfg["unet"]["in_channels"] != 1:
        raise ValueError("this Weather-DDNM experiment is intentionally scoped to T2M")
    device = get_device()
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_arg = Path(args.ckpt)
    ckpt_path = ckpt_arg if ckpt_arg.is_absolute() else Path(cfg["paths"]["ckpt_dir"]) / ckpt_arg
    weather_cfg = cfg.get("weather_ddnm", {})
    covariance_path = _resolve_covariance_path(
        args.covariance, patch_dir, weather_cfg.get("covariance_file", "spectral_covariance.npz"))

    normalizer = load_norm_stats(patch_dir)
    plain_dataset = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    requested = cfg["eval"]["n_test_patches"] if args.n_patches is None else args.n_patches
    n = min(int(requested), len(plain_dataset))
    if n < 1 or args.batch < 1:
        raise ValueError("n-patches and batch must both be positive")
    hf = torch.stack([plain_dataset[i] for i in range(n)]).to(device)
    hf_physical = normalizer.decode(hf.cpu())

    model, diffusion, ckpt_cfg = load_diffusion(ckpt_path, device)
    geo_cfg = ckpt_cfg.get("geo", {})
    coords = (_geo_payload(patch_dir, normalizer, n, geo_cfg, device)
              if geo_cfg.get("enabled", False) else None)
    projector = SpectralCovarianceProjector.from_npz(
        covariance_path, inverse_floor=float(weather_cfg.get("inverse_floor", 1e-7)))
    expected_size = tuple(hf.shape[-2:])
    if projector.image_size != expected_size:
        raise ValueError(
            f"covariance grid {projector.image_size} != test patch grid {expected_size}")
    projector.to(device=device, dtype=hf.dtype)

    eta = cfg["sample"]["ddim_eta"] if args.eta is None else args.eta
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if args.t0 is not None and not 1 <= args.t0 <= diffusion.timesteps:
        raise ValueError(f"t0 must be in [1, {diffusion.timesteps}]")
    if eta != 0:
        print("warning: matched initialization noise is still used, but eta > 0 adds "
              "fresh per-step noise and makes arm comparisons less tightly paired")
    print("T2M note: a scalar diagonal covariance cancels from K_C and is exactly "
          "ordinary DDNM, so it is not included as a redundant arm.")
    print(f"checkpoint={ckpt_path} | covariance={covariance_path} | patches={n} | "
          f"device={device}")

    arms = ["no_projection", "ddnm", "weather_ddnm"]
    if not args.primary_only:
        arms += ["ddnm_cov_init", "weather_ddnm_cov_init"]
    results = {}
    output_dir = ensure_dir(Path(cfg["paths"]["results_dir"]) / "weather_ddnm")
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for base_recon_cfg in cfg["sample"]["reconstructions"]:
        recon_cfg = copy.deepcopy(base_recon_cfg)
        if args.t0 is not None:
            recon_cfg["K"] = 1
            recon_cfg["t_steps"] = [args.t0]
        ratio = int(recon_cfg["ratio"])
        # One shared draw for all arms at this ratio. Keep it on CPU until each
        # batch is evaluated so the harness does not reserve K*N extra GPU fields.
        noise = torch.randn(
            (recon_cfg["K"], n, *hf.shape[1:]), generator=generator,
            dtype=hf.dtype, device="cpu")
        predictions = {}
        row = {}
        for arm in arms:
            prediction = _reconstruct(
                diffusion, model, hf, coords, ratio, recon_cfg, projector,
                arm, noise, args.batch, eta)
            predictions[arm] = prediction
            physical = normalizer.decode(prediction)
            row[arm] = {
                "l2_kelvin": l2_norm(physical, hf_physical),
                "spectrum_log_l1": spectrum_log_l1(physical, hf_physical),
                "coarse_rmse_kelvin": _coarse_rmse_kelvin(
                    prediction, hf.cpu(), ratio, normalizer.std),
            }
            print(f"  {ratio}x {arm:24s} | L2 {row[arm]['l2_kelvin']:.4f} K | "
                  f"spectrum {row[arm]['spectrum_log_l1']:.4f} | "
                  f"coarse {row[arm]['coarse_rmse_kelvin']:.3e} K")
        bicubic = reconstruct_bicubic(hf, ratio).cpu()
        predictions["bicubic"] = bicubic
        bicubic_physical = normalizer.decode(bicubic)
        row["bicubic"] = {
            "l2_kelvin": l2_norm(bicubic_physical, hf_physical),
            "spectrum_log_l1": spectrum_log_l1(bicubic_physical, hf_physical),
            "coarse_rmse_kelvin": _coarse_rmse_kelvin(
                bicubic, hf.cpu(), ratio, normalizer.std),
        }
        print(f"  {ratio}x {'bicubic':24s} | L2 {row['bicubic']['l2_kelvin']:.4f} K | "
              f"spectrum {row['bicubic']['spectrum_log_l1']:.4f} | "
              f"coarse {row['bicubic']['coarse_rmse_kelvin']:.3e} K")
        results[f"{ratio}x"] = row
        physical_predictions = {name: normalizer.decode(value)
                                for name, value in predictions.items()}
        _plot_qualitative(normalizer, hf, predictions, ratio,
                          output_dir / f"qualitative_{ratio}x.png")
        _plot_spectra(hf_physical, physical_predictions,
                      output_dir / f"spectrum_{ratio}x.png")

    metadata = {
        "checkpoint": str(ckpt_path), "covariance": str(covariance_path),
        "seed": seed, "n_patches": n, "eta": eta, "t0_override": args.t0,
        "primary_only": args.primary_only,
        "metrics": results,
    }
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved Weather-DDNM ablation to {output_dir}")

    wb_run, wandb = init_wandb(
        cfg, job_type="compare_weather_ddnm",
        extra_config={
            "weather_ddnm_eval": {
                "checkpoint": str(ckpt_path),
                "covariance": str(covariance_path),
                "seed": seed,
                "n_test_patches": n,
                "eta": eta,
                "t0_override": args.t0,
                "primary_only": args.primary_only,
                "arms": list(arms) + ["bicubic"],
            },
        },
        name=run_name(
            cfg, "weather-ddnm", Path(args.ckpt).stem,
            "primary" if args.primary_only else "full-ablation",
            f"seed{seed}"),
    )
    if wb_run is not None:
        table = wandb.Table(columns=[
            "ratio", "method", "l2_kelvin", "spectrum_log_l1",
            "coarse_rmse_kelvin",
        ])
        for ratio_tag, row in results.items():
            for method, values in row.items():
                table.add_data(
                    ratio_tag, method, values["l2_kelvin"],
                    values["spectrum_log_l1"], values["coarse_rmse_kelvin"])
                for metric, value in values.items():
                    wb_run.summary[f"{ratio_tag}/{method}/{metric}"] = value

        log = {"weather_ddnm/metrics": table}
        for recon_cfg in cfg["sample"]["reconstructions"]:
            ratio_tag = f"{int(recon_cfg['ratio'])}x"
            log[f"weather_ddnm/qualitative_{ratio_tag}"] = wandb.Image(
                str(output_dir / f"qualitative_{ratio_tag}.png"))
            log[f"weather_ddnm/spectrum_{ratio_tag}"] = wandb.Image(
                str(output_dir / f"spectrum_{ratio_tag}.png"))
        wb_run.log(log)

        result_artifact = wandb.Artifact(
            name=f"weather-ddnm-results-{wb_run.id}", type="evaluation",
            metadata={"seed": seed, "n_test_patches": n,
                      "primary_only": args.primary_only})
        result_artifact.add_file(str(output_dir / "metrics.json"))
        wb_run.log_artifact(result_artifact)

        covariance_artifact = wandb.Artifact(
            name="t2m-spectral-covariance", type="weather-covariance",
            metadata={**weather_cfg, "source_path": str(covariance_path)})
        covariance_artifact.add_file(str(covariance_path))
        wb_run.log_artifact(covariance_artifact)
        wb_run.finish()
        print(f"wandb: uploaded run {wb_run.name}")


if __name__ == "__main__":
    main()
