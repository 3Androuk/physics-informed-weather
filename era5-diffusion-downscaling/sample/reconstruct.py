"""Reconstruction: guided diffusion (any ratio, one model) + baselines.

All functions operate in NORMALIZED units; the caller denormalizes for physical
metrics. The diffusion reconstructor uses noise-mixing + intermediate-start DDIM
guided sampling (Algorithm 2, physics term dropped). The SAME diffusion model
serves every ratio — only the LF guidance and the (K, t_steps) schedule change.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen, degrade  # noqa: E402
from models.diffusion import build_diffusion  # noqa: E402
from models.unet import build_unet  # noqa: E402


def load_diffusion(ckpt_path, device, use_ema=True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    if cfg.get("geo", {}).get("enabled", False):
        from models.geo_encoding import GeoConditionedUNet, build_geo_encoder
        geo_enc = build_geo_encoder(cfg)
        base = build_unet(cfg, use_time=True, extra_in_channels=geo_enc.output_dim)
        model = GeoConditionedUNet(base, geo_enc)
    else:
        model = build_unet(cfg, use_time=True)
    model.load_state_dict(ck["ema"] if (use_ema and "ema" in ck) else ck["model"])
    model.eval().to(device)
    diffusion = build_diffusion(cfg).to(device)
    return model, diffusion, cfg


def load_directmap(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_unet(cfg, use_time=False)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return model, cfg


@torch.no_grad()
def reconstruct_diffusion(diffusion, model, hf_norm, ratio, recon_cfg,
                          eta=0.0, progress=False, coords=None):
    """Guided diffusion reconstruction of `hf_norm` degraded at `ratio`.

    `coords` (B,H,W,d) is required for a geo-conditioned model and ignored
    otherwise.
    """
    x_g = degrade(hf_norm, ratio, smooth_sigma=recon_cfg.get("smooth_sigma", 0.0))
    return diffusion.guided_reconstruct(
        model, x_g, t_steps=recon_cfg["t_steps"], K=recon_cfg["K"], eta=eta,
        progress=progress, cond=coords,
    )


@torch.no_grad()
def reconstruct_bicubic(hf_norm, ratio):
    """Classical bicubic upsampling from the coarsened field."""
    h, w = hf_norm.shape[-2:]
    lo = coarsen(hf_norm, ratio)
    return F.interpolate(lo, size=(h, w), mode="bicubic", align_corners=False)


@torch.no_grad()
def reconstruct_directmap(model, hf_norm, ratio, smooth_sigma=0.0):
    """Direct-mapping baseline f(X) on a (possibly OOD) ratio."""
    x = degrade(hf_norm, ratio, smooth_sigma=smooth_sigma)
    return model(x)


def _cli():
    import argparse

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.dataset import PatchDataset, load_norm_stats
    from utils import ensure_dir, get_device, load_config

    ap = argparse.ArgumentParser(description="Single-ratio reconstruction demo.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--ratio", type=int, default=4)
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    patch_dir = Path(cfg["paths"]["patch_dir"])
    normalizer = load_norm_stats(patch_dir)
    ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    hf = ds[args.index].unsqueeze(0).to(device)

    recon_cfg = next(r for r in cfg["sample"]["reconstructions"] if r["ratio"] == args.ratio)
    model, diffusion, _ = load_diffusion(Path(cfg["paths"]["ckpt_dir"]) / "diffusion.pt", device)

    diff = reconstruct_diffusion(diffusion, model, hf, args.ratio, recon_cfg,
                                 eta=cfg["sample"]["ddim_eta"], progress=True)
    bic = reconstruct_bicubic(hf, args.ratio)
    lf = degrade(hf, args.ratio, recon_cfg.get("smooth_sigma", 0.0))

    panels = {
        "Input (LF)": lf, "Bicubic": bic, "Diffusion": diff, "Reference": hf,
    }
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (title, t) in zip(axes, panels.items()):
        img = normalizer.decode(t.cpu())[0, 0].numpy()
        ax.imshow(img, cmap="RdBu_r")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{args.ratio}x reconstruction (test patch {args.index})")
    fig.tight_layout()
    out = ensure_dir(cfg["paths"]["results_dir"]) / f"recon_{args.ratio}x_{args.index}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    _cli()
