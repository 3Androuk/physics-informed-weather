"""Conditional residual diffusion model (CorrDiff-style, mean-agnostic).

The diffusion model learns the distribution of the RESIDUAL between the
high-fidelity field and a deterministic mean prediction (Phase A: bicubic
upsampling of the coarse input). The mean field is concatenated as a
conditioning channel at every denoising step (SR3-style conditioning), plus the
per-pixel geo embedding when enabled. Unlike the guided unconditional model,
consistency with the input is learned at training time, not imposed at
sampling time — though the exact DDNM projection can still be applied to the
composed output as a final guarantee.
"""

import torch
import torch.nn as nn

from models.unet import build_unet


class ResidualConditionalUNet(nn.Module):
    """Noise predictor for residual diffusion.

    forward(x_t, t, cond) where cond = (mean_field, coords-or-None):
      x_t: (B,C,H,W) noisy residual; mean_field: (B,C,H,W) the deterministic
      mean prediction on the HF grid; coords: (B,H,W,d) for geo conditioning.
    The tuple-cond signature matches GaussianDiffusion.training_loss /
    sample_unconditional, which pass `cond` through to the model unchanged.
    """

    def __init__(self, base_unet: nn.Module, geo_encoder=None, level_gate=None,
                 ddpm_timesteps=None):
        super().__init__()
        self.unet = base_unet
        self.geo = geo_encoder
        self.gate = level_gate
        self.ddpm_timesteps = ddpm_timesteps

    def forward(self, x_t, t, cond):
        mean_field, coords = cond
        chans = [x_t, mean_field]
        if self.geo is not None:
            emb = self.geo(coords)
            if self.gate is not None and t is not None and self.ddpm_timesteps:
                u = 1.0 - t.float() / self.ddpm_timesteps
                emb = self.gate(emb, u.clamp(0.0, 1.0))
            chans.append(emb.permute(0, 3, 1, 2).contiguous())
        return self.unet(torch.cat(chans, dim=1), t)


def build_residual_model(cfg: dict) -> ResidualConditionalUNet:
    geo_on = cfg.get("geo", {}).get("enabled", False)
    mean_channels = cfg["unet"]["in_channels"]  # mean field has one channel per variable
    if geo_on:
        from models.geo_encoding import build_geo_encoder, build_level_gate
        geo_enc = build_geo_encoder(cfg)
        gate = build_level_gate(cfg)
        extra = mean_channels + geo_enc.output_dim
    else:
        geo_enc, gate = None, None
        extra = mean_channels
    base = build_unet(cfg, use_time=True, extra_in_channels=extra)
    return ResidualConditionalUNet(
        base, geo_enc, level_gate=gate,
        ddpm_timesteps=cfg.get("diffusion", {}).get("timesteps"))
