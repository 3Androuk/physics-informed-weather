"""Conditional residual diffusion on the HEALPix mesh (CorrDiff-style).

The diffusion model learns the distribution of the RESIDUAL between the true
field and a deterministic mean prediction; the mean field is concatenated as a
conditioning channel at every denoising step (SR3-style conditioning). This is
the mesh version of the sibling project's split model, whose HEALPix arm
produced the best spectral score of that study.

Two mean fields are supported, mirroring the sibling's two phases:

  * `learned`  — a trained deterministic DLWP-HPX SR model (train/train_sr.py),
    frozen. Phase B.
  * `bilinear` — seam-aware bilinear upsampling of the coarse observation, the
    mesh analogue of the bicubic mean. Phase A; needs no extra checkpoint.

Splitting this way means the diffusion capacity is spent entirely on the
small-scale structure the regressor cannot produce — the smoothed high
wavenumbers that cost it the spectrum column against tiled/fused diffusion.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen_faces, upsample_bilinear_faces  # noqa: E402
from hpx.padding import HEALPixPadding  # noqa: E402
from models.hpx_unet import build_model  # noqa: E402


class HPXResidualUNet(nn.Module):
    """Noise predictor for residual diffusion on the mesh.

    forward(x_t, t, cond) with cond = (mean_field,):
      x_t: (B,12,1,F,F) noisy residual, mean_field: (B,12,1,F,F).
    The tuple-cond signature matches HPXGaussianDiffusion, which passes `cond`
    through to the model unchanged.
    """

    def __init__(self, base_unet: nn.Module):
        super().__init__()
        self.unet = base_unet

    def forward(self, x_t, t, cond):
        mean_field = cond[0] if isinstance(cond, (tuple, list)) else cond
        return self.unet(torch.cat([x_t, mean_field], dim=2), t)


def build_residual_model(cfg: dict) -> HPXResidualUNet:
    # +1 input channel for the mean field; a noise predictor has no global residual
    base = build_model(cfg, use_time=True, extra_in_channels=1)
    base.global_residual = False
    return HPXResidualUNet(base)


class MeanField(nn.Module):
    """Frozen deterministic mean predictor, evaluated under no_grad.

    `kind` is "learned" (a trained SR model) or "bilinear" (seam-aware
    upsampling of the coarse field). Both map the coarse observation to a full
    HR-grid field in normalized units.
    """

    def __init__(self, kind: str, ratio: int, model: nn.Module = None,
                 nside: int = None):
        super().__init__()
        if kind not in ("learned", "bilinear"):
            raise ValueError(f"unknown mean kind: {kind}")
        self.kind = kind
        self.ratio = int(ratio)
        self.model = model
        if model is not None:
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.padder = (HEALPixPadding(nside // ratio, 1)
                       if kind == "bilinear" and nside else None)

    @torch.no_grad()
    def forward(self, lf_up: torch.Tensor) -> torch.Tensor:
        """lf_up: the degraded field already on the HR grid (nearest-upsampled)."""
        if self.kind == "learned":
            return self.model(lf_up)
        coarse = coarsen_faces(lf_up, self.ratio)  # undo the nearest upsample
        return upsample_bilinear_faces(coarse, self.ratio, self.padder)


def load_mean_field(kind: str, ratio: int, nside: int, ckpt_path=None,
                    device="cpu") -> MeanField:
    """Build the mean predictor, loading the frozen SR checkpoint if learned."""
    model = None
    if kind == "learned":
        if ckpt_path is None:
            raise ValueError("mean kind 'learned' needs a checkpoint path")
        ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=False)
        model = build_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.grad_checkpoint = False  # frozen: never needs activation recompute
    mf = MeanField(kind, ratio, model, nside).to(device)
    mf.eval()
    return mf
