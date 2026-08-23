"""Degradation operators on HEALPix faces.

Because the nested ordering subdivides each face 2x2, average-pooling a face
grid by an integer ratio r is EXACTLY the HEALPix degrade from nside to
nside/r — the low-fidelity field is a genuine coarse HEALPix mesh, not an
arbitrary blur. The SR input is that coarse field brought back to the target
grid (nearest upsampling, i.e. each coarse pixel value repeated over its
children), matching the degrade->upsample convention of the sibling
era5-diffusion-downscaling project.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpx.padding import HEALPixPadding  # noqa: E402


def _merge(x: torch.Tensor):
    b, nf, c, h, w = x.shape
    return x.reshape(b * nf, c, h, w), (b, nf)


def _split(x: torch.Tensor, shape) -> torch.Tensor:
    b, nf = shape
    return x.reshape(b, nf, *x.shape[1:])


def coarsen_faces(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """(B, 12, C, F, F) -> (B, 12, C, F/r, F/r) by average pooling (= HEALPix degrade)."""
    if ratio == 1:
        return x
    if x.shape[-1] % ratio:
        raise ValueError(f"face size {x.shape[-1]} not divisible by ratio {ratio}")
    m, shape = _merge(x)
    return _split(F.avg_pool2d(m, ratio, ratio), shape)


def upsample_nearest_faces(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """(B, 12, C, f, f) -> (B, 12, C, r*f, r*f), each pixel repeated over children."""
    if ratio == 1:
        return x
    m, shape = _merge(x)
    return _split(F.interpolate(m, scale_factor=ratio, mode="nearest"), shape)


def degrade_faces(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """High-res faces -> low-fidelity input on the same grid (coarsen + nearest up)."""
    return upsample_nearest_faces(coarsen_faces(x, ratio), ratio)


def upsample_bilinear_faces(x: torch.Tensor, ratio: int,
                            padder: HEALPixPadding | None = None) -> torch.Tensor:
    """Seam-aware bilinear upsampling baseline.

    Pads each coarse face with a 1-pixel halo from its neighbors before
    bilinear interpolation, so the baseline (like the model) sees a seamless
    sphere instead of extrapolating at face edges.
    """
    if ratio == 1:
        return x
    f = x.shape[-1]
    if padder is None:
        padder = HEALPixPadding(f, 1).to(x.device)
    m, shape = _merge(x)
    m = padder(m)
    m = F.interpolate(m, scale_factor=ratio, mode="bilinear", align_corners=False)
    m = m[..., ratio:-ratio, ratio:-ratio]
    return _split(m, shape)
