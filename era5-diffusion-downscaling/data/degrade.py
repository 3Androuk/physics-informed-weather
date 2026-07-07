"""Degradation operators: high-fidelity -> low-fidelity -> back onto the HF grid.

The diffusion model operates on the same input/output grid, so a low-fidelity
field is always coarsened and then upsampled (nearest, per the paper) back to the
target resolution before noise mixing. Average pooling is used for the coarsen
step (area downsample). An optional Gaussian pre-smoothing reduces the aliasing
introduced by nearest upsampling — the paper applies this to out-of-distribution
(8x) inputs.
"""

import math

import torch
import torch.nn.functional as F


def coarsen(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Average-pool downsample by an integer ratio. x: (N, C, H, W)."""
    if ratio == 1:
        return x
    assert x.shape[-1] % ratio == 0 and x.shape[-2] % ratio == 0, (
        f"grid {tuple(x.shape[-2:])} not divisible by ratio {ratio}"
    )
    return F.avg_pool2d(x, kernel_size=ratio, stride=ratio)


def upsample_nearest(x: torch.Tensor, size: int | tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour upsample to the target grid size."""
    if isinstance(size, int):
        size = (size, size)
    return F.interpolate(x, size=size, mode="nearest")


def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding. x: (N, C, H, W)."""
    if sigma <= 0:
        return x
    n, c, h, w = x.shape
    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


def degrade(x: torch.Tensor, ratio: int, smooth_sigma: float = 0.0) -> torch.Tensor:
    """Produce a low-fidelity field on the original HF grid.

    coarsen (avg-pool by `ratio`) -> nearest-upsample back to HF size ->
    optional Gaussian smoothing (anti-alias for out-of-distribution inputs).

    Args:
        x: (N, C, H, W) high-fidelity field.
        ratio: downsampling factor (e.g. 4 -> 32x32, 8 -> 16x16 from 128).
        smooth_sigma: Gaussian sigma applied after upsampling (0 = off).

    Returns:
        (N, C, H, W) low-fidelity field on the HF grid.
    """
    h, w = x.shape[-2:]
    lo = coarsen(x, ratio)
    up = upsample_nearest(lo, (h, w))
    if smooth_sigma > 0:
        up = gaussian_smooth(up, smooth_sigma)
    return up
