"""Evaluation metrics: L2 (RMSE), radial power spectrum, value distribution.

Phase 1 replaces the paper's PDE equation-loss with the power spectrum and value
distribution (the physical-consistency metrics that survive without a known
governing equation). L2 is expected to be comparable across methods in-distribution;
the spectrum and the out-of-distribution L2 are where diffusion should win.
"""

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 4:        # (N, C, H, W) -> assume single channel
        x = x[:, 0]
    elif x.ndim == 2:      # (H, W) -> (1, H, W)
        x = x[None]
    return x               # (N, H, W)


def l2_norm(pred, truth) -> float:
    """Mean over samples of per-sample RMSE = sqrt(mean_grid (pred - truth)^2)."""
    p, t = _to_numpy(pred), _to_numpy(truth)
    per_sample = np.sqrt(((p - t) ** 2).mean(axis=(-2, -1)))
    return float(per_sample.mean())


def radial_power_spectrum(fields):
    """Radially-averaged 2D power spectrum E(k).

    Args:
        fields: (N, H, W) / (N, 1, H, W) / (H, W).
    Returns:
        k: (kmax+1,) integer wavenumbers.
        E: (kmax+1,) spectrum averaged over samples.
    """
    f = _to_numpy(fields)
    n, h, w = f.shape
    fhat = np.fft.fftshift(np.fft.fft2(f, axes=(-2, -1)), axes=(-2, -1))
    psd = (np.abs(fhat) ** 2) / (h * w)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.round(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(int)
    kmax = min(cy, cx)
    E = np.empty((n, kmax + 1))
    for k in range(kmax + 1):
        E[:, k] = psd[:, r == k].mean(axis=1)
    return np.arange(kmax + 1), E.mean(axis=0)


def spectrum_log_l1(pred, truth) -> float:
    """Mean absolute log10-spectrum error over wavenumbers k >= 1 (skip DC)."""
    _, ep = radial_power_spectrum(pred)
    _, et = radial_power_spectrum(truth)
    eps = 1e-30
    return float(np.mean(np.abs(np.log10(ep[1:] + eps) - np.log10(et[1:] + eps))))


def value_histogram(fields, bins=100, value_range=None):
    """Normalized histogram (density) of field values."""
    f = _to_numpy(fields).ravel()
    hist, edges = np.histogram(f, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist
