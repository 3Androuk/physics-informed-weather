"""Metrics, copied verbatim from the sibling era5-diffusion-downscaling
project's eval/metrics.py so numbers are directly comparable between the two
studies. Do not "improve" these — their value is that they are identical.

On the mesh these are applied per face: HEALPix faces are regular square grids
of equal-area pixels, so a per-face radial power spectrum is well defined and
averaging it across the 12 faces is area-fair (no latitude weighting).
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
    """Radially-averaged 2D power spectrum E(k). fields: (N, H, W)."""
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


def faces_as_images(faces) -> np.ndarray:
    """(N, 12, F, F) or (N, 12, 1, F, F) -> (N*12, F, F) for the 2-D metrics."""
    f = np.asarray(faces.detach().cpu().numpy() if isinstance(faces, torch.Tensor)
                   else faces, dtype=np.float64)
    if f.ndim == 5:       # (N, 12, C, F, F) -> drop the channel axis
        f = f[:, :, 0]
    return f.reshape(-1, *f.shape[-2:])
