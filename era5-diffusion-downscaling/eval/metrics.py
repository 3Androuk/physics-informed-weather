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
    if x.ndim == 4:        # (N, C, H, W) -> channels become samples (N*C, H, W).
        # Identical to before for C == 1. Multi-channel callers wanting a
        # single variable's metric slice the channel first (mixing channels is
        # only meaningful in normalized units).
        x = x.reshape(-1, *x.shape[-2:])
    elif x.ndim == 2:      # (H, W) -> (1, H, W)
        x = x[None]
    return x               # (N, H, W)


def l2_per_channel(pred, truth) -> list:
    """Per-channel RMSE, one number per channel, in the inputs' own units.

    Use this instead of pooling when the inputs are in PHYSICAL units. Averaging
    physical RMSE across channels adds quantities with different units and lets
    the largest-magnitude variable define the score: in the 20-variable config,
    mean-sea-level pressure (~10^2 Pa error) and geopotential (~10^1 m2/s2)
    supply ~90% of the mean while 2m temperature contributes under 1% and
    specific humidity (~10^-4 kg/kg) contributes nothing measurable. Their
    standard deviations span a factor of ~10^6.
    """
    p, t = np.asarray(_as_nchw(pred)), np.asarray(_as_nchw(truth))
    return [float(np.sqrt(((p[:, c] - t[:, c]) ** 2).mean(axis=(-2, -1))).mean())
            for c in range(p.shape[1])]


def _as_nchw(x) -> np.ndarray:
    """(N, C, H, W) float64, without collapsing the channel axis."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x[None, None]
    if x.ndim == 3:
        return x[:, None]
    return x


def l2_norm(pred, truth) -> float:
    """Mean over samples of per-sample RMSE = sqrt(mean_grid (pred - truth)^2).

    Multi-channel inputs are pooled (channels treated as samples), which is only
    meaningful in NORMALIZED units — see l2_per_channel for why physical-unit
    pooling is dominated by whichever variable has the largest magnitude.
    """
    p, t = _to_numpy(pred), _to_numpy(truth)
    per_sample = np.sqrt(((p - t) ** 2).mean(axis=(-2, -1)))
    return float(per_sample.mean())


def radial_power_spectrum(fields):
    """Radially-averaged 2D power spectrum E(k).

    Args:
        fields: (N, H, W) / (N, C, H, W) (channels pooled as samples) / (H, W).
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


def radial_coherence(pred, truth):
    """Radially-averaged spectral coherence between prediction and reference.

    coh(k) = |sum P_xy| / sqrt(sum P_xx * sum P_yy), with the sums over samples
    and the annulus at wavenumber k. 1 = the predicted structure at that scale
    is phase-locked to the reference (informative), 0 = uncorrelated (invented
    texture). Distinguishes scales the model RECONSTRUCTS from scales it merely
    HALLUCINATES with the right power — the radial power spectrum alone cannot
    tell those apart.

    Returns (k, coh) like radial_power_spectrum.
    """
    p, t = _to_numpy(pred), _to_numpy(truth)
    fp = np.fft.fftshift(np.fft.fft2(p, axes=(-2, -1)), axes=(-2, -1))
    ft = np.fft.fftshift(np.fft.fft2(t, axes=(-2, -1)), axes=(-2, -1))
    _, h, w = p.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.round(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)).astype(int)
    kmax = min(cy, cx)
    coh = np.empty(kmax + 1)
    for k in range(kmax + 1):
        m = r == k
        pxy = (fp[:, m] * np.conj(ft[:, m])).sum()
        pxx = (np.abs(fp[:, m]) ** 2).sum()
        pyy = (np.abs(ft[:, m]) ** 2).sum()
        coh[k] = np.abs(pxy) / max(np.sqrt(pxx * pyy), 1e-30)
    return np.arange(kmax + 1), coh


def spectrum_log_l1(pred, truth) -> float:
    """Mean absolute log10-spectrum error over wavenumbers k >= 1 (skip DC)."""
    _, ep = radial_power_spectrum(pred)
    _, et = radial_power_spectrum(truth)
    eps = 1e-30
    return float(np.mean(np.abs(np.log10(ep[1:] + eps) - np.log10(et[1:] + eps))))


def crps_ensemble(members, truth) -> float:
    """Fair empirical-ensemble CRPS, averaged over samples and pixels.

    CRPS = mean_i |x_i - y| - (1 / (2 M (M-1))) sum_{i != j} |x_i - x_j|

    Args:
        members: sequence of M predictions, each (N, H, W)-like.
        truth: (N, H, W)-like reference.
    """
    p = np.stack([_to_numpy(m) for m in members])  # (M, N, H, W)
    t = _to_numpy(truth)[None]
    m = p.shape[0]
    assert m >= 2, "CRPS needs at least 2 ensemble members"
    term1 = np.abs(p - t).mean()
    spread = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            spread += np.abs(p[i] - p[j]).mean()
    return float(term1 - spread / (m * (m - 1)))


def value_histogram(fields, bins=100, value_range=None):
    """Normalized histogram (density) of field values."""
    f = _to_numpy(fields).ravel()
    hist, edges = np.histogram(f, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist
