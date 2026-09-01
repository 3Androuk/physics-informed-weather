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


def latitude_weights(lat) -> np.ndarray:
    """Area weights cos(phi) for a lat-lon grid, normalized to unit mean.

    `lat` is degrees, shape (H,) or (N, H). Equal-angle cells shrink poleward
    as cos(latitude), so an unweighted grid mean over-counts high latitudes;
    the WeatherBench2 convention normalizes cos(phi) to mean 1 so weighted and
    unweighted scores are directly comparable in magnitude.
    """
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=np.float64)))
    w = np.clip(w, 0.0, None)
    mean = w.mean(axis=-1, keepdims=True)
    return w / np.where(mean > 0, mean, 1.0)


def l2_norm_weighted(pred, truth, lat) -> float:
    """Latitude-weighted RMSE: the standard WeatherBench2-style score.

    Identical in form to `l2_norm` but each row is weighted by cos(latitude)
    (unit mean), so a patch spanning many degrees is not dominated by its
    poleward rows. `lat` is degrees, shape (H,) for a shared grid or (N, H)
    for per-patch latitudes.
    """
    p, t = _to_numpy(pred), _to_numpy(truth)
    w = latitude_weights(lat)
    if w.ndim == 1:
        w = np.broadcast_to(w, (p.shape[0], w.shape[0]))
    if w.shape[0] != p.shape[0]:
        raise ValueError(f"latitude rows {w.shape[0]} != samples {p.shape[0]}")
    if w.shape[-1] != p.shape[-2]:
        raise ValueError(f"latitude length {w.shape[-1]} != field height {p.shape[-2]}")
    sq = ((p - t) ** 2).mean(axis=-1)              # (N, H) mean over longitude
    per_sample = np.sqrt((sq * w).mean(axis=-1))   # weighted mean over latitude
    return float(per_sample.mean())


def patch_latitudes(patch_dir, n: int, size: int, split: str = "test") -> np.ndarray:
    """Per-patch latitude rows (n, size) from saved origins + full-grid coords.

    Returns None when the geo artifacts are absent (legacy patch dirs), so
    callers can fall back to unweighted metrics.
    """
    from pathlib import Path
    patch_dir = Path(patch_dir)
    origins_path = patch_dir / f"{split}_origins.npy"
    coords_path = patch_dir / "coords_full.npz"
    if not (origins_path.exists() and coords_path.exists()):
        return None
    origins = np.load(origins_path)[:n]
    lat_full = np.load(coords_path)["lat"]
    return np.stack([lat_full[int(r):int(r) + size] for r, _ in origins])


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
