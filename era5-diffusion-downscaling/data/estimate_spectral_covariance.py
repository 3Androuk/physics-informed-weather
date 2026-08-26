"""Estimate a stationary T2M covariance spectrum from HR training patches.

The saved spectrum is consumed by ``sample.weather_ddnm``.  No neural network
is trained.  Patches are normalized with the same scalar statistics as the
diffusion model, then their mean periodogram is estimated in streaming batches.

Run on the machine/cluster that already holds the T2M patches::

    python -m data.estimate_spectral_covariance --config config/t2m.yaml

The default output is ``<patch_dir>/spectral_covariance.npz``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config  # noqa: E402


def _isotropize(power: np.ndarray, h: int, w: int) -> np.ndarray:
    """Radially average an rFFT spectrum while preserving its rFFT layout."""
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.rfftfreq(w) * w
    radius = np.rint(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(np.int32)
    out = np.empty_like(power)
    for channel in range(power.shape[0]):
        sums = np.bincount(radius.ravel(), weights=power[channel].ravel())
        counts = np.bincount(radius.ravel())
        radial = sums / np.maximum(counts, 1)
        out[channel] = radial[radius]
    return out


def _planar_detrend(x: np.ndarray) -> np.ndarray:
    """Remove a per-patch least-squares plane (mean + linear ramps).

    A patch of a weather field is a window on a much larger structure, so it
    carries a strong linear gradient. The periodogram of a non-periodic ramp
    leaks power across ALL wavenumbers (the discontinuity at the wrap edge
    looks like a broadband signal), which flattens the estimated spectrum
    toward white. Removing the plane first is the cheapest large reduction in
    that bias; it only touches the two lowest wavenumbers of real signal."""
    h, w = x.shape[-2:]
    ry = np.linspace(-1.0, 1.0, h)[:, None]
    rx = np.linspace(-1.0, 1.0, w)[None, :]
    x = x - x.mean(axis=(-2, -1), keepdims=True)
    # ry, rx are orthogonal and zero-mean, so the two fits are independent.
    # The basis norms run over the FULL 2D grid: ry is constant along x (hence
    # the extra factor w) and rx constant along y (factor h).
    x = x - (x * ry).sum(axis=(-2, -1), keepdims=True) / ((ry ** 2).sum() * w) * ry
    x = x - (x * rx).sum(axis=(-2, -1), keepdims=True) / ((rx ** 2).sum() * h) * rx
    return x


def _hann2d(h: int, w: int) -> np.ndarray:
    """Separable 2D Hann taper (periodic convention, matching the FFT grid)."""
    wy = np.hanning(h + 1)[:-1]
    wx = np.hanning(w + 1)[:-1]
    return np.outer(wy, wx)


def estimate(patches, mean: float, std: float, indices, batch_size: int,
             detrend: bool = True, window: str = "hann"):
    """Streaming mean periodogram in normalized model units.

    `detrend` and `window` control leakage suppression; both bias the estimate
    toward WHITE when disabled, which makes the projector behave more like
    ordinary DDNM (a conservative failure direction, not a spurious gain)."""
    _, channels, h, w = patches.shape
    if window not in {"none", "hann"}:
        raise ValueError("window must be 'none' or 'hann'")
    taper = _hann2d(h, w) if window == "hann" else None
    # Power normalization so the taper does not rescale the spectrum (the
    # overall scale cancels from K_C, but keeping it interpretable is free).
    taper_power = float((taper ** 2).mean()) if taper is not None else 1.0
    total = np.zeros((channels, h, w // 2 + 1), dtype=np.float64)
    seen = 0
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        x = np.asarray(patches[ids], dtype=np.float64)
        x = (x - mean) / std
        if detrend:
            x = _planar_detrend(x)
        if taper is not None:
            x = x * taper
        freq = np.fft.rfft2(x, axes=(-2, -1), norm="ortho")
        total += np.square(np.abs(freq)).sum(axis=0) / taper_power
        seen += len(ids)
        print(f"\rperiodograms: {seen:,}/{len(indices):,}", end="", flush=True)
    print()
    return total / max(seen, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/t2m.yaml")
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-patches", type=int, default=None,
                    help="Random subset size; default comes from weather_ddnm config.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--directional", action="store_true",
                    help="Keep the full 2D spectrum instead of radial averaging.")
    ap.add_argument("--no-detrend", action="store_true",
                    help="Skip the per-patch planar detrend (leakage control).")
    ap.add_argument("--window", choices=["none", "hann"], default=None,
                    help="Taper before the periodogram (leakage control); "
                         "default comes from weather_ddnm.estimation_window.")
    ap.add_argument("--localization-radius", type=float, default=None,
                    help="Gaspari-Cohn taper half-width in PIXELS applied to "
                         "the covariance kernel, confining the projection "
                         "correction to a physical neighborhood so the "
                         "projector's periodic embedding cannot wrap it across "
                         "the patch edge. Default from config; 0 disables.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["data"]["variable"] != "2m_temperature":
        raise ValueError("this first Weather-DDNM experiment is scoped to T2M")
    wc = cfg.get("weather_ddnm", {})
    patch_dir = Path(cfg["paths"]["patch_dir"])
    patch_path = patch_dir / "train_patches.npy"
    stats_path = patch_dir / "norm_stats.npz"
    patches = np.load(patch_path, mmap_mode="r")
    if patches.ndim != 4 or patches.shape[1] != 1:
        raise ValueError(f"expected single-channel T2M patches, got {patches.shape}")
    if len(patches) == 0:
        raise ValueError(f"no training patches found in {patch_path}")
    with np.load(stats_path) as stats:
        mean, std = float(stats["mean"]), float(stats["std"])
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError(f"invalid normalization statistics: mean={mean}, std={std}")

    max_patches = (int(wc.get("max_patches", 8192))
                   if args.max_patches is None else args.max_patches)
    if max_patches < 1 or args.batch_size < 1:
        raise ValueError("max-patches and batch-size must both be positive")
    n = min(max_patches, len(patches))
    seed = cfg["seed"] if args.seed is None else args.seed
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(patches), size=n, replace=False))
    window = args.window or wc.get("estimation_window", "hann")
    detrend = not args.no_detrend
    power = estimate(patches, mean, std, indices, args.batch_size,
                     detrend=detrend, window=window)
    mode = "directional" if args.directional else wc.get("spectrum", "isotropic")
    if mode not in {"isotropic", "directional"}:
        raise ValueError("weather_ddnm.spectrum must be isotropic or directional")
    h, w = patches.shape[-2:]
    if mode == "isotropic":
        power = _isotropize(power, h, w)

    # Shrink toward white covariance and apply a positive floor.  Overall scale
    # cancels from K_C; only the spectral shape matters.
    shrinkage = float(wc.get("shrinkage", 0.02))
    relative_floor = float(wc.get("relative_floor", 1e-4))
    if not 0 <= shrinkage <= 1 or relative_floor <= 0:
        raise ValueError("shrinkage must be in [0,1] and relative_floor > 0")
    white = power.mean(axis=(-2, -1), keepdims=True)
    power = (1.0 - shrinkage) * power + shrinkage * white
    power = np.maximum(power, relative_floor * white).astype(np.float32)

    # Localization: confine the covariance kernel to compact spatial support so
    # the projector's circular convolution cannot wrap the correction across
    # the (non-periodic) patch edge. Applied last so it tapers the final
    # shrunk/floored covariance.
    radius = (float(wc.get("localization_radius", h / 4.0))
              if args.localization_radius is None else args.localization_radius)
    if radius > 0:
        from sample.weather_ddnm import localize_spectrum
        power = localize_spectrum(power, (h, w), radius).numpy().astype(np.float32)
    if not np.isfinite(power).all() or np.any(power <= 0):
        raise ValueError("estimated covariance spectrum is not finite and positive")

    out = Path(args.output) if args.output else patch_dir / wc.get(
        "covariance_file", "spectral_covariance.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out, power=power, image_size=np.array([h, w], dtype=np.int64),
        channels=np.int64(patches.shape[1]), n_patches=np.int64(n),
        mean=np.float64(mean), std=np.float64(std), spectrum=np.array(mode),
        shrinkage=np.float64(shrinkage), relative_floor=np.float64(relative_floor),
        localization_radius=np.float64(radius), detrend=np.bool_(detrend),
        estimation_window=np.array(window), seed=np.int64(seed),
    )
    print(f"saved {out} | patches={n:,} | grid={h}x{w} | mode={mode} | "
          f"detrend={detrend} | window={window} | localization={radius:g}px | "
          f"power=[{power.min():.3e}, {power.max():.3e}]")


if __name__ == "__main__":
    main()
