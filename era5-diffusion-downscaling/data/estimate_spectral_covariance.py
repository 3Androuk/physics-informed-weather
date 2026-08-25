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


def estimate(patches, mean: float, std: float, indices, batch_size: int):
    """Streaming mean periodogram in normalized model units."""
    _, channels, h, w = patches.shape
    total = np.zeros((channels, h, w // 2 + 1), dtype=np.float64)
    seen = 0
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        x = np.asarray(patches[ids], dtype=np.float64)
        x = (x - mean) / std
        freq = np.fft.rfft2(x, axes=(-2, -1), norm="ortho")
        total += np.square(np.abs(freq)).sum(axis=0)
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
    power = estimate(patches, mean, std, indices, args.batch_size)
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
        seed=np.int64(seed),
    )
    print(f"saved {out} | patches={n:,} | grid={h}x{w} | mode={mode} | "
          f"power=[{power.min():.3e}, {power.max():.3e}]")


if __name__ == "__main__":
    main()
