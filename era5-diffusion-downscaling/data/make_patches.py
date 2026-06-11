"""Crop high-fidelity Z500 patches and compute normalization statistics.

Reads the raw per-split fields produced by download_era5.py, extracts random
square patches (time-based split is already baked into train.npy / test.npy),
and saves patch tensors plus the train-set z-score statistics.

Run:
    python -m data.make_patches --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config  # noqa: E402


def crop_patches(fields: np.ndarray, size: int, per_field: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Random square crops. fields: (T, H, W) -> patches: (N, 1, size, size)."""
    t, h, w = fields.shape
    assert h >= size and w >= size, f"field {h}x{w} smaller than patch {size}"
    out = np.empty((t * per_field, 1, size, size), dtype=np.float32)
    idx = 0
    for f in range(t):
        for _ in range(per_field):
            r = rng.integers(0, h - size + 1)
            c = rng.integers(0, w - size + 1)
            out[idx, 0] = fields[f, r:r + size, c:c + size]
            idx += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Crop Z500 patches + norm stats.")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    patch_dir = ensure_dir(cfg["paths"]["patch_dir"])
    size = cfg["patches"]["size"]
    per_field = cfg["patches"]["per_field"]
    rng = np.random.default_rng(cfg["seed"])

    train_fields = np.load(raw_dir / "train.npy")
    test_fields = np.load(raw_dir / "test.npy")

    train_patches = crop_patches(train_fields, size, per_field, rng)
    test_patches = crop_patches(test_fields, size, per_field, rng)

    # Shuffle train patches (decorrelate consecutive crops from the same field).
    perm = rng.permutation(len(train_patches))
    train_patches = train_patches[perm]

    mean = float(train_patches.mean())
    std = float(train_patches.std())

    np.save(patch_dir / "train_patches.npy", train_patches)
    np.save(patch_dir / "test_patches.npy", test_patches)
    np.savez(patch_dir / "norm_stats.npz", mean=mean, std=std, size=size)

    print(
        f"train patches {train_patches.shape}, test patches {test_patches.shape}\n"
        f"z-score mean={mean:.3f} std={std:.3f} -> {patch_dir}"
    )


if __name__ == "__main__":
    main()
