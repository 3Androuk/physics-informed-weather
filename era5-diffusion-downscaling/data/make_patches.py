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
                 rng: np.random.Generator):
    """Random square crops. fields: (T, H, W) -> (patches, origins).

    patches: (N, 1, size, size); origins: (N, 2) int (row, col) of each crop's
    top-left corner, used to recover per-pixel (lat, lon) for geo-conditioning.
    """
    t, h, w = fields.shape
    assert h >= size and w >= size, f"field {h}x{w} smaller than patch {size}"
    out = np.empty((t * per_field, 1, size, size), dtype=np.float32)
    origins = np.empty((t * per_field, 2), dtype=np.int32)
    idx = 0
    for f in range(t):
        for _ in range(per_field):
            r = rng.integers(0, h - size + 1)
            c = rng.integers(0, w - size + 1)
            out[idx, 0] = fields[f, r:r + size, c:c + size]
            origins[idx] = (r, c)
            idx += 1
    return out, origins


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    """Write via a .tmp then rename, so an interrupted save can't leave a
    half-written .npy that a later run would mistake for complete."""
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:  # file object: np.save won't append .npy
        np.save(fh, arr)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Crop Z500 patches + norm stats.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--force", action="store_true",
                    help="regenerate patches even if the outputs already exist.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    patch_dir = ensure_dir(cfg["paths"]["patch_dir"])
    size = cfg["patches"]["size"]
    per_field = cfg["patches"]["per_field"]
    seed = cfg["seed"]

    train_out = patch_dir / "train_patches.npy"
    test_out = patch_dir / "test_patches.npy"
    stats_out = patch_dir / "norm_stats.npz"

    train_origins_out = patch_dir / "train_origins.npy"
    test_origins_out = patch_dir / "test_origins.npy"

    # ── Train patches + z-score stats (skip if already built) ─────────────
    # The two splits use independent RNGs so either can be regenerated on its
    # own without perturbing the other.
    if (not args.force and train_out.exists() and stats_out.exists()
            and train_origins_out.exists()):
        print(f"[skip] {train_out.name} + {stats_out.name} already exist")
    else:
        train_fields = np.load(raw_dir / "train.npy", mmap_mode="r")  # slices read lazily
        rng = np.random.default_rng(seed)
        train_patches, train_origins = crop_patches(train_fields, size, per_field, rng)
        # Shuffle to decorrelate consecutive crops from the same field.
        perm = rng.permutation(len(train_patches))
        train_patches = train_patches[perm]
        train_origins = train_origins[perm]
        mean = float(train_patches.mean())
        std = float(train_patches.std())
        _save_npy_atomic(train_out, train_patches)
        _save_npy_atomic(train_origins_out, train_origins)
        np.savez(stats_out, mean=mean, std=std, size=size)
        print(f"train patches {train_patches.shape} | z-score mean={mean:.3f} std={std:.3f}")
        del train_fields, train_patches, train_origins

    # ── Test patches (skip if already built) ──────────────────────────────
    if not args.force and test_out.exists() and test_origins_out.exists():
        print(f"[skip] {test_out.name} already exists")
    else:
        test_fields = np.load(raw_dir / "test.npy", mmap_mode="r")
        rng = np.random.default_rng(seed + 1)
        test_patches, test_origins = crop_patches(test_fields, size, per_field, rng)
        _save_npy_atomic(test_out, test_patches)
        _save_npy_atomic(test_origins_out, test_origins)
        print(f"test patches {test_patches.shape}")
        del test_fields, test_patches, test_origins

    # Copy global lat/lon so the dataset can recover per-pixel coordinates for
    # geo-conditioning (origins index into these arrays).
    coords = np.load(raw_dir / "coords.npz")
    np.savez(patch_dir / "coords_full.npz", lat=coords["lat"], lon=coords["lon"])

    print(f"-> {patch_dir}")


if __name__ == "__main__":
    main()
