"""Crop high-fidelity patches and compute normalization statistics.

Reads the raw per-split fields produced by download_era5.py — (T, C, H, W)
multi-channel, or legacy (T, H, W) single-channel — extracts random square
patches (time-based split is already baked into train.npy / test.npy), and
saves patch tensors plus the per-channel train-set z-score statistics.

Patches are written straight into an on-disk .npy memmap: a 20-channel patch
set runs to tens of GB and must never be materialized in RAM. The train split
is shuffled by writing each crop to a pre-permuted position.

Run:
    python -m data.make_patches --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import ensure_dir, load_config  # noqa: E402


def _progress(iterable, desc):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc)
    except ImportError:
        return iterable


def crop_patches(fields: np.ndarray, size: int, per_field: int,
                 rng: np.random.Generator, desc: str = "crop"):
    """In-RAM random square crops (small datasets / sanity checks only).

    fields: (T, C, H, W) or legacy (T, H, W) -> (patches, origins).
    patches: (N, C, size, size); origins: (N, 2) int (row, col) of each crop's
    top-left corner, used to recover per-pixel (lat, lon) for geo-conditioning.
    """
    if fields.ndim == 3:
        fields = fields[:, None]
    t, c, h, w = fields.shape
    assert h >= size and w >= size, f"field {h}x{w} smaller than patch {size}"
    out = np.empty((t * per_field, c, size, size), dtype=np.float32)
    origins = np.empty((t * per_field, 2), dtype=np.int32)
    idx = 0
    for f in _progress(range(t), desc):
        for _ in range(per_field):
            r = rng.integers(0, h - size + 1)
            cc = rng.integers(0, w - size + 1)
            out[idx] = fields[f, :, r:r + size, cc:cc + size]
            origins[idx] = (r, cc)
            idx += 1
    return out, origins


def crop_patches_to_disk(raw_path: Path, size: int, per_field: int,
                         rng: np.random.Generator, out_path: Path,
                         origins_path: Path, shuffle: bool, desc: str):
    """Stream random crops from the raw fields into an on-disk .npy memmap.

    With shuffle=True each crop lands at a pre-permuted position (decorrelates
    consecutive crops from the same field without ever loading the patch array
    into RAM). Returns (shape, per-channel mean, per-channel std) — the stats
    are float64 running sums over every written patch.
    """
    fields = np.load(raw_path, mmap_mode="r")
    if fields.ndim == 3:
        fields = fields[:, None]  # legacy single-channel raw -> (T, 1, H, W)
    t, c, h, w = fields.shape
    assert h >= size and w >= size, f"field {h}x{w} smaller than patch {size}"
    n = t * per_field
    perm = rng.permutation(n) if shuffle else np.arange(n)

    tmp = out_path.with_suffix(".npy.tmp")
    out = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(n, c, size, size))
    origins = np.empty((n, 2), dtype=np.int32)
    ch_sum = np.zeros(c, dtype=np.float64)
    ch_sumsq = np.zeros(c, dtype=np.float64)
    idx = 0
    for f in _progress(range(t), desc):
        field = np.asarray(fields[f], dtype=np.float32)  # one (C, H, W) field in RAM
        for _ in range(per_field):
            r = rng.integers(0, h - size + 1)
            cc = rng.integers(0, w - size + 1)
            patch = field[:, r:r + size, cc:cc + size]
            j = perm[idx]
            out[j] = patch
            origins[j] = (r, cc)
            p64 = patch.astype(np.float64)
            ch_sum += p64.sum(axis=(1, 2))
            ch_sumsq += (p64 * p64).sum(axis=(1, 2))
            idx += 1
    out.flush()
    del out, fields
    tmp.replace(out_path)
    _save_npy_atomic(origins_path, origins)

    n_pix = float(n) * size * size
    mean = ch_sum / n_pix
    std = np.sqrt(np.maximum(ch_sumsq / n_pix - mean ** 2, 0.0))
    return (n, c, size, size), mean.astype(np.float32), std.astype(np.float32)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    """Write via a .tmp then rename, so an interrupted save can't leave a
    half-written .npy that a later run would mistake for complete."""
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:  # file object: np.save won't append .npy
        np.save(fh, arr)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description="Crop patches + per-channel norm stats.")
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
        rng = np.random.default_rng(seed)
        shape, mean, std = crop_patches_to_disk(
            raw_dir / "train.npy", size, per_field, rng,
            train_out, train_origins_out, shuffle=True, desc="train crops")
        np.savez(stats_out, mean=mean, std=std, size=size)
        with np.printoptions(precision=3, suppress=True):
            print(f"train patches {shape} | z-score per-channel\n"
                  f"  mean={mean}\n  std={std}")

    # ── Test patches (skip if already built) ──────────────────────────────
    if not args.force and test_out.exists() and test_origins_out.exists():
        print(f"[skip] {test_out.name} already exists")
    else:
        rng = np.random.default_rng(seed + 1)
        shape, _, _ = crop_patches_to_disk(
            raw_dir / "test.npy", size, per_field, rng,
            test_out, test_origins_out, shuffle=False, desc="test crops")
        print(f"test patches {shape}")

    # Copy global lat/lon (and channel labels when present) so the dataset can
    # recover per-pixel coordinates for geo-conditioning (origins index into
    # these arrays).
    coords = np.load(raw_dir / "coords.npz")
    np.savez(patch_dir / "coords_full.npz", **{k: coords[k] for k in coords.files})

    print(f"-> {patch_dir}")


if __name__ == "__main__":
    main()
