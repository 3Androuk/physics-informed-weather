"""Patch dataset + normalization helpers.

The diffusion model trains on z-score-normalized high-fidelity patches only.
Low-fidelity inputs are generated on the fly (see data.degrade) and are never
part of the diffusion training set.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Normalizer:
    """Z-score normalization with stored scalar mean/std (in physical units)."""

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-8 else 1.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    @classmethod
    def from_npz(cls, path: str | Path) -> "Normalizer":
        d = np.load(path)
        return cls(float(d["mean"]), float(d["std"]))


def load_norm_stats(patch_dir: str | Path) -> Normalizer:
    return Normalizer.from_npz(Path(patch_dir) / "norm_stats.npz")


class PatchDataset(Dataset):
    """High-fidelity patches, returned z-score normalized as (1, H, W) tensors.

    If `origins_path` and `coords_full_path` are given, each item is instead a
    tuple (patch, coords) where coords is the per-pixel normalized geographic
    coordinate grid (H, W, d) for that crop — used for geo-conditioning.
    """

    def __init__(self, patch_path: str | Path, normalizer: Normalizer,
                 origins_path=None, coords_full_path=None,
                 geo_input_dim: int = 3, altitude=None):
        self.patches = np.load(patch_path, mmap_mode="r")
        self.normalizer = normalizer
        self.geo = origins_path is not None and coords_full_path is not None
        if self.geo:
            from models.geo_encoding import build_patch_coords  # local import
            self._build_coords = build_patch_coords
            self.origins = np.load(origins_path)
            cf = np.load(coords_full_path)
            self.lat_full, self.lon_full = cf["lat"], cf["lon"]
            self.geo_input_dim = geo_input_dim
            self.altitude = altitude

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int):
        # Copy out of the read-only mmap: from_numpy on a non-writable view is
        # undefined behavior if the tensor is ever written to.
        x = torch.from_numpy(np.array(self.patches[i], dtype=np.float32))
        x = self.normalizer.encode(x)
        if not self.geo:
            return x
        s = x.shape[-1]
        r, c = int(self.origins[i, 0]), int(self.origins[i, 1])
        alt = self.altitude if self.geo_input_dim == 4 else None
        coords = self._build_coords(
            self.lat_full[r:r + s], self.lon_full[c:c + s], altitude=alt
        )
        return x, torch.from_numpy(coords)
