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
    """High-fidelity patches, returned z-score normalized as (1, H, W) tensors."""

    def __init__(self, patch_path: str | Path, normalizer: Normalizer):
        self.patches = np.load(patch_path, mmap_mode="r")
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, i: int) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(self.patches[i])).float()
        return self.normalizer.encode(x)
