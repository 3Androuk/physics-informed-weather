"""HEALPix-face dataset + normalization helpers.

Samples are single-time t2m fields on the 12 HEALPix faces, z-score
normalized with scalar train-split statistics (stored in physical units, K).
Low-fidelity inputs are generated on the fly (data.degrade) in the training
loop, so only the high-res faces are stored on disk.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Normalizer:
    """Z-score normalization with stored scalar mean/std (physical units)."""

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-8 else 1.0

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean

    @classmethod
    def from_npz(cls, path: str | Path) -> "Normalizer":
        d = np.load(path)
        return cls(float(d["mean"]), float(d["std"]))


def load_norm_stats(hpx_dir: str | Path) -> Normalizer:
    return Normalizer.from_npz(Path(hpx_dir) / "norm_stats.npz")


class HPXDataset(Dataset):
    """High-res HEALPix faces, returned normalized as (12, 1, F, F) tensors."""

    def __init__(self, path: str | Path, normalizer: Normalizer):
        self.fields = np.load(path, mmap_mode="r")  # (T, 12, F, F)
        if self.fields.ndim != 4 or self.fields.shape[1] != 12:
            raise ValueError(f"expected (T, 12, F, F), got {self.fields.shape}")
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, i: int) -> torch.Tensor:
        # Copy out of the read-only mmap before wrapping in a tensor.
        x = torch.from_numpy(np.array(self.fields[i], dtype=np.float32))
        return self.normalizer.encode(x).unsqueeze(1)  # (12, 1, F, F)
