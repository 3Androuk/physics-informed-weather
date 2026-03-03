"""
ERA5 baseline dataset utilities for FNO (neuralop) and AFNO (modulus) baselines.

Provides two data preparation paths:
  - create_fno_data():  neuralop-native (UnitGaussianNormalizer + TensorDataset + DefaultDataProcessor)
  - create_afno_data(): plain PyTorch (manual z-score + torch TensorDataset + DataLoader)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset as TorchTensorDataset

from neuralop.data.datasets.tensor_dataset import TensorDataset as NeuralOpTensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer


# Shared data loading
def load_era5_z500(data_dir: str, years: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load z500 NetCDF files for the specified years, sort by time, deduplicate.

    Args:
        data_dir: Path to era5-5.625deg/geopotential_500/ directory.
        years: List of years to load (e.g. [2015, 2016, 2017]).

    Returns:
        z: float32 array of shape (T, 32, 64) — geopotential values.
        times: datetime64 array of shape (T,).
    """
    data_dir = Path(data_dir)
    z_chunks, t_chunks = [], []

    for year in sorted(years):
        pattern = f"geopotential_500hPa_{year}_5.625deg.nc"
        matches = list(data_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No file found for year {year} in {data_dir}")
        for fpath in matches:
            with xr.open_dataset(fpath) as ds:
                da = ds["z"].transpose("time", "lat", "lon")
                z_chunks.append(np.asarray(da.values, dtype=np.float32))
                t_chunks.append(np.asarray(da["time"].values))

    z = np.concatenate(z_chunks, axis=0)
    times = np.concatenate(t_chunks, axis=0)

    # Sort by time and deduplicate
    order = np.argsort(times)
    z, times = z[order], times[order]
    _, unique_idx = np.unique(times, return_index=True)
    z, times = z[np.sort(unique_idx)], times[np.sort(unique_idx)]

    if not np.isfinite(z).all():
        raise ValueError("Data contains NaN or Inf values.")

    print(f"Loaded z500: {z.shape[0]} timesteps, grid {z.shape[1]}x{z.shape[2]}, "
          f"years {min(years)}-{max(years)}")
    return z, times


def build_pairs(z: np.ndarray, lead_steps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Create supervised (input, target) pairs with the given lead time.

    Args:
        z: Array of shape (T, H, W).
        lead_steps: Number of timesteps ahead for prediction target.

    Returns:
        x: float32 array (N, 1, H, W) — inputs.
        y: float32 array (N, 1, H, W) — targets.
    """
    x = z[:-lead_steps, None, :, :]  # (N, 1, H, W)
    y = z[lead_steps:, None, :, :]
    return x, y


def _split(x: np.ndarray, y: np.ndarray, train_frac: float = 0.8):
    """Chronological train/val split."""
    n_train = int(len(x) * train_frac)
    x_train = torch.from_numpy(np.ascontiguousarray(x[:n_train]))
    y_train = torch.from_numpy(np.ascontiguousarray(y[:n_train]))
    x_val = torch.from_numpy(np.ascontiguousarray(x[n_train:]))
    y_val = torch.from_numpy(np.ascontiguousarray(y[n_train:]))
    return x_train, y_train, x_val, y_val


# neuralop data path for FNO
def create_fno_data(
    data_dir: str,
    years: List[int],
    lead_steps: int = 24,
    batch_size: int = 64,
    train_frac: float = 0.8,
    num_workers: int = 2,
) -> Tuple[DataLoader, Dict[str, DataLoader], DefaultDataProcessor]:
    """Prepare data for neuralop FNO using library-native utilities.

    Returns:
        train_loader: DataLoader yielding {'x': ..., 'y': ...} dicts.
        test_loaders: Dict with 'val' key.
        data_processor: DefaultDataProcessor with fitted normalizers.
    """
    z, _ = load_era5_z500(data_dir, years)
    x, y = build_pairs(z, lead_steps)
    x_train, y_train, x_val, y_val = _split(x, y, train_frac)

    # neuralop normalizers 
    # fit on training data
    in_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    in_normalizer.fit(x_train)
    out_normalizer = UnitGaussianNormalizer(dim=[0, 2, 3])
    out_normalizer.fit(y_train)

    # neuralop TensorDataset returns {'x': ..., 'y': ...} dicts
    train_ds = NeuralOpTensorDataset(x_train, y_train)
    val_ds = NeuralOpTensorDataset(x_val, y_val)

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    data_processor = DefaultDataProcessor(
        in_normalizer=in_normalizer,
        out_normalizer=out_normalizer,
    )

    print(f"FNO data: train={len(train_ds)}, val={len(val_ds)}, "
          f"batch_size={batch_size}")
    return train_loader, {"val": val_loader}, data_processor


# PyTorch data path for AFNO

def create_afno_data(
    data_dir: str,
    years: List[int],
    lead_steps: int = 24,
    batch_size: int = 64,
    train_frac: float = 0.8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Prepare data for Modulus AFNO using manual normalization.

    Returns:
        train_loader: DataLoader yielding (x, y) tuples.
        val_loader: DataLoader yielding (x, y) tuples.
        mean: Normalization mean tensor (1, 1, 1, 1).
        std: Normalization std tensor (1, 1, 1, 1).
    """
    z, _ = load_era5_z500(data_dir, years)
    x, y = build_pairs(z, lead_steps)
    x_train, y_train, x_val, y_val = _split(x, y, train_frac)

    # Manual z-score normalization
    mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    std = x_train.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    y_train = (y_train - mean) / std
    x_val = (x_val - mean) / std
    y_val = (y_val - mean) / std

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        TorchTensorDataset(x_train, y_train), shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        TorchTensorDataset(x_val, y_val), shuffle=False, **loader_kwargs
    )

    print(f"AFNO data: train={len(x_train)}, val={len(x_val)}, "
          f"batch_size={batch_size}")
    return train_loader, val_loader, mean, std
