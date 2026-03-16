"""
WeatherBench 2-style evaluation metrics.

Provides per-variable latitude-weighted RMSE and Anomaly Correlation
Coefficient (ACC) for multi-variable forecasts, following WB2 conventions.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from data import HOURS_PER_STEP, VARIABLES, VAR_NAMES, build_pairs, denormalize


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Run model on all inputs and return predictions on CPU."""
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(device)
        preds.append(model(xb).cpu())
    return torch.cat(preds, dim=0)


def lat_weighted_rmse(
    pred: torch.Tensor,
    truth: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Latitude-weighted RMSE per channel, in physical units.

    Args:
        pred:  (N, C, H, W) predictions in physical units.
        truth: (N, C, H, W) targets in physical units.
        lat_weights: (H,) cosine-latitude weights, mean-normalized to 1.

    Returns:
        Tensor of shape (C,) with RMSE per variable.
    """
    se = (pred - truth).pow(2)                   # (N, C, H, W)
    w = lat_weights.view(1, 1, -1, 1)             # (1, 1, H, 1)
    # Mean over N, H, W (weighted); shape -> (C,)
    return (se * w).mean(dim=(0, 2, 3)).sqrt()


def anomaly_correlation(
    pred: torch.Tensor,
    truth: torch.Tensor,
    clim: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    """Anomaly Correlation Coefficient per channel.

    Args:
        pred:  (N, C, H, W) predictions in physical units.
        truth: (N, C, H, W) targets in physical units.
        clim:  (N, C, H, W) climatology values for each target timestep.
        lat_weights: (H,) weights.

    Returns:
        Tensor of shape (C,) with ACC per variable.
    """
    w = lat_weights.view(1, 1, -1, 1)

    pred_anom  = pred  - clim
    truth_anom = truth - clim

    # Sum over spatial dims per sample, then average over time
    num = (w * pred_anom * truth_anom).sum(dim=(2, 3))         # (N, C)
    dp  = (w * pred_anom.pow(2)).sum(dim=(2, 3)).sqrt()         # (N, C)
    dt  = (w * truth_anom.pow(2)).sum(dim=(2, 3)).sqrt()        # (N, C)

    acc_per_sample = num / (dp * dt + 1e-8)                     # (N, C)
    return acc_per_sample.mean(dim=0)                            # (C,)


def get_climatology_for_targets(
    lead_hours: int,
    years: tuple,
    clim_ds: xr.Dataset,
    variables: list = VARIABLES,
) -> torch.Tensor:
    """Extract climatology values matching target timesteps (vectorized).

    Returns:
        Tensor of shape (N, C, 120, 240).
    """
    lead_steps = lead_hours // HOURS_PER_STEP

    start = pd.Timestamp(f"{years[0]}-01-01")
    end   = pd.Timestamp(f"{years[1]}-12-31T18:00:00")
    all_times    = pd.date_range(start, end, freq="6h")
    target_times = all_times[lead_steps:]

    doys  = xr.DataArray(target_times.dayofyear.values, dims="sample")
    hours = xr.DataArray(target_times.hour.values,      dims="sample")

    channels = []
    for var_name, level in variables:
        da = clim_ds[var_name].sel(level=level) if level is not None else clim_ds[var_name]
        da = da.transpose("dayofyear", "hour", "latitude", "longitude")
        arr = da.sel(dayofyear=doys, hour=hours).values.astype(np.float32)  # (N, 121, 240)
        arr = arr[:, :120, :]                                                # crop lat
        channels.append(arr)

    clim_arr = np.stack(channels, axis=1)   # (N, C, 120, 240)
    return torch.from_numpy(clim_arr)


@torch.no_grad()
def evaluate_normalized_mse(
    model: torch.nn.Module,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Per-element MSE averaged over all channels and spatial dims."""
    model.eval()
    loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )
    total_se, total_elements = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total_se       += (pred - yb).pow(2).sum().item()
        total_elements += yb.numel()
    return total_se / total_elements


@torch.no_grad()
def evaluate_wb2(
    model: torch.nn.Module,
    data_val_raw: np.ndarray,
    lead_hours: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    lat_weights: torch.Tensor,
    device: torch.device,
    val_years: tuple,
    clim_ds: xr.Dataset,
    batch_size: int = 64,
    variables: list = VARIABLES,
    var_names: list = VAR_NAMES,
) -> dict:
    """Full WB2-style evaluation at a given lead time.

    Returns dict with normalized_mse and per-variable rmse + acc.
    """
    model.eval()

    x_np, y_np = build_pairs(data_val_raw, lead_hours)
    x_raw = torch.from_numpy(np.ascontiguousarray(x_np))
    y_raw = torch.from_numpy(np.ascontiguousarray(y_np))

    x_norm    = (x_raw - mean) / std
    pred_norm = predict_all(model, x_norm, device, batch_size)
    pred_phys = denormalize(pred_norm, mean, std)

    y_norm   = (y_raw - mean) / std
    norm_mse = (pred_norm - y_norm).pow(2).mean().item()

    rmse_per_var = lat_weighted_rmse(pred_phys, y_raw, lat_weights)
    clim = get_climatology_for_targets(lead_hours, val_years, clim_ds, variables)
    acc_per_var  = anomaly_correlation(pred_phys, y_raw, clim, lat_weights)

    result = {
        "lead_hours": lead_hours,
        "normalized_mse": round(norm_mse, 6),
    }
    for i, name in enumerate(var_names):
        result[f"rmse_{name}"] = round(rmse_per_var[i].item(), 4)
        result[f"acc_{name}"]  = round(acc_per_var[i].item(),  4)
    return result


VAR_UNITS = {"z500": "m²/s²", "t850": "K"}


@torch.no_grad()
def plot_forecast_maps(
    model: torch.nn.Module,
    data_val_raw: np.ndarray,
    lead_hours: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    lat: np.ndarray,
    lon: np.ndarray,
    device: torch.device,
    model_name: str,
    var_names: list = VAR_NAMES,
    sample_idx: int = 0,
    save_dir: str = "results",
):
    """Plot truth / prediction / error maps for a single sample.

    Saves a PNG with one row per variable and 3 columns (Truth, Pred, Error).
    """
    model.eval()
    x_np, y_np = build_pairs(data_val_raw, lead_hours)
    x_sample = torch.from_numpy(np.ascontiguousarray(x_np[sample_idx : sample_idx + 1]))
    y_sample = torch.from_numpy(np.ascontiguousarray(y_np[sample_idx : sample_idx + 1]))

    x_norm = (x_sample - mean) / std
    pred_norm = model(x_norm.to(device)).cpu()
    pred_phys = denormalize(pred_norm, mean, std)

    truth = y_sample[0].numpy()   # (C, H, W)
    pred  = pred_phys[0].numpy()  # (C, H, W)
    error = pred - truth           # (C, H, W)

    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, 3, figsize=(18, 5 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for row, vn in enumerate(var_names):
        unit = VAR_UNITS.get(vn, "")
        t = truth[row]
        p = pred[row]
        e = error[row]

        vmin = min(t.min(), p.min())
        vmax = max(t.max(), p.max())
        eabs = max(abs(e.min()), abs(e.max()))

        # Truth
        im0 = axes[row, 0].pcolormesh(lon, lat, t, vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
        axes[row, 0].set_title(f"Truth — {vn.upper()} [{unit}]")
        fig.colorbar(im0, ax=axes[row, 0], shrink=0.8)

        # Prediction
        im1 = axes[row, 1].pcolormesh(lon, lat, p, vmin=vmin, vmax=vmax, cmap="viridis", shading="auto")
        axes[row, 1].set_title(f"Prediction — {vn.upper()} [{unit}]")
        fig.colorbar(im1, ax=axes[row, 1], shrink=0.8)

        # Error
        im2 = axes[row, 2].pcolormesh(lon, lat, e, vmin=-eabs, vmax=eabs, cmap="RdBu_r", shading="auto")
        axes[row, 2].set_title(f"Error — {vn.upper()} [{unit}]")
        fig.colorbar(im2, ax=axes[row, 2], shrink=0.8)

        for ax in axes[row]:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

    fig.suptitle(f"{model_name.upper()} — {lead_hours}h forecast (sample {sample_idx})", fontsize=14, y=1.01)
    fig.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name}_{lead_hours}h_forecast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved forecast map: {path}")
