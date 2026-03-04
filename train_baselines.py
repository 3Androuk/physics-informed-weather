"""
Baseline comparison: FNO (neuralop) vs AFNO (modulus) on ERA5 z500.

Usage:
    python train_baselines.py [--data-dir PATH] [--years 2015 2016 2017] [--epochs 40]

Trains both models on the same data split, then prints a comparison table.
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset as TorchTensorDataset

from era5_baseline_dataset import build_pairs, load_era5_z500

# neuralop imports for FNO
from neuralop.data.datasets.tensor_dataset import TensorDataset as NeuralOpTensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses import LpLoss
from neuralop.models import FNO
from neuralop.training import Trainer

# modulus import for AFNO
from modulus.models.afno import AFNO


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir / "era5-5.625deg" / "geopotential_500"

    p = argparse.ArgumentParser(description="FNO vs AFNO baseline comparison.")
    p.add_argument("--data-dir", type=str, default=str(default_data_dir))
    p.add_argument("--years", type=int, nargs="+", default=[2015, 2016, 2017])
    p.add_argument("--lead-steps", type=int, default=24)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--no-amp", action="store_true")
    # FNO hyperparams
    p.add_argument("--fno-modes", type=int, default=12)
    p.add_argument("--fno-width", type=int, default=32)
    p.add_argument("--fno-layers", type=int, default=4)
    # AFNO hyperparams
    p.add_argument("--afno-patch-size", type=int, default=8)
    p.add_argument("--afno-embed-dim", type=int, default=256)
    p.add_argument("--afno-depth", type=int, default=4)
    p.add_argument("--afno-num-blocks", type=int, default=8)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Shared data loading

def load_and_split(args):
    """Load ERA5 z500, create pairs, split, and normalize.

    Returns normalized tensors and normalization stats.
    Both models are trained on the same normalized data for a fair comparison.
    """
    z, _ = load_era5_z500(args.data_dir, args.years)
    x_np, y_np = build_pairs(z, args.lead_steps)

    n_train = int(len(x_np) * args.train_frac)
    x_train = torch.from_numpy(np.ascontiguousarray(x_np[:n_train]))
    y_train = torch.from_numpy(np.ascontiguousarray(y_np[:n_train]))
    x_val = torch.from_numpy(np.ascontiguousarray(x_np[n_train:]))
    y_val = torch.from_numpy(np.ascontiguousarray(y_np[n_train:]))

    # z-score normalization (from training data only)
    mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    std = x_train.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
    x_train = (x_train - mean) / std
    y_train = (y_train - mean) / std
    x_val = (x_val - mean) / std
    y_val = (y_val - mean) / std

    print(f"Data: {len(x_train)} train, {len(x_val)} val, "
          f"shape={tuple(x_train.shape[1:])}")

    return x_train, y_train, x_val, y_val, mean, std


# Common evaluation

@torch.no_grad()
def evaluate_mse(model, x_val, y_val, device, batch_size=256):
    """Compute per-element MSE on normalized validation data. Works for any nn.Module."""
    model.eval()
    ds = TorchTensorDataset(x_val, y_val)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_se = 0.0
    total_elements = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        total_se += (pred - yb).pow(2).sum().item()
        total_elements += yb.numel()
    return total_se / total_elements


# FNO training via neuralop's Trainer
def train_fno(args, device, x_train, y_train, x_val, y_val) -> tuple:
    """Train FNO using neuralop's native Trainer + DataProcessor pipeline.

    The data is already normalized, so we set up neuralop's normalizers
    as identity (mean=0, std=1) so the Trainer pipeline doesn't double-normalize.
    """
    print("\n" + "=" * 60)
    print("  TRAINING FNO (neuralop)")
    print("=" * 60)

    # Identity normalizers as data is already normalised
    in_norm = UnitGaussianNormalizer(
        mean=torch.zeros(1, 1, 1, 1), std=torch.ones(1, 1, 1, 1)
    )
    out_norm = UnitGaussianNormalizer(
        mean=torch.zeros(1, 1, 1, 1), std=torch.ones(1, 1, 1, 1)
    )

    train_ds = NeuralOpTensorDataset(x_train, y_train)
    val_ds = NeuralOpTensorDataset(x_val, y_val)

    lkw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        lkw["persistent_workers"] = True
    train_loader = DataLoader(train_ds, shuffle=True, **lkw)
    val_loader = DataLoader(val_ds, shuffle=False, **lkw)

    data_processor = DefaultDataProcessor(
        in_normalizer=in_norm, out_normalizer=out_norm
    )

    model = FNO(
        n_modes=(args.fno_modes, args.fno_modes),
        in_channels=1,
        out_channels=1,
        hidden_channels=args.fno_width,
        n_layers=args.fno_layers,
    ).to(device)

    n_params = count_params(model)
    print(f"FNO params: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        data_processor=data_processor,
        mixed_precision=not args.no_amp,
        verbose=True,
    )

    t0 = time.time()
    trainer.train(
        train_loader=train_loader,
        test_loaders={"val": val_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=LpLoss(d=2, p=2),
        eval_losses={"l2": LpLoss(d=2, p=2)},
        save_dir="checkpoints/fno_baseline",
        save_best="val_l2",
    )
    elapsed = time.time() - t0

    return model, n_params, elapsed


# AFNO training via PyTorch loop
def train_afno(args, device, x_train, y_train, x_val, y_val) -> tuple:
    """Train AFNO using standard PyTorch training loop."""
    print("\n" + "=" * 60)
    print("  TRAINING AFNO (modulus)")
    print("=" * 60)

    lkw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        lkw["persistent_workers"] = True
    train_loader = DataLoader(
        TorchTensorDataset(x_train, y_train), shuffle=True, **lkw
    )
    val_loader = DataLoader(
        TorchTensorDataset(x_val, y_val), shuffle=False, **lkw
    )

    model = AFNO(
        inp_shape=[32, 64],
        in_channels=1,
        out_channels=1,
        patch_size=[args.afno_patch_size, args.afno_patch_size],
        embed_dim=args.afno_embed_dim,
        depth=args.afno_depth,
        num_blocks=args.afno_num_blocks,
    ).to(device)

    n_params = count_params(model)
    print(f"AFNO params: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    loss_fn = nn.MSELoss()

    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    afno_history = []

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        # validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)

        elapsed_epoch = time.time() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"lr={lr:.2e} | {elapsed_epoch:.1f}s"
        )
        afno_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss": round(val_loss, 8),
            "lr": lr,
            "time_s": round(elapsed_epoch, 2),
        })

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch}. Best val={best_val:.6f}")
            break

    elapsed = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, n_params, elapsed, afno_history


# Results persistence

def save_results(args, afno_history, fno_metrics, afno_metrics):
    os.makedirs("results", exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "afno_history": afno_history,
        "final": {
            "fno": fno_metrics,
            "afno": afno_metrics,
        },
    }
    path = "results/baseline_results.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# Main

def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Config: years={args.years}, lead={args.lead_steps}h, "
          f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load data once
    # both models train on the same normalized split
    x_train, y_train, x_val, y_val, mean, std = load_and_split(args)

    # Train both models
    fno_model, fno_params, fno_time = train_fno(
        args, device, x_train, y_train, x_val, y_val
    )
    afno_model, afno_params, afno_time, afno_history = train_afno(
        args, device, x_train, y_train, x_val, y_val
    )

    # Evaluate both on the same data with the same metric
    print("\nEvaluating both models (MSE on normalized val data)...")
    fno_mse = evaluate_mse(fno_model, x_val, y_val, device)
    afno_mse = evaluate_mse(afno_model, x_val, y_val, device)

    # Save checkpoints
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, mdl, mse in [("fno", fno_model, fno_mse), ("afno", afno_model, afno_mse)]:
        state = mdl.state_dict()
        # Move all tensors to CPU for saving
        cpu_state = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state.items()
        }
        torch.save(
            {
                "model_state_dict": cpu_state,
                "config": vars(args),
                "best_val_mse": mse,
                "normalization_mean": mean,
                "normalization_std": std,
            },
            ckpt_dir / f"{name}_baseline.pt",
        )

    # comparison table
    print("\n" + "=" * 70)
    print("  BASELINE COMPARISON (MSE on normalized val data)")
    print("=" * 70)
    print(f"{'Model':<8} | {'Val MSE':>12} | {'Params':>10} | {'Time/epoch':>12} | {'Total':>10}")
    print("-" * 70)
    for name, mse, params, total in [
        ("FNO", fno_mse, fno_params, fno_time),
        ("AFNO", afno_mse, afno_params, afno_time),
    ]:
        tpe = total / args.epochs
        print(f"{name:<8} | {mse:>12.6f} | {params:>10,} | {tpe:>10.1f}s | {total:>8.1f}s")
    print("=" * 70)

    save_results(
        args,
        afno_history,
        fno_metrics={
            "val_mse": fno_mse,
            "params": fno_params,
            "time_per_epoch_s": round(fno_time / args.epochs, 2),
            "total_time_s": round(fno_time, 2),
        },
        afno_metrics={
            "val_mse": afno_mse,
            "params": afno_params,
            "time_per_epoch_s": round(afno_time / args.epochs, 2),
            "total_time_s": round(afno_time, 2),
        },
    )


if __name__ == "__main__":
    main()
