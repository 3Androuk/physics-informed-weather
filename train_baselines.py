"""
Baseline comparison: FNO (neuralop) vs AFNO (physicsnemo) on ERA5 Z500 + T850.

Uses WeatherBench 2 ERA5 data at 1.5deg resolution (120x240 grid after crop),
downloaded automatically from Google Cloud Storage (public, no auth required).
Trains on 2 channels (geopotential@500hPa, temperature@850hPa) and evaluates
with latitude-weighted RMSE and ACC per variable at multiple lead times.

Usage:
    python train_baselines.py [--epochs 40] [--lead-hours 24 72 120]
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

from neuralop.data.datasets.tensor_dataset import TensorDataset as NeuralOpTensorDataset
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.losses import LpLoss
from neuralop.models import FNO
from neuralop.training import Trainer
from physicsnemo.models.afno import AFNO

from data import (
    build_pairs, get_lat_weights, load_climatology, load_era5_data,
    VARIABLES, VAR_NAMES,
)
from evaluate import evaluate_normalized_mse, evaluate_wb2


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FNO vs AFNO baseline comparison.")
    # Data
    p.add_argument("--train-years", type=int, nargs=2, default=[1979, 2015])
    p.add_argument("--val-years", type=int, nargs=2, default=[2016, 2017])
    p.add_argument("--cache-dir", type=str, default="cache")
    # Lead times
    p.add_argument("--train-lead-hours", type=int, default=24)
    p.add_argument("--lead-hours", type=int, nargs="+", default=[24, 72, 120])
    # Training
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--models", nargs="+", choices=["fno", "afno"], default=["fno", "afno"])
    # FNO
    p.add_argument("--fno-modes", type=int, default=12)
    p.add_argument("--fno-width", type=int, default=32)
    p.add_argument("--fno-layers", type=int, default=4)
    # AFNO
    p.add_argument("--afno-patch-size", type=int, default=8)
    p.add_argument("--afno-embed-dim", type=int, default=256)
    p.add_argument("--afno-depth", type=int, default=4)
    p.add_argument("--afno-num-blocks", type=int, default=8)
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── FNO Training ──────────────────────────────────────────────────────────────

def train_fno(args, device, x_train, y_train, x_val, y_val):
    """Train FNO using neuralop's Trainer."""
    print("\n" + "=" * 60)
    print("  TRAINING FNO (neuralop)")
    print("=" * 60)

    # Identity normalizers — data already normalized; shape must match (1, C, 1, 1)
    n_vars = x_train.shape[1]
    in_norm = UnitGaussianNormalizer(
        mean=torch.zeros(1, n_vars, 1, 1), std=torch.ones(1, n_vars, 1, 1)
    )
    out_norm = UnitGaussianNormalizer(
        mean=torch.zeros(1, n_vars, 1, 1), std=torch.ones(1, n_vars, 1, 1)
    )

    train_ds = NeuralOpTensorDataset(x_train, y_train)
    val_ds = NeuralOpTensorDataset(x_val, y_val)

    lkw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        lkw["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **lkw)
    val_loader = DataLoader(val_ds, shuffle=False, **lkw)
    data_processor = DefaultDataProcessor(in_normalizer=in_norm, out_normalizer=out_norm)

    model = FNO(
        n_modes=(args.fno_modes, args.fno_modes),
        in_channels=n_vars,
        out_channels=n_vars,
        hidden_channels=args.fno_width,
        n_layers=args.fno_layers,
    ).to(device)

    n_params = count_params(model)
    print(f"FNO params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = Trainer(
        model=model,
        n_epochs=args.epochs,
        device=device,
        data_processor=data_processor,
        # AMP disabled for FNO: cuFFT requires power-of-2 dims in half precision
        mixed_precision=False,
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


# ── AFNO Training ─────────────────────────────────────────────────────────────

def train_afno(args, device, x_train, y_train, x_val, y_val):
    """Train AFNO with a standard PyTorch loop."""
    print("\n" + "=" * 60)
    print("  TRAINING AFNO (physicsnemo)")
    print("=" * 60)

    lkw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    if args.num_workers > 0:
        lkw["persistent_workers"] = True

    train_loader = DataLoader(TorchTensorDataset(x_train, y_train), shuffle=True, **lkw)
    val_loader = DataLoader(TorchTensorDataset(x_val, y_val), shuffle=False, **lkw)

    n_vars = x_train.shape[1]
    model = AFNO(
        inp_shape=[120, 240],
        in_channels=n_vars,
        out_channels=n_vars,
        patch_size=[args.afno_patch_size, args.afno_patch_size],
        embed_dim=args.afno_embed_dim,
        depth=args.afno_depth,
        num_blocks=args.afno_num_blocks,
    ).to(device)

    n_params = count_params(model)
    print(f"AFNO params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()

        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * xb.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    val_loss_sum += loss_fn(model(xb), yb).item() * xb.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)

        ep_time = time.time() - ep_start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.6f} | val={val_loss:.6f} | "
            f"lr={lr:.2e} | {ep_time:.1f}s"
        )
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "val_loss": round(val_loss, 8),
            "lr": lr,
            "time_s": round(ep_time, 2),
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

    return model, n_params, elapsed, history


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: train={args.train_years}, val={args.val_years}, "
        f"train_lead={args.train_lead_hours}h, eval_leads={args.lead_hours}h, "
        f"epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}"
    )

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Load data ─────────────────────────────────────────────────────────
    data_train_raw, data_val_raw, lat, lon = load_era5_data(
        train_years=tuple(args.train_years),
        val_years=tuple(args.val_years),
        cache_dir=args.cache_dir,
    )
    lat_weights = get_lat_weights(lat)

    # Build training pairs and normalize
    x_np, y_np = build_pairs(data_train_raw, args.train_lead_hours)
    xv_np, yv_np = build_pairs(data_val_raw, args.train_lead_hours)

    x_train = torch.from_numpy(np.ascontiguousarray(x_np))
    y_train = torch.from_numpy(np.ascontiguousarray(y_np))
    x_val = torch.from_numpy(np.ascontiguousarray(xv_np))
    y_val = torch.from_numpy(np.ascontiguousarray(yv_np))

    # Free training numpy arrays (val raw kept for WB2 multi-lead evaluation)
    del data_train_raw, x_np, y_np, xv_np, yv_np

    mean = x_train.mean(dim=(0, 2, 3), keepdim=True)
    std = x_train.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)

    x_train_n = (x_train - mean) / std
    y_train_n = (y_train - mean) / std
    x_val_n = (x_val - mean) / std
    y_val_n = (y_val - mean) / std

    print(f"Data: {len(x_train)} train, {len(x_val)} val, shape={tuple(x_train.shape[1:])}")

    models = args.models

    # ── Train ─────────────────────────────────────────────────────────────
    fno_model = fno_params = fno_time = fno_mse = None
    afno_model = afno_params = afno_time = afno_mse = None
    afno_history = []

    if "fno" in models:
        fno_model, fno_params, fno_time = train_fno(
            args, device, x_train_n, y_train_n, x_val_n, y_val_n
        )
    if "afno" in models:
        afno_model, afno_params, afno_time, afno_history = train_afno(
            args, device, x_train_n, y_train_n, x_val_n, y_val_n
        )

    # ── Normalized MSE ────────────────────────────────────────────────────
    print("\nEvaluating normalized MSE...")
    if fno_model is not None:
        fno_mse = evaluate_normalized_mse(fno_model, x_val_n, y_val_n, device)
    if afno_model is not None:
        afno_mse = evaluate_normalized_mse(afno_model, x_val_n, y_val_n, device)

    # ── Save checkpoints ──────────────────────────────────────────────────
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, mdl, mse in [("fno", fno_model, fno_mse), ("afno", afno_model, afno_mse)]:
        if mdl is None:
            continue
        cpu_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in mdl.state_dict().items()}
        torch.save(
            {"model_state_dict": cpu_state, "config": vars(args),
             "best_val_mse": mse, "normalization_mean": mean, "normalization_std": std},
            ckpt_dir / f"{name}_baseline.pt",
        )

    # ── WB2 evaluation ────────────────────────────────────────────────────
    clim_ds = load_climatology()
    wb2_eval = {}
    active = [(n, m) for n, m in [("fno", fno_model), ("afno", afno_model)] if m is not None]
    for lead_h in args.lead_hours:
        print(f"\n--- Evaluating at {lead_h}h lead time ---")
        for name, mdl in active:
            m = evaluate_wb2(
                model=mdl, data_val_raw=data_val_raw, lead_hours=lead_h,
                mean=mean, std=std, lat_weights=lat_weights,
                device=device, val_years=tuple(args.val_years), clim_ds=clim_ds,
                variables=VARIABLES, var_names=VAR_NAMES,
            )
            wb2_eval[f"{name}_{lead_h}h"] = m
            parts = "  |  ".join(
                f"{vn.upper()}: RMSE={m[f'rmse_{vn}']:>8.2f}  ACC={m[f'acc_{vn}']:.4f}"
                for vn in VAR_NAMES
            )
            print(f"  {name.upper():5s} | {parts}")

    # ── Print tables ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)
    print(f"{'Model':<8} | {'Val MSE':>12} | {'Params':>10} | {'Time/ep':>10} | {'Total':>10}")
    print("-" * 70)
    fno_epochs = args.epochs
    afno_epochs = len(afno_history) or args.epochs
    for name, mse, params, total, ep in [("FNO", fno_mse, fno_params, fno_time, fno_epochs),
                                          ("AFNO", afno_mse, afno_params, afno_time, afno_epochs)]:
        if mse is None:
            continue
        print(f"{name:<8} | {mse:>12.6f} | {params:>10,} | {total / ep:>8.1f}s | {total:>8.1f}s")

    var_cols = "  |  ".join(f"{'RMSE_' + vn.upper():>12}  {'ACC_' + vn.upper():>10}" for vn in VAR_NAMES)
    sep_width = 24 + 16 * len(VAR_NAMES)
    print("\n" + "=" * sep_width)
    print("  WB2 EVALUATION")
    print("=" * sep_width)
    print(f"{'Lead':>6} | {'Model':<6} | {var_cols}")
    print("-" * sep_width)
    for lead_h in args.lead_hours:
        for name in ["fno", "afno"]:
            if f"{name}_{lead_h}h" not in wb2_eval:
                continue
            m = wb2_eval[f"{name}_{lead_h}h"]
            var_vals = "  |  ".join(
                f"{m[f'rmse_{vn}']:>12.2f}  {m[f'acc_{vn}']:>10.4f}" for vn in VAR_NAMES
            )
            print(f"{lead_h:>5}h | {name.upper():<6} | {var_vals}")
    print("=" * sep_width)

    # ── Save results ──────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    final = {}
    if fno_mse is not None:
        fno_epochs = args.epochs  # neuralop Trainer always runs all epochs
        final["fno"] = {"val_mse": fno_mse, "params": fno_params,
                        "time_per_epoch_s": round(fno_time / fno_epochs, 2), "total_time_s": round(fno_time, 2)}
    if afno_mse is not None:
        afno_epochs = len(afno_history) or args.epochs
        final["afno"] = {"val_mse": afno_mse, "params": afno_params,
                         "time_per_epoch_s": round(afno_time / afno_epochs, 2), "total_time_s": round(afno_time, 2)}
    results = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "afno_history": afno_history,
        "final": final,
        "wb2_evaluation": wb2_eval,
    }
    path = "results/baseline_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
