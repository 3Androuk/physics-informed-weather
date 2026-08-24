"""Train the DLWP-HPX U-Net for t2m super-resolution.

Direct mapping f: degraded t2m -> high-res t2m on the HEALPix mesh, MSE loss
in normalized space. Because HEALPix pixels are equal-area, the unweighted
pixel MSE is already an area-fair global loss — no latitude weighting needed
(one of the practical advantages of the mesh highlighted by the paper).

The validation split is the time-ordered tail of the training years, so no
temporal leakage into validation.

Run:
    python -m train.train_sr --config config/default.yaml
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import HPXDataset, load_norm_stats  # noqa: E402
from data.degrade import degrade_faces  # noqa: E402
from models.hpx_unet import build_model, count_params  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.no_grad()
def validate(model, loader, ratio, device):
    model.eval()
    se = n = 0.0
    for y in loader:
        y = y.to(device, non_blocking=True)
        x = degrade_faces(y, ratio)
        pred = model(x)
        se += torch.nn.functional.mse_loss(pred, y, reduction="sum").item()
        n += y.numel()
    model.train()
    return math.sqrt(se / n)  # RMSE in normalized units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    tc = cfg["train"]
    ratio = int(cfg["sr"]["ratio"])
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])

    normalizer = load_norm_stats(hpx_dir)
    full = HPXDataset(hpx_dir / "train.npy", normalizer)
    n_val = max(1, int(len(full) * tc["val_fraction"]))
    train_ds = Subset(full, range(len(full) - n_val))
    val_ds = Subset(full, range(len(full) - n_val, len(full)))

    loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tc["batch_size"], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = build_model(cfg).to(device)
    print(f"DLWP-HPX SR | nside {cfg['hpx']['nside']} | ratio {ratio}x | "
          f"train {len(train_ds)} val {n_val} | params {count_params(model):,}")

    wb_run, _ = init_wandb(cfg, job_type="train_sr", extra_config={
        "n_train": len(train_ds), "n_val": n_val,
        "n_params": count_params(model)})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
             if tc.get("cosine_lr") else None)
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    step = 0
    # Loss accumulator persists across epoch boundaries (see sibling project):
    # batches/epoch is rarely a multiple of log_every.
    running, running_n = 0.0, 0
    t0 = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        for y in loader:  # y: normalized HR faces (B, 12, 1, F, F)
            y = y.to(device, non_blocking=True)
            x = degrade_faces(y, ratio)  # LR input on the HR grid
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; aborting "
                                   "instead of training garbage")
            scaler.scale(loss).backward()
            if tc["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            running_n += 1
            step += 1
            if step % tc["log_every"] == 0:
                avg = running / running_n
                print(f"epoch {epoch:03d} step {step:07d} | mse {avg:.5f} | "
                      f"{time.time() - t0:.0f}s")
                running, running_n = 0.0, 0
                if wb_run is not None:
                    wb_run.log({"train/mse": avg, "epoch": epoch,
                                "lr": opt.param_groups[0]["lr"]}, step=step)
        if sched is not None:
            sched.step()

        val_rmse = validate(model, val_loader, ratio, device)
        units = cfg["data"].get("units", "phys")
        print(f"epoch {epoch:03d} | val rmse (norm) {val_rmse:.5f} "
              f"({val_rmse * normalizer.std:.3f} {units})")
        if wb_run is not None:
            wb_run.log({"val/rmse_norm": val_rmse,
                        "val/rmse_phys": val_rmse * normalizer.std,
                        "epoch": epoch}, step=step)

        for p in model.parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"non-finite weights after epoch {epoch}; "
                                   "aborting instead of saving garbage")

        state = {
            "model": model.state_dict(), "config": cfg, "epoch": epoch,
            "val_rmse_norm": val_rmse,
            "norm_mean": normalizer.mean, "norm_std": normalizer.std,
        }
        if epoch % tc["ckpt_every_epochs"] == 0:
            _atomic_save(state, ckpt_dir / "last.pt")
        if val_rmse < best_val:
            best_val = val_rmse
            _atomic_save(state, ckpt_dir / "best.pt")

    if wb_run is not None:
        wb_run.finish()
    print(f"Done. best val rmse (norm) {best_val:.5f} | checkpoints -> {ckpt_dir}")


if __name__ == "__main__":
    main()
