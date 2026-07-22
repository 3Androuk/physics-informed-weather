"""Direct-mapping UNet baseline: f: X -> Y, trained on 4x pairs ONLY.

This baseline is deliberately locked to one input distribution (4x). The
robustness experiment then applies it to 8x (out-of-distribution), where it is
expected to degrade — reproducing the paper's central comparison. Do NOT train
it on multiple ratios.

Run:
    python -m train.train_directmap --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import degrade  # noqa: E402
from models.unet import build_unet  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from paths.ckpt_dir/directmap.pt if it exists.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    dc = cfg["directmap"]
    ratio = dc["train_ratio"]
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])

    normalizer = load_norm_stats(patch_dir)
    ds = PatchDataset(patch_dir / "train_patches.npy", normalizer)
    loader = DataLoader(
        ds, batch_size=dc["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=cfg["train"]["num_workers"] > 0,
    )
    print(f"Direct-map baseline | train ratio {ratio}x | patches {len(ds)}")

    # Held-out patches for per-epoch val MSE at the training ratio AND at 8x:
    # the in-distribution/out-of-distribution gap is the brittleness story as
    # a live training curve.
    val_x = None
    test_path = patch_dir / "test_patches.npy"
    if test_path.exists():
        val_ds = PatchDataset(test_path, normalizer)
        n_val = min(int(cfg["train"].get("val_patches", 256)), len(val_ds))
        val_x = torch.stack([val_ds[i] for i in range(n_val)])
        print(f"Val patches: {n_val}")

    wb_run, _ = init_wandb(cfg, job_type="train_directmap",
                           extra_config={"train_ratio": ratio, "n_train_patches": len(ds)})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    model = build_unet(cfg, use_time=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=dc["lr"])
    use_amp = dc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    start_epoch, step = 1, 0
    ckpt_path = ckpt_dir / "directmap.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            start_epoch = ck["epoch"] + 1
            step = ck["step"]
            print(f"Resumed from {ckpt_path} at epoch {ck['epoch']} (step {step})")
        else:
            print("(old checkpoint: weights only — resuming from epoch 1 counters)")
    elif args.resume:
        print(f"(no checkpoint at {ckpt_path} — starting fresh)")
    # Loss accumulator persists across epoch boundaries: batches/epoch is rarely
    # a multiple of log_every, and resetting per epoch both drops the tail
    # batches and makes the next log divide a partial sum by the full window.
    running, running_n = 0.0, 0
    for epoch in range(start_epoch, dc["epochs"] + 1):
        model.train()
        for y in loader:  # y: normalized HF target
            y = y.to(device, non_blocking=True)
            # Low-fidelity input: degrade in normalized space (equivalent up to
            # the affine z-score), produced ONLY at the training ratio.
            x = degrade(y, ratio)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            if dc["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), dc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            running_n += 1
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                avg = running / running_n
                print(f"epoch {epoch:03d} step {step:07d} | mse {avg:.5f}")
                running, running_n = 0.0, 0
                if wb_run is not None:
                    wb_run.log({"train/mse": avg, "epoch": epoch}, step=step)

        # ── Per-epoch val MSE at train ratio and at 8x (OOD gap) ──────────
        if val_x is not None:
            model.eval()
            metrics = {"epoch": epoch}
            with torch.no_grad():
                for r, key in ((ratio, f"val/mse_{ratio}x"), (8, "val/mse_8x")):
                    total, n = 0.0, 0
                    for i in range(0, len(val_x), dc["batch_size"]):
                        y = val_x[i:i + dc["batch_size"]].to(device, non_blocking=True)
                        total += loss_fn(model(degrade(y, r)), y).item() * y.shape[0]
                        n += y.shape[0]
                    metrics[key] = total / max(n, 1)
            print(f"epoch {epoch:03d} done | val {ratio}x {metrics[f'val/mse_{ratio}x']:.5f} "
                  f"| val 8x {metrics['val/mse_8x']:.5f}")
            if wb_run is not None:
                wb_run.log(metrics, step=step)

        # ── Atomic per-epoch checkpoint (a crash loses at most one epoch) ─
        tmp = ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "scaler": scaler.state_dict(), "config": cfg, "train_ratio": ratio,
            "norm_mean": normalizer.mean, "norm_std": normalizer.std,
            "epoch": epoch, "step": step,
        }, tmp)
        tmp.replace(ckpt_path)

    if wb_run is not None:
        wb_run.finish()
    print(f"Done. Checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
