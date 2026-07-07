"""Train the DDPM noise predictor on high-fidelity Z500 patches only.

The diffusion model never sees low-fidelity data during training — this is what
gives the inference-time distribution robustness across downsampling ratios.

Run:
    python -m train.train_diffusion --config config/default.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from models.diffusion import build_diffusion  # noqa: E402
from models.unet import build_unet  # noqa: E402
from train.ema import EMA  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from paths.ckpt_dir/diffusion.pt if it exists.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    tc = cfg["train"]
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    normalizer = load_norm_stats(patch_dir)
    ds = PatchDataset(patch_dir / "train_patches.npy", normalizer)
    loader = DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
    )
    print(f"Train patches: {len(ds)} | batches/epoch: {len(loader)}")

    # Held-out patches for a fixed-RNG validation loss (comparable across epochs).
    val_loader = None
    test_path = patch_dir / "test_patches.npy"
    if test_path.exists():
        val_ds = PatchDataset(test_path, normalizer)
        n_val = min(int(tc.get("val_patches", 256)), len(val_ds))
        val_loader = DataLoader(Subset(val_ds, range(n_val)),
                                batch_size=tc["batch_size"], shuffle=False, num_workers=0)
        print(f"Val patches: {n_val}")
    else:
        print("(no test_patches.npy — skipping val loss)")

    model = build_unet(cfg, use_time=True).to(device)
    diffusion = build_diffusion(cfg).to(device)
    ema = EMA(model, decay=tc["ema_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, step = 1, 0
    ckpt_path = ckpt_dir / "diffusion.pt"
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:  # checkpoints from before resume support lack these
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
        else:
            print("(old checkpoint: no optimizer/scaler state — resuming weights only)")
        start_epoch = ck["epoch"] + 1
        step = ck["step"]
        print(f"Resumed from {ckpt_path} at epoch {ck['epoch']} (step {step})")
    elif args.resume:
        print(f"(no checkpoint at {ckpt_path} — starting fresh)")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cfg["paths"]["log_dir"])
    except Exception:
        print("(tensorboard unavailable — skipping logging)")

    wb_run, wandb = init_wandb(cfg, job_type="train_diffusion",
                               extra_config={"unet_params": n_params, "n_train_patches": len(ds)})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    # Accumulators persist across epoch boundaries: batches/epoch is rarely
    # a multiple of log_every, and resetting per epoch both drops the tail
    # batches and makes the next log divide a partial sum by the full window.
    running, running_n, grad_sum = 0.0, 0, 0.0
    bucket_sum, bucket_n = [0.0] * 4, [0] * 4  # loss by timestep quartile
    t_last_log = time.time()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        epoch_loss, epoch_batches = 0.0, 0
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for x0 in loader:
            x0 = x0.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss, per_sample, t = diffusion.training_loss(model, x0, return_details=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                tc["grad_clip"] if tc["grad_clip"] > 0 else float("inf"))
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running += loss.item()
            running_n += 1
            grad_sum += grad_norm.item()
            epoch_loss += loss.item()
            epoch_batches += 1
            q = ((t - 1) * 4 // diffusion.timesteps).clamp(max=3)
            for k in range(4):
                sel = q == k
                if sel.any():
                    bucket_sum[k] += per_sample[sel].sum().item()
                    bucket_n[k] += int(sel.sum())
            step += 1
            if step % tc["log_every"] == 0:
                now = time.time()
                metrics = {
                    "train/loss": running / running_n,
                    "train/grad_norm": grad_sum / running_n,
                    "train/imgs_per_sec": running_n * x0.shape[0] / (now - t_last_log),
                    "epoch": epoch,
                }
                for k in range(4):  # q1 = lowest-noise quartile of t
                    if bucket_n[k]:
                        metrics[f"train/loss_t_q{k + 1}"] = bucket_sum[k] / bucket_n[k]
                if use_amp:
                    metrics["train/amp_scale"] = scaler.get_scale()
                print(f"epoch {epoch:03d} step {step:07d} | "
                      f"loss {metrics['train/loss']:.5f} | "
                      f"grad {metrics['train/grad_norm']:.3f} | "
                      f"{metrics['train/imgs_per_sec']:.1f} img/s")
                if writer:
                    for k_, v_ in metrics.items():
                        writer.add_scalar(k_, v_, step)
                if wb_run is not None:
                    wb_run.log(metrics, step=step)
                running, running_n, grad_sum = 0.0, 0, 0.0
                bucket_sum, bucket_n = [0.0] * 4, [0] * 4
                t_last_log = now

        epoch_metrics = {
            "train/epoch_loss": epoch_loss / max(epoch_batches, 1),
            "train/epoch_time_s": time.time() - epoch_start,
            "epoch": epoch,
        }
        if device.type == "cuda":
            epoch_metrics["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if val_loader is not None:
            epoch_metrics["val/loss"] = _val_loss(diffusion, model, val_loader, device)
            epoch_metrics["val/loss_ema"] = _val_loss(diffusion, ema.shadow, val_loader, device)
            print(f"epoch {epoch:03d} done | val loss {epoch_metrics['val/loss']:.5f} "
                  f"(ema {epoch_metrics['val/loss_ema']:.5f}) | "
                  f"{epoch_metrics['train/epoch_time_s']:.0f}s")
        if writer:
            for k_, v_ in epoch_metrics.items():
                writer.add_scalar(k_, v_, step)
        if wb_run is not None:
            wb_run.log(epoch_metrics, step=step)
        t_last_log = time.time()  # exclude val/sampling time from throughput

        if epoch % tc["sample_every_epochs"] == 0:
            sample_path = results_dir / f"uncond_epoch{epoch:03d}.png"
            _save_samples(diffusion, ema.shadow, normalizer, device, sample_path, cfg)
            if wb_run is not None:
                wb_run.log({"samples": wandb.Image(str(sample_path))}, step=step)
        if epoch % tc["ckpt_every_epochs"] == 0 or epoch == tc["epochs"]:
            _save_ckpt(ckpt_path, model, ema, opt, scaler, cfg, normalizer, epoch, step)

    if wb_run is not None:
        wb_run.finish()
    print(f"Done. Checkpoint -> {ckpt_dir / 'diffusion.pt'}")


@torch.no_grad()
def _val_loss(diffusion, model, val_loader, device):
    """Noise-prediction loss on held-out patches under a fixed RNG, so every
    epoch scores the same (timestep, noise) draws and values are comparable."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        for x0 in val_loader:
            x0 = x0.to(device, non_blocking=True)
            loss = diffusion.training_loss(model, x0)
            total += loss.item() * x0.shape[0]
            n += x0.shape[0]
    if was_training:
        model.train()
    return total / max(n, 1)


def _save_ckpt(path, model, ema, opt, scaler, cfg, normalizer, epoch, step):
    # Write via a .tmp then rename: this path overwrites the previous checkpoint
    # every ckpt_every_epochs, and an interrupted torch.save would otherwise
    # corrupt the only copy.
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg,
        "norm_mean": normalizer.mean,
        "norm_std": normalizer.std,
        "epoch": epoch,
        "step": step,
    }, tmp)
    tmp.replace(path)


@torch.no_grad()
def _save_samples(diffusion, model, normalizer, device, path, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = cfg["patches"]["size"]
    samples = diffusion.sample_unconditional(model, (4, 1, size, size), device, n_steps=100)
    samples = normalizer.decode(samples.cpu()).numpy()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, s in zip(axes, samples):
        ax.imshow(s[0], cmap="RdBu_r")
        ax.axis("off")
    fig.suptitle("Unconditional diffusion samples (Z500)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved samples -> {path}")


if __name__ == "__main__":
    main()
