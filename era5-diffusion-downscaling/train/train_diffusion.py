"""Train the DDPM noise predictor on high-fidelity Z500 patches only.

The diffusion model never sees low-fidelity data during training — this is what
gives the inference-time distribution robustness across downsampling ratios.

Run:
    python -m train.train_diffusion --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from models.diffusion import build_diffusion  # noqa: E402
from models.unet import build_unet  # noqa: E402
from train.ema import EMA  # noqa: E402
from utils import ensure_dir, get_device, load_config, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
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

    model = build_unet(cfg, use_time=True).to(device)
    diffusion = build_diffusion(cfg).to(device)
    ema = EMA(model, decay=tc["ema_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cfg["paths"]["log_dir"])
    except Exception:
        print("(tensorboard unavailable — skipping logging)")

    step = 0
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        running = 0.0
        for x0 in loader:
            x0 = x0.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(model, x0)
            scaler.scale(loss).backward()
            if tc["grad_clip"] > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running += loss.item()
            step += 1
            if step % tc["log_every"] == 0:
                avg = running / tc["log_every"]
                running = 0.0
                print(f"epoch {epoch:03d} step {step:07d} | loss {avg:.5f}")
                if writer:
                    writer.add_scalar("train/loss", avg, step)

        if epoch % tc["sample_every_epochs"] == 0:
            _save_samples(diffusion, ema.shadow, normalizer, device,
                          results_dir / f"uncond_epoch{epoch:03d}.png", cfg)
        if epoch % tc["ckpt_every_epochs"] == 0 or epoch == tc["epochs"]:
            _save_ckpt(ckpt_dir / "diffusion.pt", model, ema, cfg, normalizer, epoch, step)

    print(f"Done. Checkpoint -> {ckpt_dir / 'diffusion.pt'}")


def _save_ckpt(path, model, ema, cfg, normalizer, epoch, step):
    torch.save({
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "config": cfg,
        "norm_mean": normalizer.mean,
        "norm_std": normalizer.std,
        "epoch": epoch,
        "step": step,
    }, path)


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
