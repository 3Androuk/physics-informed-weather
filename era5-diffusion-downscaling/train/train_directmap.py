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

    wb_run, _ = init_wandb(cfg, job_type="train_directmap",
                           extra_config={"train_ratio": ratio, "n_train_patches": len(ds)})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    model = build_unet(cfg, use_time=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=dc["lr"])
    use_amp = dc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    step = 0
    for epoch in range(1, dc["epochs"] + 1):
        model.train()
        running = 0.0
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
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                avg = running / cfg["train"]["log_every"]
                print(f"epoch {epoch:03d} step {step:07d} | mse {avg:.5f}")
                running = 0.0
                if wb_run is not None:
                    wb_run.log({"train/mse": avg, "epoch": epoch}, step=step)

    torch.save({
        "model": model.state_dict(), "config": cfg, "train_ratio": ratio,
        "norm_mean": normalizer.mean, "norm_std": normalizer.std,
    }, ckpt_dir / "directmap.pt")
    if wb_run is not None:
        wb_run.finish()
    print(f"Done. Checkpoint -> {ckpt_dir / 'directmap.pt'}")


if __name__ == "__main__":
    main()
