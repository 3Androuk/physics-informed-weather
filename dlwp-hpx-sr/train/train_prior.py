"""Train an UNCONDITIONAL diffusion prior over HEALPix fields.

This is the sphere-native counterpart of the sibling project's guided-diffusion
model (Shu et al. 2023): the DDPM is trained on high-resolution fields ONLY,
with no low-resolution input and no degradation ratio anywhere in training.
Super-resolution happens entirely at inference, via
HPXGaussianDiffusion.guided_reconstruct — noise-mixing at a per-ratio start
level, DDIM down to zero, and the exact mesh data-consistency projection at
every step.

Why this rather than the conditional/residual model in train_diffusion.py: a
model that learns p(y | x) at particular ratios cannot extrapolate to ratios it
never saw. Measured on this project's own data at 8x (both trained at 4x), the
patch direct map scores 1.23 and the mesh regressor 1.26, against bicubic's
0.96 — while patch guided diffusion holds 0.77. Ratio-agnostic training is the
property that survives distribution shift, and it is the reason this arm exists.

Note there is no residual rescaling here: the z-scored field already has ~unit
variance, which is what the beta schedule assumes. (The residual trainer needs
that rescaling precisely because a good mean leaves a tiny residual.)

Run:
    python -m train.train_prior --config config/prior.yaml --wandb
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import HPXDataset, load_norm_stats  # noqa: E402
from models.hpx_diffusion import build_diffusion  # noqa: E402
from models.hpx_unet import build_model, count_params  # noqa: E402
from train.ema import EMA  # noqa: E402
from utils import (ensure_dir, get_device, init_wandb, load_config,  # noqa: E402
                   resolve_amp, set_seed)


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def build_prior_model(cfg: dict):
    """Unconditional noise predictor: no conditioning channels, no input skip."""
    model = build_model(cfg, use_time=True, extra_in_channels=0)
    model.global_residual = False   # the output is epsilon, not a correction
    return model


@torch.no_grad()
def validate(model, diffusion, loader, device, seed=1234):
    """DDPM noise loss with FIXED timesteps and noise, so epochs are comparable."""
    model.eval()
    total = n = 0.0
    for bi, y in enumerate(loader):
        y = y.to(device, non_blocking=True)
        g = torch.Generator(device="cpu").manual_seed(seed + bi)
        t = torch.randint(1, diffusion.timesteps + 1, (y.shape[0],),
                          generator=g).to(device)
        noise = torch.randn(y.shape, generator=g).to(device)
        pred = model(diffusion.q_sample(y, t, noise), t.float())
        total += torch.nn.functional.mse_loss(pred, noise).item() * y.shape[0]
        n += y.shape[0]
    model.train()
    return total / max(n, 1)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/prior.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", nargs="?", const="last.pt", default=None,
                    help="resume from <ckpt_dir>/last.pt or an explicit path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    tc = cfg["train"]
    hpx_dir = Path(cfg["paths"]["hpx_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
    accum = max(1, int(tc.get("accum_steps", 1)))

    normalizer = load_norm_stats(hpx_dir)
    full = HPXDataset(hpx_dir / "train.npy", normalizer)
    n_val = max(1, int(len(full) * tc["val_fraction"]))
    train_ds = Subset(full, range(len(full) - n_val))
    val_ds = Subset(full, range(len(full) - n_val, len(full)))
    loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True,
                        num_workers=tc["num_workers"], pin_memory=True,
                        drop_last=True, persistent_workers=tc["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False,
                            num_workers=0, pin_memory=True)

    diffusion = build_diffusion(cfg).to(device)
    model = build_prior_model(cfg).to(device)
    ema = EMA(model, decay=tc.get("ema_decay", 0.999))
    print(f"HPX unconditional prior | nside {cfg['hpx']['nside']} | "
          f"T {diffusion.timesteps} {cfg['diffusion']['beta_schedule']} | "
          f"train {len(train_ds)} val {n_val} | params {count_params(model):,} | "
          f"batch {tc['batch_size']}x{accum} (effective {tc['batch_size'] * accum}) | "
          f"NO ratio seen in training")

    wb_run, _ = init_wandb(cfg, job_type="train_prior", extra_config={
        "n_train": len(train_ds), "n_val": n_val,
        "n_params": count_params(model)})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
             if tc.get("cosine_lr") else None)
    use_amp, amp_dtype = resolve_amp(tc, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    best_val = float("inf")
    step = 0
    start_epoch = 1
    if args.resume:
        rpath = Path(args.resume)
        if not rpath.exists():
            rpath = Path(ckpt_dir) / rpath
        ck = torch.load(rpath, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        if ck.get("ema") is not None:
            ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        if sched is not None and ck.get("sched") is not None:
            sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        step = int(ck.get("step", 0))
        best_val = float(ck.get("best_val", ck.get("val_loss", float("inf"))))
        print(f"resumed {rpath}: starting epoch {start_epoch}, best {best_val:.5f}")

    running, running_n = 0.0, 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        for y in loader:                     # y: normalized HR faces, ~unit variance
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = diffusion.training_loss(model, y, cond=None)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}; aborting "
                                   "instead of training garbage")
            scaler.scale(loss / accum).backward()
            step += 1
            if step % accum == 0:
                if tc["grad_clip"] > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
            running += loss.item()
            running_n += 1
            if step % tc["log_every"] == 0:
                avg = running / running_n
                print(f"epoch {epoch:03d} step {step:07d} | eps mse {avg:.5f} | "
                      f"{time.time() - t0:.0f}s")
                running, running_n = 0.0, 0
                if wb_run is not None:
                    wb_run.log({"train/eps_mse": avg, "epoch": epoch,
                                "lr": opt.param_groups[0]["lr"]}, step=step)
        if sched is not None:
            sched.step()

        val_loss = validate(ema.shadow, diffusion, val_loader, device)
        print(f"epoch {epoch:03d} | val eps mse (EMA) {val_loss:.5f}")
        if wb_run is not None:
            wb_run.log({"val/eps_mse": val_loss, "epoch": epoch}, step=step)

        for p in model.parameters():
            if not torch.isfinite(p).all():
                raise RuntimeError(f"non-finite weights after epoch {epoch}; "
                                   "aborting instead of saving garbage")

        state = {
            "model": model.state_dict(), "ema": ema.state_dict(),
            "config": cfg, "epoch": epoch, "val_loss": val_loss,
            "norm_mean": normalizer.mean, "norm_std": normalizer.std,
            "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "scaler": scaler.state_dict(), "step": step,
            "best_val": min(best_val, val_loss),
        }
        if epoch % tc["ckpt_every_epochs"] == 0:
            _atomic_save(state, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            _atomic_save(state, ckpt_dir / "best.pt")

    if wb_run is not None:
        wb_run.finish()
    print(f"Done. best val eps mse {best_val:.5f} | checkpoints -> {ckpt_dir}")


if __name__ == "__main__":
    main()
