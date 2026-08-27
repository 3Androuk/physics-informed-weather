"""Train the sphere-native residual diffusion model on the HEALPix mesh.

The DLWP-HPX backbone becomes the noise predictor of a DDPM whose target is the
residual between the true field and a frozen deterministic mean prediction
(models/hpx_residual.py). One sample is the whole globe, so unlike the sibling
patch pipeline there is no tiling anywhere — in training or in sampling.

Validation is the DDPM noise loss evaluated with FIXED timesteps and noise
(seeded per batch), so the curve is comparable across epochs instead of being
dominated by the random draw.

Run:
    # Phase B: reuse the trained deterministic SR model as the mean
    python -m train.train_diffusion --config config/diffusion.yaml \
        --mean learned --mean-ckpt checkpoints/t2m_hpx256/best.pt

    # Phase A: bilinear mean, no extra checkpoint needed
    python -m train.train_diffusion --config config/diffusion.yaml --mean bilinear
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import HPXDataset, load_norm_stats  # noqa: E402
from data.degrade import degrade_faces  # noqa: E402
from models.hpx_diffusion import build_diffusion  # noqa: E402
from models.hpx_residual import build_residual_model, load_mean_field  # noqa: E402
from models.hpx_unet import count_params  # noqa: E402
from train.ema import EMA  # noqa: E402
from utils import (ensure_dir, get_device, init_wandb, load_config,  # noqa: E402
                   resolve_amp, set_seed)


def _atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.no_grad()
def estimate_residual_scale(mean_field, loader, ratio, device, n_batches=16):
    """RMS of the residual y - mean(x), used to rescale x0 to ~unit variance.

    The DDPM schedule assumes x0 has roughly unit variance. A good deterministic
    mean leaves a SMALL residual (measured here: ~0.022 in normalized units for
    the t2m HPX256 regressor), and feeding that to an unscaled schedule leaves
    over 99% of timesteps at signal-to-noise << 1 — a regime where the
    loss-minimising prediction is the trivial eps = x_t / sqrt(1 - abar_t),
    computable from the input alone. Dividing the residual by this scalar
    restores a usable SNR range across the schedule; sampling multiplies it
    back (models/hpx_diffusion.py sample(residual_scale=...)).
    """
    s = torch.zeros((), device=device, dtype=torch.float64)
    n = 0
    for i, y in enumerate(loader):
        if i >= n_batches:
            break
        y = y.to(device, non_blocking=True)
        r = y - mean_field(degrade_faces(y, ratio))
        s += (r.double() ** 2).sum()
        n += r.numel()
    return float((s / max(n, 1)).sqrt())


@torch.no_grad()
def validate(model, diffusion, mean_field, loader, ratio, device,
             res_scale=1.0, seed=1234):
    """Mean DDPM noise loss with fixed t/noise per batch (comparable across epochs)."""
    model.eval()
    total = n = 0.0
    for bi, y in enumerate(loader):
        y = y.to(device, non_blocking=True)
        mean = mean_field(degrade_faces(y, ratio))
        x0 = (y - mean) / res_scale
        g = torch.Generator(device="cpu").manual_seed(seed + bi)
        t = torch.randint(1, diffusion.timesteps + 1, (y.shape[0],),
                          generator=g).to(device)
        noise = torch.randn(x0.shape, generator=g).to(device)
        x_t = diffusion.q_sample(x0, t, noise)
        pred = model(x_t, t.float(), (mean,))
        total += torch.nn.functional.mse_loss(pred, noise).item() * y.shape[0]
        n += y.shape[0]
    model.train()
    return total / max(n, 1)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/diffusion.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--mean", choices=["learned", "bilinear"], default=None,
                    help="deterministic mean field (default: config residual.mean)")
    ap.add_argument("--mean-ckpt", default=None,
                    help="frozen SR checkpoint for --mean learned")
    ap.add_argument("--resume", nargs="?", const="last.pt", default=None,
                    help="resume from <ckpt_dir>/last.pt or an explicit path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    set_seed(cfg["seed"])
    device = get_device()

    tc = cfg["train"]
    rc = cfg.get("residual", {})
    mean_kind = args.mean or rc.get("mean", "learned")
    mean_ckpt = args.mean_ckpt or rc.get("mean_ckpt")
    ratio = int(cfg["sr"]["ratio"])
    nside = int(cfg["hpx"]["nside"])
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

    mean_field = load_mean_field(mean_kind, ratio, nside, mean_ckpt, device)
    if mean_kind == "learned":
        mck = torch.load(mean_ckpt, map_location="cpu", weights_only=False)
        if abs(mck["norm_std"] - normalizer.std) > 1e-6 or \
           abs(mck["norm_mean"] - normalizer.mean) > 1e-6:
            raise ValueError(
                "mean checkpoint was trained with different normalization "
                f"(ckpt {mck['norm_mean']:.4f}/{mck['norm_std']:.4f} vs data "
                f"{normalizer.mean:.4f}/{normalizer.std:.4f}) — wrong dataset?")
        print(f"frozen mean: {mean_ckpt} (epoch {mck['epoch']}, "
              f"val rmse {mck['val_rmse_norm']:.5f} norm)")
    else:
        print("frozen mean: seam-aware bilinear upsampling")

    units = cfg["data"].get("units", "phys")
    res_scale = rc.get("scale")
    if res_scale is None:
        res_scale = estimate_residual_scale(mean_field, loader, ratio, device)
    res_scale = float(res_scale)
    if not res_scale > 0:
        raise ValueError(f"residual scale must be positive, got {res_scale}")
    print(f"residual scale: {res_scale:.5f} normalized "
          f"({res_scale * normalizer.std:.4f} {units}) — the chain models "
          f"(y - mean)/{res_scale:.5f}, so x0 has ~unit variance")

    diffusion = build_diffusion(cfg).to(device)
    model = build_residual_model(cfg).to(device)
    ema = EMA(model, decay=tc.get("ema_decay", 0.999))
    print(f"HPX residual diffusion | nside {nside} | ratio {ratio}x | "
          f"T {diffusion.timesteps} {cfg['diffusion']['beta_schedule']} | "
          f"train {len(train_ds)} val {n_val} | params {count_params(model):,} | "
          f"batch {tc['batch_size']}x{accum} (effective {tc['batch_size'] * accum})")

    wb_run, _ = init_wandb(cfg, job_type="train_diffusion", extra_config={
        "n_train": len(train_ds), "n_val": n_val, "mean_kind": mean_kind,
        "n_params": count_params(model)})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                            weight_decay=tc["weight_decay"])
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
             if tc.get("cosine_lr") else None)
    use_amp, amp_dtype = resolve_amp(tc, device)
    # bf16 keeps fp32's exponent range, so no loss scaling is required
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
        # the target definition must not change mid-run
        ck_scale = float(ck.get("residual_scale", 1.0))
        if abs(ck_scale - res_scale) > 1e-6 * max(1.0, ck_scale):
            print(f"  note: using the checkpoint's residual scale {ck_scale:.5f} "
                  f"(this run estimated {res_scale:.5f})")
            res_scale = ck_scale
        print(f"resumed {rpath}: starting epoch {start_epoch}, best {best_val:.5f}")

    running, running_n = 0.0, 0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        for y in loader:
            y = y.to(device, non_blocking=True)
            lf_up = degrade_faces(y, ratio)
            mean = mean_field(lf_up)
            x0 = (y - mean) / res_scale        # ~unit-variance residual target
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = diffusion.training_loss(model, x0, cond=(mean,))
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

        val_loss = validate(ema.shadow, diffusion, mean_field, val_loader,
                            ratio, device, res_scale)
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
            "mean_kind": mean_kind, "mean_ckpt": str(mean_ckpt or ""),
            "residual_scale": res_scale,
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
