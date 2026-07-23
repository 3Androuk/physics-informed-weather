"""Train the conditional RESIDUAL diffusion model (split-model Phase A).

Deterministic mean = bicubic upsampling of the coarse field (no training);
the diffusion learns the residual HF - bicubic, conditioned on the bicubic
field (+ geo embedding with --geo). The degradation ratio is RANDOMIZED per
batch over residual.train_ratios, so one model serves all ratios — test at a
held-out ratio (e.g. 6) for the generalization claim.

Run:
    python -m train.train_residual --config config/t2m.yaml --wandb [--geo] [--seed N]
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import coarsen  # noqa: E402
from eval.metrics import spectrum_log_l1  # noqa: E402
from models.diffusion import build_diffusion  # noqa: E402
from models.residual import build_residual_model  # noqa: E402
from train.ema import EMA  # noqa: E402
from utils import ensure_dir, get_device, init_wandb, load_config, set_seed  # noqa: E402


def _mean_field(y: torch.Tensor, ratio: int) -> torch.Tensor:
    """Phase-A deterministic mean: bicubic upsample of the coarse field."""
    lo = coarsen(y, ratio)
    return F.interpolate(lo, size=y.shape[-2:], mode="bicubic", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the residual checkpoint if it exists.")
    ap.add_argument("--geo", action="store_true",
                    help="Condition the residual model on the hash-grid location "
                         "embedding as well.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override config seed; suffixes the checkpoint name.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    if args.geo:
        cfg.setdefault("geo", {})["enabled"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])
    device = get_device()

    tc = cfg["train"]
    rcfg = cfg.get("residual", {})
    ratios = rcfg.get("train_ratios", [2, 4, 8])
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    geo_on = cfg.get("geo", {}).get("enabled", False)
    seed_suffix = f"_s{cfg['seed']}" if args.seed is not None else ""
    hpx = geo_on and cfg["geo"].get("encoder", "hash") == "healpix"
    ckpt_name = f"residual{'_geo' if geo_on else ''}{'_hpx' if hpx else ''}{seed_suffix}.pt"

    normalizer = load_norm_stats(patch_dir)
    gkw = {}
    if geo_on:
        gcfg = cfg["geo"]
        gkw = dict(origins_path=patch_dir / "train_origins.npy",
                   coords_full_path=patch_dir / "coords_full.npz",
                   geo_input_dim=gcfg["input_dim"], altitude=gcfg["altitude"],
                   geo_encoder=gcfg.get("encoder", "hash"))
    ds = PatchDataset(patch_dir / "train_patches.npy", normalizer, **gkw)
    loader = DataLoader(
        ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=tc["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=tc["num_workers"] > 0,
    )
    print(f"Residual diffusion | ratios {ratios} | patches {len(ds)} | geo={geo_on}")

    # ── Residual normalization: one scalar std over ratios (estimated once).
    # The mean-field channel tells the model which regime it is in, so a
    # shared scale is sufficient; the value is stored in the checkpoint.
    with torch.no_grad():
        chunks = []
        for r in ratios:
            y = torch.stack([ds[i][0] if geo_on else ds[i] for i in range(64)])
            chunks.append((y - _mean_field(y, r)).flatten())
        res_std = float(torch.cat(chunks).std())
    print(f"Residual std (normalized units): {res_std:.4f}")

    # ── Val: fixed-RNG residual noise-prediction loss at the middle ratio.
    val_loader, val_ratio = None, ratios[len(ratios) // 2]
    test_path = patch_dir / "test_patches.npy"
    if test_path.exists():
        vkw = dict(gkw)
        if geo_on:
            vkw["origins_path"] = patch_dir / "test_origins.npy"
        val_ds = PatchDataset(test_path, normalizer, **vkw)
        n_val = min(int(tc.get("val_patches", 256)), len(val_ds))
        val_loader = DataLoader(Subset(val_ds, range(n_val)),
                                batch_size=tc["batch_size"], shuffle=False, num_workers=0)
        print(f"Val patches: {n_val} (ratio {val_ratio}x)")

    model = build_residual_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    ema = EMA(model, decay=tc["ema_decay"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    use_amp = tc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, step = 1, 0
    ckpt_path = ckpt_dir / ckpt_name
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        res_std = ck["res_std"]
        start_epoch = ck["epoch"] + 1
        step = ck["step"]
        print(f"Resumed from {ckpt_path} at epoch {ck['epoch']} (step {step})")
    elif args.resume:
        print(f"(no checkpoint at {ckpt_path} — starting fresh)")

    wb_run, wandb = init_wandb(cfg, job_type="train_residual",
                               extra_config={"unet_params": n_params,
                                             "n_train_patches": len(ds),
                                             "train_ratios": ratios,
                                             "res_std": res_std})
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")

    running, running_n, grad_sum = 0.0, 0, 0.0
    t_last_log = time.time()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        model.train()
        epoch_loss, epoch_batches = 0.0, 0
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:
            if geo_on:
                y, coords = batch
                y = y.to(device, non_blocking=True)
                coords = coords.to(device, non_blocking=True)
            else:
                y, coords = batch.to(device, non_blocking=True), None
            ratio = random.choice(ratios)
            mean_f = _mean_field(y, ratio)
            x0 = (y - mean_f) / res_std
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = diffusion.training_loss(model, x0, cond=(mean_f, coords))
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
            step += 1
            if step % tc["log_every"] == 0:
                now = time.time()
                metrics = {
                    "train/loss": running / running_n,
                    "train/grad_norm": grad_sum / running_n,
                    "train/imgs_per_sec": running_n * y.shape[0] / (now - t_last_log),
                    "epoch": epoch,
                }
                print(f"epoch {epoch:03d} step {step:07d} | "
                      f"loss {metrics['train/loss']:.5f} | "
                      f"grad {metrics['train/grad_norm']:.3f} | "
                      f"{metrics['train/imgs_per_sec']:.1f} img/s")
                if wb_run is not None:
                    wb_run.log(metrics, step=step)
                running, running_n, grad_sum = 0.0, 0, 0.0
                t_last_log = now

        epoch_metrics = {
            "train/epoch_loss": epoch_loss / max(epoch_batches, 1),
            "train/epoch_time_s": time.time() - epoch_start,
            "epoch": epoch,
        }
        if device.type == "cuda":
            epoch_metrics["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if val_loader is not None:
            epoch_metrics["val/loss"] = _val_loss(
                diffusion, ema.shadow, val_loader, device, val_ratio, res_std, geo_on)
            print(f"epoch {epoch:03d} done | val loss {epoch_metrics['val/loss']:.5f} | "
                  f"{epoch_metrics['train/epoch_time_s']:.0f}s")
        if wb_run is not None:
            wb_run.log(epoch_metrics, step=step)
        t_last_log = time.time()

        if epoch % tc["sample_every_epochs"] == 0 and val_loader is not None:
            fig_path = results_dir / f"residual_epoch{epoch:03d}.png"
            spec = _save_recons(diffusion, ema.shadow, val_loader.dataset, normalizer,
                                device, res_std, rcfg.get("n_steps", 100), geo_on,
                                fig_path)
            if wb_run is not None:
                log = {"recons": wandb.Image(str(fig_path))}
                if spec is not None:
                    log["samples/spectrum_log_l1"] = spec
                wb_run.log(log, step=step)

        if epoch % tc["ckpt_every_epochs"] == 0 or epoch == tc["epochs"]:
            if not (_weights_finite(model) and _weights_finite(ema.shadow)):
                raise RuntimeError(
                    f"non-finite weights at epoch {epoch} — training has diverged. "
                    f"Checkpoint NOT overwritten; last good state kept at {ckpt_path}.")
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({
                "model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "config": cfg, "res_std": res_std,
                "norm_mean": normalizer.mean, "norm_std": normalizer.std,
                "epoch": epoch, "step": step,
            }, tmp)
            tmp.replace(ckpt_path)

    if wb_run is not None:
        wb_run.finish()
    print(f"Done. Checkpoint -> {ckpt_path}")


@torch.no_grad()
def _weights_finite(model) -> bool:
    return all(p.isfinite().all() for p in model.parameters())


@torch.no_grad()
def _val_loss(diffusion, model, val_loader, device, ratio, res_std, geo_on):
    """Fixed-RNG residual noise-prediction loss at one ratio (comparable
    across epochs)."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        for batch in val_loader:
            if geo_on:
                y, coords = batch
                y = y.to(device, non_blocking=True)
                coords = coords.to(device, non_blocking=True)
            else:
                y, coords = batch.to(device, non_blocking=True), None
            mean_f = _mean_field(y, ratio)
            x0 = (y - mean_f) / res_std
            loss = diffusion.training_loss(model, x0, cond=(mean_f, coords))
            total += loss.item() * y.shape[0]
            n += y.shape[0]
    if was_training:
        model.train()
    return total / max(n, 1)


@torch.no_grad()
def _save_recons(diffusion, model, val_subset, normalizer, device, res_std,
                 n_steps, geo_on, path):
    """2 fixed val patches reconstructed at 4x and 8x: mean | mean+residual |
    target, shared color scale. Returns spectrum_log_l1 of the recons vs
    targets (both ratios pooled) or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [val_subset[i] for i in range(2)]
    if geo_on:
        y = torch.stack([it[0] for it in items]).to(device)
        coords = torch.stack([it[1] for it in items]).to(device)
    else:
        y = torch.stack(items).to(device)
        coords = None

    recs, panels = [], []
    for i in range(len(y)):
        yi = y[i:i + 1]
        ci = None if coords is None else coords[i:i + 1]
        row = []
        for r in (4, 8):
            mean_f = _mean_field(yi, r)
            res = diffusion.sample_unconditional(
                model, yi.shape, device, n_steps=n_steps, cond=(mean_f, ci))
            rec = mean_f + res_std * res
            recs.append(rec)
            row += [(f"Mean {r}x", mean_f), (f"Recon {r}x", rec)]
        row.append(("Target", yi))
        panels.append(row)

    fig, axes = plt.subplots(len(panels), 5, figsize=(20, 4 * len(panels)))
    axes = axes.reshape(len(panels), 5)
    for r_i, row in enumerate(panels):
        ref = normalizer.decode(row[-1][1].cpu())[0, 0].numpy()
        vmin, vmax = float(ref.min()), float(ref.max())
        for ax, (title, t) in zip(axes[r_i], row):
            ax.imshow(normalizer.decode(t.cpu())[0, 0].numpy(), cmap="RdBu_r",
                      vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
    fig.suptitle("Residual model reconstructions (fixed val patches)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved recons -> {path}")

    rec_phys = normalizer.decode(torch.cat(recs).cpu())
    tgt_phys = normalizer.decode(torch.cat([y, y]).cpu())
    return spectrum_log_l1(rec_phys, tgt_phys)


if __name__ == "__main__":
    main()
