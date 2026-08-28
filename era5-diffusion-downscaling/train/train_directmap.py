"""Direct-mapping UNet baseline: f: X -> Y, trained on 4x pairs ONLY.

This baseline is deliberately locked to one input distribution (4x). The
robustness experiment then applies it to 8x (out-of-distribution), where it is
expected to degrade — reproducing the paper's central comparison. Do NOT train
it on multiple ratios.

Run:
    python -m train.train_directmap --config config/default.yaml

Optionally, the same run can be split across several GPUs/nodes (e.g. 4 nodes)
by launching under torchrun — see train/distributed.py and
scripts/train_multinode.sh. Single-process behavior is unchanged.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import degrade  # noqa: E402
from models.unet import build_unet  # noqa: E402
from train.distributed import (cleanup, init_distributed,  # noqa: E402
                               make_train_loader, set_epoch, wrap_model)
from utils import (display_channel, ensure_dir, geo_suffix,  # noqa: E402
                   init_wandb, load_config, run_name, set_seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true",
                    help="Enable wandb logging (overrides config wandb.enabled).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from paths.ckpt_dir/directmap.pt if it exists.")
    ap.add_argument("--geo", action="store_true",
                    help="Geo-conditioned regression: concat the hash-grid location "
                         "embedding to the input (CorrDiff-style static "
                         "conditioning); checkpoint becomes directmap_geo.pt.")
    ap.add_argument("--encoder", choices=["hash", "healpix", "xyz", "sinusoidal", "static", "hash_static", "xyz_static", "sinusoidal_static"], default=None,
                    help="Override geo.encoder from the CLI so the config can "
                         "keep its default.")
    ap.add_argument("--random-ratio", action="store_true",
                    help="Ratio-randomized REGRESSION MEAN for the split model "
                         "(samples residual.train_ratios per batch; checkpoint "
                         "becomes meanmap*.pt). The plain direct map stays "
                         "fixed-ratio by design — this mode is a different role, "
                         "not a replacement.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    if args.geo:
        cfg.setdefault("geo", {})["enabled"] = True
    if args.encoder is not None:
        cfg.setdefault("geo", {})["encoder"] = args.encoder
    set_seed(cfg["seed"])
    dist = init_distributed()  # no-op unless launched under torchrun
    device = dist.device

    dc = cfg["directmap"]
    ratio = dc["train_ratio"]
    train_ratios = (cfg.get("residual", {}).get("train_ratios", [2, 4, 8])
                    if args.random_ratio else None)
    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])

    normalizer = load_norm_stats(patch_dir)
    geo_on = cfg.get("geo", {}).get("enabled", False)
    gkw = {}
    if geo_on:
        gcfg = cfg["geo"]
        gkw = dict(origins_path=patch_dir / "train_origins.npy",
                   coords_full_path=patch_dir / "coords_full.npz",
                   geo_input_dim=gcfg["input_dim"], altitude=gcfg["altitude"],
                   geo_encoder=gcfg.get("encoder", "hash"),
                   healpix_index_path=((patch_dir / gcfg["healpix_index"])
                                       if gcfg.get("healpix_index") else None))
    ds = PatchDataset(patch_dir / "train_patches.npy", normalizer, **gkw)
    loader = make_train_loader(ds, dc["batch_size"], cfg["train"]["num_workers"],
                               dist, seed=cfg["seed"])
    mode = f"ratios {train_ratios} (mean mode)" if train_ratios else f"train ratio {ratio}x"
    print(f"{'Regression mean' if train_ratios else 'Direct-map baseline'} | {mode} "
          f"| patches {len(ds)} | geo={geo_on}")

    # Held-out patches for per-epoch val MSE at the training ratio AND at 8x:
    # the in-distribution/out-of-distribution gap is the brittleness story as
    # a live training curve.
    # Rank 0 only under DDP: weights are identical on every rank, so one
    # process scoring the full val set reproduces single-process values.
    val_x, val_coords = None, None
    test_path = patch_dir / "test_patches.npy"
    if dist.is_main and test_path.exists():
        vkw = dict(gkw)
        if geo_on:
            vkw["origins_path"] = patch_dir / "test_origins.npy"
        val_ds = PatchDataset(test_path, normalizer, **vkw)
        n_val = min(int(cfg["train"].get("val_patches", 256)), len(val_ds))
        if geo_on:
            items = [val_ds[i] for i in range(n_val)]
            val_x = torch.stack([it[0] for it in items])
            val_coords = torch.stack([it[1] for it in items])
        else:
            val_x = torch.stack([val_ds[i] for i in range(n_val)])
        print(f"Val patches: {n_val}")

    if geo_on:
        # No level gate: the direct map has no diffusion time input (t=None),
        # so noise-dependent gating is undefined for this model.
        from models.geo_encoding import GeoConditionedUNet, build_geo_encoder
        geo_enc = build_geo_encoder(cfg)
        base = build_unet(cfg, use_time=False, extra_in_channels=geo_enc.output_dim)
        model = GeoConditionedUNet(base, geo_enc).to(device)
    else:
        model = build_unet(cfg, use_time=False).to(device)
    print(f"UNet params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=dc["lr"])
    use_amp = dc["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    def fwd(x, c):
        return model(x, None, c) if geo_on else model(x)

    start_epoch, step = 1, 0
    stem = "meanmap" if args.random_ratio else "directmap"
    ckpt_path = ckpt_dir / f"{stem}{geo_suffix(cfg)}.pt"
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

    # DDP wrap AFTER resume so state loads into the raw module. fwd late-binds
    # `model`, so training goes through the wrapper (gradient averaging), while
    # fwd_eval keeps validation/figures on the raw module (rank 0 only — a DDP
    # forward there would deadlock waiting for the other ranks).
    raw_model = model
    model = wrap_model(model, dist, cfg)
    if dist.enabled:
        set_seed(cfg["seed"] + dist.rank)  # decorrelate per-batch ratio draws

    def fwd_eval(x, c):
        return raw_model(x, None, c) if geo_on else raw_model(x)

    wb_run, _ = (None, None)
    if dist.is_main:
        wb_run, _ = init_wandb(cfg, job_type="train_directmap",
                               extra_config={"train_ratio": train_ratios if train_ratios else ratio,
                                             "n_train_patches": len(ds),
                                             "world_size": dist.world_size},
                               name=run_name(cfg, ckpt_path.stem,
                                             "resumed" if start_epoch > 1 else ""))
    if wb_run is not None:
        print(f"wandb: logging to {wb_run.url}")
    # Loss accumulator persists across epoch boundaries: batches/epoch is rarely
    # a multiple of log_every, and resetting per epoch both drops the tail
    # batches and makes the next log divide a partial sum by the full window.
    running, running_n, grad_sum = 0.0, 0, 0.0
    t_last_log = time.time()
    for epoch in range(start_epoch, dc["epochs"] + 1):
        set_epoch(loader, epoch)  # reshuffle the per-rank shard (DDP only)
        model.train()
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for batch in loader:  # normalized HF target (+ coords when geo)
            if geo_on:
                y, coords = batch
                y = y.to(device, non_blocking=True)
                coords = coords.to(device, non_blocking=True)
            else:
                y, coords = batch.to(device, non_blocking=True), None
            # Low-fidelity input: degrade in normalized space (equivalent up to
            # the affine z-score). Fixed ratio for the baseline; randomized per
            # batch in --random-ratio (mean) mode.
            x = degrade(y, random.choice(train_ratios) if train_ratios else ratio)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(fwd(x, coords), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                dc["grad_clip"] if dc["grad_clip"] > 0 else float("inf"))
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            running_n += 1
            grad_sum += grad_norm.item()
            step += 1
            if step % cfg["train"]["log_every"] == 0:
                now = time.time()
                metrics = {
                    "train/mse": running / running_n,
                    "train/grad_norm": grad_sum / running_n,
                    # Global throughput: ranks step in lockstep, so rank 0's
                    # window x world_size counts every rank's images.
                    "train/imgs_per_sec": (running_n * y.shape[0] * dist.world_size
                                           / (now - t_last_log)),
                    "epoch": epoch,
                }
                print(f"epoch {epoch:03d} step {step:07d} | "
                      f"mse {metrics['train/mse']:.5f} | "
                      f"grad {metrics['train/grad_norm']:.3f} | "
                      f"{metrics['train/imgs_per_sec']:.1f} img/s")
                running, running_n, grad_sum = 0.0, 0, 0.0
                t_last_log = now
                if wb_run is not None:
                    wb_run.log(metrics, step=step)

        # ── Per-epoch val MSE at train ratio and at 8x (OOD gap) ──────────
        epoch_metrics = {"train/epoch_time_s": time.time() - epoch_start, "epoch": epoch}
        if device.type == "cuda":
            epoch_metrics["train/gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        if wb_run is not None:
            wb_run.log(epoch_metrics, step=step)
        if val_x is not None:
            model.eval()
            metrics = {"epoch": epoch}
            with torch.no_grad():
                for r, key in ((ratio, f"val/mse_{ratio}x"), (8, "val/mse_8x")):
                    total, n = 0.0, 0
                    for i in range(0, len(val_x), dc["batch_size"]):
                        y = val_x[i:i + dc["batch_size"]].to(device, non_blocking=True)
                        c = (val_coords[i:i + dc["batch_size"]].to(device, non_blocking=True)
                             if val_coords is not None else None)
                        total += loss_fn(fwd_eval(degrade(y, r), c), y).item() * y.shape[0]
                        n += y.shape[0]
                    metrics[key] = total / max(n, 1)
            print(f"epoch {epoch:03d} done | val {ratio}x {metrics[f'val/mse_{ratio}x']:.5f} "
                  f"| val 8x {metrics['val/mse_8x']:.5f} | "
                  f"{epoch_metrics['train/epoch_time_s']:.0f}s")
            if wb_run is not None:
                wb_run.log(metrics, step=step)
        t_last_log = time.time()  # exclude val time from throughput

        # ── Periodic reconstruction panels on fixed val patches ───────────
        if (val_x is not None and wb_run is not None
                and epoch % cfg["train"].get("sample_every_epochs", 10) == 0):
            fig_path = ensure_dir(cfg["paths"]["results_dir"]) / f"directmap_epoch{epoch:03d}.png"
            _save_recons(fwd_eval, val_x[:2],
                         None if val_coords is None else val_coords[:2],
                         normalizer, device, ratio, fig_path, raw_model,
                         disp=display_channel(cfg))
            import wandb as _wandb
            wb_run.log({"recons": _wandb.Image(str(fig_path))}, step=step)

        # ── Atomic per-epoch checkpoint (a crash loses at most one epoch) ─
        if dist.is_main:
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({
                "model": raw_model.state_dict(), "opt": opt.state_dict(),
                "scaler": scaler.state_dict(), "config": cfg,
                "train_ratio": train_ratios if train_ratios else ratio,
                "norm_mean": normalizer.mean, "norm_std": normalizer.std,
                "epoch": epoch, "step": step,
            }, tmp)
            tmp.replace(ckpt_path)

    if wb_run is not None:
        wb_run.finish()
    cleanup(dist)
    print(f"Done. Checkpoint -> {ckpt_path}")


@torch.no_grad()
def _save_recons(fwd, val_batch, val_coords, normalizer, device, ratio, path, model,
                 disp=0):
    """Fixed val patches: input (train ratio), prediction at train ratio and at
    8x, target — all on the target's color scale (display channel), comparable
    across epochs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    was_training = model.training
    model.eval()
    y = val_batch.to(device)
    cs = None if val_coords is None else val_coords.to(device)
    panels = []
    for i in range(len(y)):
        yi = y[i:i + 1]
        ci = None if cs is None else cs[i:i + 1]
        x4 = degrade(yi, ratio)
        panels.append([
            ("Input (LF)", x4), (f"Pred {ratio}x", fwd(x4, ci)),
            ("Pred 8x", fwd(degrade(yi, 8), ci)), ("Target", yi),
        ])
    fig, axes = plt.subplots(len(panels), 4, figsize=(16, 4 * len(panels)))
    axes = axes.reshape(len(panels), 4)
    for r, row in enumerate(panels):
        ref = normalizer.decode(row[-1][1].cpu())[0, disp].numpy()
        vmin, vmax = float(ref.min()), float(ref.max())
        for ax, (title, t) in zip(axes[r], row):
            ax.imshow(normalizer.decode(t.cpu())[0, disp].numpy(), cmap="RdBu_r",
                      vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
    fig.suptitle("Direct-map reconstructions (fixed val patches)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if was_training:
        model.train()
    print(f"  saved recons -> {path}")


if __name__ == "__main__":
    main()
