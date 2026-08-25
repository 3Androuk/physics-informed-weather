"""Shared trainer for conditional flow matching and stochastic interpolants."""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import coarsen, degrade  # noqa: E402
from eval.metrics import spectrum_log_l1  # noqa: E402
from models.transport import build_transport, build_transport_model  # noqa: E402
from train.ema import EMA  # noqa: E402
from utils import (ensure_dir, geo_suffix, get_device, init_wandb,  # noqa: E402
                   load_config, run_name, set_seed)


def _parser(method: str):
    label = "flow matching" if method == "flow" else "stochastic interpolants"
    ap = argparse.ArgumentParser(description=f"Train conditional {label} for ERA5 SR.")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--geo", action="store_true",
                    help="Enable the configured geographic encoder.")
    ap.add_argument("--encoder", choices=["hash", "healpix", "xyz", "sinusoidal", "static", "hash_static"], default=None)
    ap.add_argument("--gated", action="store_true",
                    help="Force geo.level_gating: true — noise-dependent gating "
                         "of the embedding levels; checkpoint gains _gated.")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--residual", action="store_true",
                    help="Transport the RESIDUAL (y - mean)/res_std instead of the "
                         "full field, conditioning on the mean rather than the "
                         "nearest-upsampled LF. Mean is bicubic unless --mean-ckpt "
                         "is given. Shorter, straighter transport paths — the same "
                         "split-model idea as train_residual.py.")
    ap.add_argument("--mean-ckpt", default=None,
                    help="Checkpoint name in paths.ckpt_dir of a frozen "
                         "ratio-randomized regression mean (meanmap*.pt from "
                         "train_directmap --random-ratio). Implies --residual; the "
                         "checkpoint stem gains an _lm suffix and remembers the mean.")
    return ap


def run(method: str):
    args = _parser(method).parse_args()
    cfg = load_config(args.config)
    if args.wandb:
        cfg.setdefault("wandb", {})["enabled"] = True
    if args.geo:
        cfg.setdefault("geo", {})["enabled"] = True
    if args.encoder is not None:
        cfg.setdefault("geo", {})["encoder"] = args.encoder
    if args.gated:
        cfg.setdefault("geo", {})["level_gating"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed
    set_seed(cfg["seed"])

    device = get_device()
    train_cfg = cfg["train"]
    transport_cfg = cfg.get("transport", {})
    ratios = list(transport_cfg.get("train_ratios", [2, 4, 8]))
    if not ratios:
        raise ValueError("transport.train_ratios must not be empty")
    _validate_ratios(cfg["patches"]["size"], ratios, "transport.train_ratios")
    _validate_ratios(cfg["patches"]["size"],
                     transport_cfg.get("eval_ratios", [4, 8]),
                     "transport.eval_ratios")

    patch_dir = Path(cfg["paths"]["patch_dir"])
    ckpt_dir = ensure_dir(cfg["paths"]["ckpt_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)
    geo_on = cfg.get("geo", {}).get("enabled", False)

    # ── Residual mode: transport (y - mean)/res_std, conditioned on the mean.
    # The mean may be geo-conditioned even when the transport model is not, in
    # which case coords are still needed to evaluate it.
    residual_mode = args.residual or args.mean_ckpt is not None
    mean_cfg, mean_geo = None, False
    if args.mean_ckpt:
        from sample.reconstruct import load_directmap  # noqa: PLC0415
        mean_model, mean_cfg = load_directmap(ckpt_dir / args.mean_ckpt, device)
        for p in mean_model.parameters():
            p.requires_grad_(False)
        mean_model.eval()
        mean_geo = mean_cfg.get("geo", {}).get("enabled", False)
        print(f"Learned mean: {args.mean_ckpt} (geo={mean_geo}, frozen)")

        def mean_fn(y, r, coords=None):
            with torch.no_grad():
                x = degrade(y, r)
                return mean_model(x, None, coords) if mean_geo else mean_model(x)
    else:
        def mean_fn(y, r, coords=None):
            lo = coarsen(y, r)
            return torch.nn.functional.interpolate(
                lo, size=y.shape[-2:], mode="bicubic", align_corners=False)

    if args.mean_ckpt and mean_geo and geo_on:
        assert (mean_cfg["geo"].get("encoder", "hash")
                == cfg["geo"].get("encoder", "hash")), \
            "mean and transport geo encoders must match (they share one coords payload)"
    need_coords = geo_on or mean_geo

    ds_kwargs = (_geo_dataset_kwargs(cfg if geo_on else mean_cfg, patch_dir, "train")
                 if need_coords else {})
    dataset = PatchDataset(patch_dir / "train_patches.npy", normalizer, **ds_kwargs)
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], pin_memory=True, drop_last=True,
        persistent_workers=train_cfg["num_workers"] > 0,
    )

    # ── Residual normalization: one scalar std estimated once against the mean
    # actually in use, over all training ratios. The mean-field conditioning
    # channel tells the model which regime it is in, so a shared scale suffices;
    # the value is stored in the checkpoint so sampling can undo it.
    res_std = 1.0
    if residual_mode:
        with torch.no_grad():
            items = [dataset[i] for i in range(min(64, len(dataset)))]
            if need_coords:
                y64 = torch.stack([it[0] for it in items]).to(device)
                c64 = torch.stack([it[1] for it in items]).to(device)
            else:
                y64, c64 = torch.stack(items).to(device), None
            chunks = [(y64 - mean_fn(y64, r, c64)).flatten().cpu() for r in ratios]
            res_std = float(torch.cat(chunks).std())
            del y64, c64
        print(f"Residual std (normalized units): {res_std:.4f}")

    model = build_transport_model(cfg, method).to(device)
    process = build_transport(cfg, method)
    ema = EMA(model, decay=train_cfg["ema_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    use_amp = train_cfg.get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    stem = _checkpoint_stem(method, cfg, args.seed is not None,
                            residual_mode, args.mean_ckpt is not None)
    ckpt_path = ckpt_dir / f"{stem}.pt"
    start_epoch, step = 1, 0
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        if residual_mode and "res_std" in ck:
            # Reuse the stored scale rather than re-estimating: a fresh estimate
            # from a different 64-patch draw would silently change the target.
            res_std = ck["res_std"]
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"Resumed {ckpt_path} at epoch {ck['epoch']} (step {step})")
    elif args.resume:
        print(f"No checkpoint at {ckpt_path}; starting fresh")

    val_loader = _validation_loader(cfg, patch_dir, normalizer, need_coords,
                                    cfg if geo_on else mean_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{method} | ratios={ratios} | patches={len(dataset)} | "
          f"params={n_params:,} | geo={geo_on} | device={device}")

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(Path(cfg["paths"]["log_dir"]) / stem)
    except Exception:
        print("(tensorboard unavailable — skipping logging)")
    wb_run, wandb = init_wandb(
        cfg, job_type=f"train_{method}",
        extra_config={"method": method, "train_ratios": ratios,
                      "unet_params": n_params, "n_train_patches": len(dataset),
                      "residual": residual_mode, "res_std": res_std,
                      "mean_ckpt": args.mean_ckpt},
        name=run_name(cfg, stem, "resumed" if start_epoch > 1 else ""),
    )

    running, running_n, grad_sum = 0.0, 0, 0.0
    velocity_sum, score_sum = 0.0, 0.0
    last_log = time.time()
    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        model.train()
        epoch_sum, epoch_n = 0.0, 0
        epoch_start = time.time()
        for batch in loader:
            target, coords = _move_batch(batch, need_coords, device)
            ratio = random.choice(ratios)
            if residual_mode:
                # Condition on the mean field (strictly more informative than the
                # nearest-upsampled LF) and transport the normalized residual.
                cond_field = mean_fn(target, ratio, coords)
                train_target = (target - cond_field) / res_std
            else:
                cond_field, train_target = degrade(target, ratio), target
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                result = process.training_loss(
                    model, train_target, cond_field, coords, return_details=True)
                loss, _, _, *extra = result
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.get("grad_clip", 1.0)
                if train_cfg.get("grad_clip", 1.0) > 0 else float("inf"))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running += loss.item()
            running_n += 1
            grad_sum += grad_norm.item()
            epoch_sum += loss.item()
            epoch_n += 1
            if extra:
                velocity_sum += extra[0]["velocity"].mean().item()
                score_sum += extra[0]["score"].mean().item()
            step += 1

            if step % train_cfg["log_every"] == 0:
                now = time.time()
                metrics = {
                    "train/loss": running / running_n,
                    "train/grad_norm": grad_sum / running_n,
                    "train/imgs_per_sec": running_n * target.shape[0] / (now - last_log),
                    "train/ratio": ratio,
                    "epoch": epoch,
                }
                if method == "stochastic_interpolant":
                    metrics["train/velocity_loss"] = velocity_sum / running_n
                    metrics["train/score_loss"] = score_sum / running_n
                print(f"epoch {epoch:03d} step {step:07d} | "
                      f"loss {metrics['train/loss']:.5f} | "
                      f"grad {metrics['train/grad_norm']:.3f} | "
                      f"{metrics['train/imgs_per_sec']:.1f} img/s")
                _log_metrics(metrics, step, writer, wb_run)
                running = running_n = grad_sum = velocity_sum = score_sum = 0.0
                last_log = now

        epoch_metrics = {
            "train/epoch_loss": epoch_sum / max(epoch_n, 1),
            "train/epoch_time_s": time.time() - epoch_start,
            "epoch": epoch,
        }
        if val_loader is not None:
            epoch_metrics.update(_validate(
                process, ema.shadow, val_loader, device, need_coords, ratios,
                residual_mode, mean_fn, res_std))
            vals = " | ".join(f"{k} {v:.5f}" for k, v in epoch_metrics.items()
                              if k.startswith("val/"))
            print(f"epoch {epoch:03d} done | {vals}")
        _log_metrics(epoch_metrics, step, writer, wb_run)
        last_log = time.time()

        if (val_loader is not None
                and epoch % train_cfg.get("sample_every_epochs", 10) == 0):
            sample_path = results_dir / f"{stem}_epoch{epoch:03d}.png"
            sample_metrics = _save_samples(
                process, ema.shadow, val_loader.dataset, normalizer, device,
                need_coords, cfg, method, sample_path,
                residual_mode, mean_fn, res_std)
            _log_metrics(sample_metrics, step, writer, wb_run)
            if wb_run is not None:
                wb_run.log({"samples/reconstructions": wandb.Image(str(sample_path))}, step=step)

        if epoch % train_cfg.get("ckpt_every_epochs", 1) == 0 or epoch == train_cfg["epochs"]:
            if not _weights_finite(model) or not _weights_finite(ema.shadow):
                raise RuntimeError("non-finite transport weights; checkpoint not overwritten")
            _save_checkpoint(ckpt_path, model, ema, optimizer, scaler, cfg,
                             normalizer, method, epoch, step,
                             residual_mode, res_std, args.mean_ckpt)

    if writer:
        writer.close()
    if wb_run is not None:
        wb_run.finish()
    print(f"Done. Checkpoint -> {ckpt_path}")


def _geo_dataset_kwargs(cfg, patch_dir, split):
    g = cfg["geo"]
    return dict(
        origins_path=patch_dir / f"{split}_origins.npy",
        coords_full_path=patch_dir / "coords_full.npz",
        geo_input_dim=g["input_dim"], altitude=g.get("altitude"),
        geo_encoder=g.get("encoder", "hash"),
        healpix_index_path=((patch_dir / g["healpix_index"])
                            if g.get("healpix_index") else None),
    )


def _validate_ratios(size, ratios, name):
    invalid = [r for r in ratios if not isinstance(r, int) or r < 1 or size % r]
    if invalid:
        raise ValueError(f"{name} contains ratios that do not divide patch size "
                         f"{size}: {invalid}")


def _validation_loader(cfg, patch_dir, normalizer, need_coords, geo_cfg=None):
    path = patch_dir / "test_patches.npy"
    if not path.exists():
        return None
    kwargs = (_geo_dataset_kwargs(geo_cfg or cfg, patch_dir, "test")
              if need_coords else {})
    ds = PatchDataset(path, normalizer, **kwargs)
    n = min(int(cfg["train"].get("val_patches", 256)), len(ds))
    return DataLoader(Subset(ds, range(n)), batch_size=cfg["train"]["batch_size"],
                      shuffle=False, num_workers=0)


def _move_batch(batch, need_coords, device):
    if need_coords:
        target, coords = batch
        return target.to(device, non_blocking=True), coords.to(device, non_blocking=True)
    return batch.to(device, non_blocking=True), None


@torch.no_grad()
def _validate(process, model, loader, device, need_coords, ratios,
              residual_mode=False, mean_fn=None, res_std=1.0):
    was_training = model.training
    model.eval()
    totals = {r: 0.0 for r in ratios}
    count = 0
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)
        for batch in loader:
            target, coords = _move_batch(batch, need_coords, device)
            for ratio in ratios:
                if residual_mode:
                    cond_field = mean_fn(target, ratio, coords)
                    tgt = (target - cond_field) / res_std
                else:
                    cond_field, tgt = degrade(target, ratio), target
                loss = process.training_loss(model, tgt, cond_field, coords)
                totals[ratio] += loss.item() * target.shape[0]
            count += target.shape[0]
    if was_training:
        model.train()
    return {f"val/loss_{r}x": total / max(count, 1) for r, total in totals.items()}


@torch.no_grad()
def _save_samples(process, model, subset, normalizer, device, geo_on,
                  cfg, method, path, residual_mode=False, mean_fn=None,
                  res_std=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [subset[i] for i in range(min(2, len(subset)))]
    if geo_on:
        target = torch.stack([x[0] for x in items]).to(device)
        coords = torch.stack([x[1] for x in items]).to(device)
    else:
        target = torch.stack(items).to(device)
        coords = None
    ratios = cfg.get("transport", {}).get("eval_ratios", [4, 8])
    recs, panels = [], []
    for i in range(len(target)):
        row = []
        for ratio in ratios:
            y, c = target[i:i + 1], None if coords is None else coords[i:i + 1]
            low_res = degrade(y, ratio)
            coarse = coarsen(y, ratio)
            if residual_mode:
                mean_f = mean_fn(y, ratio, c)
                rec = mean_f + res_std * _sample(process, model, mean_f, c,
                                                 coarse, ratio, cfg, method)
            else:
                rec = _sample(process, model, low_res, c, coarse, ratio, cfg, method)
            recs.append((rec, y))
            row.extend([(f"Input {ratio}x", low_res), (f"Sample {ratio}x", rec)])
        row.append(("Target", target[i:i + 1]))
        panels.append(row)

    fig, axes = plt.subplots(len(panels), len(panels[0]),
                             figsize=(4 * len(panels[0]), 4 * len(panels)))
    axes = axes.reshape(len(panels), len(panels[0]))
    for row_idx, row in enumerate(panels):
        ref = normalizer.decode(row[-1][1].cpu())[0, 0].numpy()
        vmin, vmax = float(ref.min()), float(ref.max())
        for ax, (title, tensor) in zip(axes[row_idx], row):
            ax.imshow(normalizer.decode(tensor.cpu())[0, 0].numpy(),
                      cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
    fig.suptitle(f"{method.replace('_', ' ').title()} super-resolution")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    pred_phys = normalizer.decode(torch.cat([r for r, _ in recs]).cpu())
    truth_phys = normalizer.decode(torch.cat([y for _, y in recs]).cpu())
    return {"samples/spectrum_log_l1": spectrum_log_l1(pred_phys, truth_phys)}


def _sample(process, model, low_res, coords, coarse, ratio, cfg, method):
    tc = cfg.get("transport", {})
    common = dict(
        coords=coords, steps=tc.get("sample_steps", 100),
        solver=tc.get("solver", "heun"), project=tc.get("projection", "final"),
        coarse=coarse, ratio=ratio,
    )
    if method == "stochastic_interpolant":
        si = tc.get("stochastic_interpolant", {})
        common.update(sampler=si.get("sampler", "ode"),
                      stochasticity=si.get("stochasticity", 0.1))
    return process.sample(model, low_res, **common)


def _checkpoint_stem(method, cfg, seed_overridden, residual_mode=False,
                     learned_mean=False):
    stem = "flow_matching" if method == "flow" else "stochastic_interpolant"
    if residual_mode:
        stem += "_res"
    stem += geo_suffix(cfg)
    if learned_mean:
        stem += "_lm"
    if seed_overridden:
        stem += f"_s{cfg['seed']}"
    return stem


def _log_metrics(metrics, step, writer, wb_run):
    if writer:
        for key, value in metrics.items():
            writer.add_scalar(key, value, step)
    if wb_run is not None:
        wb_run.log(metrics, step=step)


@torch.no_grad()
def _weights_finite(model):
    return all(p.isfinite().all() for p in model.parameters())


def _save_checkpoint(path, model, ema, optimizer, scaler, cfg,
                     normalizer, method, epoch, step,
                     residual_mode=False, res_std=1.0, mean_ckpt=None):
    tmp = path.with_suffix(".pt.tmp")
    torch.save({
        "model": model.state_dict(), "ema": ema.state_dict(),
        "opt": optimizer.state_dict(), "scaler": scaler.state_dict(),
        "config": cfg, "method": method,
        "norm_mean": normalizer.mean, "norm_std": normalizer.std,
        "epoch": epoch, "step": step,
        # Residual mode needs both to reconstruct: the scale the residual was
        # normalized by, and which frozen mean it was trained against.
        "residual": residual_mode, "res_std": res_std, "mean_ckpt": mean_ckpt,
    }, tmp)
    tmp.replace(path)
