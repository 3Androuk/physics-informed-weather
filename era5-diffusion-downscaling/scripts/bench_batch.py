"""Sweep batch size x precision on ONE GPU to pick training settings from data.

Runs real training steps (the trainer's own loss, model and optimizer) on
synthetic patches of the configured shape, and reports peak GPU memory and
throughput for each combination. Synthetic input is deliberate: memory and step
time depend on tensor shapes and dtypes, not on the values, so this needs no
patches on disk and cannot be skewed by dataloader speed.

Why measure rather than guess: billing is proportional to the share allocated
(a 1-GPU job records billing=72 of a 288-core node), so an undersized batch
leaves paid-for GPU idle, and an oversized one dies partway into a long run.

    python scripts/bench_batch.py --config config/wb2_20var.yaml
    python scripts/bench_batch.py --config config/wb2_20var.yaml \
        --model diffusion --batches 16,32,64 --amp bf16

NOTE this measures throughput and the memory ceiling ONLY. It cannot tell you
whether a larger GLOBAL batch still converges — under torchrun the global batch
is this value x world size, and changing it changes the optimization regime.
config/wb2_20var.yaml already records that lr 2e-4 collapsed this model, so
scale the LR with the batch and watch the first epochs.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_config, resolve_amp  # noqa: E402


def _build(model_kind, cfg, device):
    """Return (model, step_fn) where step_fn(batch) -> loss."""
    if model_kind in ("flow", "stochastic_interpolant"):
        from models.transport import build_transport, build_transport_model
        from data.degrade import degrade
        model = build_transport_model(cfg, model_kind).to(device)
        process = build_transport(cfg, model_kind)
        ratio = int(cfg.get("transport", {}).get("train_ratios", [4])[0])

        def step(y, coords):
            return process.training_loss(model, y, degrade(y, ratio), coords)
        return model, step

    if model_kind == "diffusion":
        from models.diffusion import build_diffusion
        from models.unet import build_unet
        # Matches train_diffusion's non-geo path; the geo path wraps this in
        # GeoConditionedUNet, which only adds the embedding channels.
        model = build_unet(cfg, use_time=True).to(device)
        diff = build_diffusion(cfg).to(device)

        def step(y, coords):
            return diff.training_loss(model, y, cond=coords)
        return model, step

    raise ValueError(f"unknown --model {model_kind}")


def _geo_coords(cfg, batch, size, device):
    """Coordinate payload if geo is on, else None."""
    if not cfg.get("geo", {}).get("enabled", False):
        return None
    dim = int(cfg["geo"].get("input_dim", 3))
    return torch.rand(batch, size, size, dim, device=device)


def _measure(args, cfg, device, batch, chans, size, enabled, dtype):
    """One (batch, precision) point -> (peak GiB, seconds/step).

    A function rather than an inline block on purpose: everything allocated
    here goes out of scope on return, so the caching allocator can reuse it.
    `locals().pop(...)` would NOT do that — writes to locals() are discarded in
    CPython, and a leaked model makes the next, larger batch OOM spuriously.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, step_fn = _build(args.model, cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=enabled and dtype is torch.float16)
    y = torch.randn(batch, chans, size, size, device=device)
    coords = _geo_coords(cfg, batch, size, device)

    t0 = None
    for i in range(3 + args.steps):      # 3 warmup steps, then timed
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=enabled, dtype=dtype):
            loss = step_fn(y, coords)
            if isinstance(loss, tuple):  # some losses return (loss, extras...)
                loss = loss[0]
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()

    dt = (time.time() - t0) / args.steps
    peak = torch.cuda.max_memory_allocated() / 2**30
    return peak, dt


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--model", default="flow",
                    choices=["flow", "stochastic_interpolant", "diffusion"])
    ap.add_argument("--batches", default="8,16,24,32,48,64,96,128")
    ap.add_argument("--amp", default="off,bf16",
                    help="comma-separated: off, fp16, bf16")
    ap.add_argument("--steps", type=int, default=12,
                    help="timed steps per point (after 3 warmup steps).")
    ap.add_argument("--geo", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.geo:
        cfg.setdefault("geo", {})["enabled"] = True
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible — run this inside an srun/sbatch allocation")

    device = torch.device("cuda")
    size = int(cfg["patches"]["size"])
    chans = int(cfg["unet"]["in_channels"])
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"{name} | {total:.0f} GiB | model={args.model} "
          f"| {chans}ch {size}x{size} | geo={cfg.get('geo', {}).get('enabled', False)}")
    print(f"{'amp':>5} {'batch':>6} {'peak GiB':>9} {'%mem':>6} "
          f"{'step s':>8} {'samp/s':>8}  note")

    results = []
    for amp_mode in [a.strip() for a in args.amp.split(",") if a.strip()]:
        enabled, dtype = resolve_amp({"amp_dtype": amp_mode}, "cuda")
        for b in [int(x) for x in args.batches.split(",") if x.strip()]:
            try:
                peak, dt = _measure(args, cfg, device, b, chans, size,
                                    enabled, dtype)
            except torch.cuda.OutOfMemoryError:
                print(f"{amp_mode:>5} {b:>6} {'-':>9} {'-':>6} {'-':>8} {'-':>8}  OOM")
                torch.cuda.empty_cache()
                break   # larger batches will only OOM sooner
            print(f"{amp_mode:>5} {b:>6} {peak:>9.1f} {100*peak/total:>5.0f}% "
                  f"{dt:>8.3f} {b/dt:>8.1f}")
            results.append((amp_mode, b, peak, dt, b / dt))

    if results:
        best = max(results, key=lambda r: r[4])
        print(f"\nfastest: amp={best[0]} batch={best[1]} "
              f"-> {best[4]:.1f} samples/s at {best[2]:.1f} GiB")
        print("global batch under 4-GPU DDP would be "
              f"{best[1] * 4} — scale lr accordingly and watch early epochs.")


if __name__ == "__main__":
    main()
