"""Show the status of every training run from its checkpoint.

Each trainer checkpoints atomically every epoch and stores the epoch counter,
so the checkpoint file is the ground truth of how far a run got — independent
of tmux scrollback or wandb state. Also shows the file's last-modified age: a
DONE run stops updating; an "in progress" run whose file hasn't changed in a
long time is a crashed run that needs --resume.

Run:
    python status.py --config config/t2m.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_config  # noqa: E402


def _age(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def main():
    ap = argparse.ArgumentParser(description="Training-run status from checkpoints.")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    if not ckpt_dir.exists():
        raise SystemExit(f"no checkpoint dir: {ckpt_dir}")

    now = time.time()
    print(f"{'checkpoint':45s} {'progress':>16s}   {'updated':>10s}   state")
    print("-" * 90)
    for p in sorted(ckpt_dir.glob("*.pt")):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001 - mid-write or corrupt file
            print(f"{p.name:45s} {'?':>16s}   {_age(now - p.stat().st_mtime):>10s}   UNREADABLE ({type(e).__name__})")
            continue
        epoch = ck.get("epoch")
        target = (cfg["directmap"]["epochs"]
                  if p.name.startswith(("directmap", "meanmap"))
                  else cfg["train"]["epochs"])
        age = now - p.stat().st_mtime
        if epoch is None:
            progress, state = "no epoch field", "legacy checkpoint"
        elif epoch >= target:
            progress, state = f"epoch {epoch}/{target}", "DONE"
        else:
            progress = f"epoch {epoch}/{target}"
            # per-epoch saves: silence much longer than an epoch means crashed
            state = "in progress" if age < 3600 else "STALLED? (crashed -> --resume)"
        print(f"{p.name:45s} {progress:>16s}   {_age(age):>10s}   {state}")


if __name__ == "__main__":
    main()
