"""Shared utilities: config loading, seeding, device, paths, wandb."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(path: str | os.PathLike = "config/default.yaml") -> dict:
    """Load a YAML config, resolving relative paths against the project root."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, val in cfg.get("paths", {}).items():
        p = Path(val)
        cfg["paths"][key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_amp(train_cfg: dict, device) -> tuple[bool, "torch.dtype"]:
    """(enabled, dtype) for mixed-precision training.

    `train.amp_dtype` defaults to **bfloat16**, not fp16: torch's autocast
    default on CUDA is float16, whose 5-bit exponent overflows easily and needs
    loss scaling to survive — that is what NaN'ed in the sibling project.
    bfloat16 keeps fp32's 8-bit exponent (trading mantissa bits instead), so the
    failure mode is precision, not blow-up, and no GradScaler is needed. Both
    the RTX 3090 Ti and the H100 have native bf16 tensor cores.

    Sampling and evaluation deliberately stay in fp32: inference is cheap, and
    the exact mesh projection (coarsen(pred) == lf) depends on the pooling
    arithmetic.
    """
    enabled = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    name = str(train_cfg.get("amp_dtype", "bfloat16")).lower()
    dtypes = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
              "float16": torch.float16, "fp16": torch.float16, "half": torch.float16}
    if name not in dtypes:
        raise ValueError(f"train.amp_dtype must be one of {sorted(set(dtypes))}, "
                         f"got {name!r}")
    dtype = dtypes[name]
    if enabled and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("train.amp_dtype is bfloat16 but this GPU lacks bf16 "
                           "support; set amp: false or amp_dtype: float16")
    if enabled:
        print(f"mixed precision: {name} "
              f"({'GradScaler on' if dtype is torch.float16 else 'no GradScaler needed'})")
    return enabled, dtype


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def init_wandb(cfg: dict, job_type: str, extra_config: dict | None = None):
    """Start a wandb run when cfg['wandb'].enabled is true.

    Opt-in: returns (None, None) when disabled so callers can guard with
    `if run is not None`. Returns (run, wandb_module) when enabled — the module
    is handed back so callers can build wandb.Image() etc. without re-importing.
    """
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled"):
        return None, None
    import wandb
    config = {**cfg, **(extra_config or {})}
    run = wandb.init(
        project=wcfg.get("project", "dlwp-hpx-sr"),
        entity=wcfg.get("entity"),
        name=wcfg.get("name"),
        job_type=job_type,
        config=config,
    )
    return run, wandb
