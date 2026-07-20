"""Shared utilities: config loading, seeding, device, paths."""

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
    # Resolve all paths.* entries relative to project root.
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
    # Default name identifies variable + geo mode + job, so runs launched from
    # a shared config (e.g. baseline vs --geo) can't end up mislabeled.
    name = wcfg.get("name")
    if not name and "data" in cfg:
        geo_tag = "geo" if cfg.get("geo", {}).get("enabled") else "base"
        name = f"{cfg['data']['variable']}-{geo_tag}-{job_type}"
    run = wandb.init(
        project=wcfg.get("project", "era5-diffusion-downscaling"),
        entity=wcfg.get("entity"),
        name=name,
        job_type=job_type,
        config=config,
    )
    return run, wandb
