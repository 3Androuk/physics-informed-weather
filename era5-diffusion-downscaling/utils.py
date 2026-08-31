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


_VAR_SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "total_column_water_vapour": "tcwv",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "vertical_velocity": "w",
}


def channel_specs(dcfg: dict) -> list[dict]:
    """Per-channel {name, level} specs, in channel order.

    Multi-channel configs list them under data.variables; a legacy config's
    single data.variable/level becomes a one-element list."""
    if dcfg.get("variables"):
        return [{"name": v["name"], "level": v.get("level")} for v in dcfg["variables"]]
    return [{"name": dcfg["variable"], "level": dcfg.get("level")}]


def channel_label(name: str, level=None) -> str:
    """Short channel label, e.g. 2m_temperature -> t2m, geopotential@500 -> z500."""
    short = _VAR_SHORT.get(name, name)
    return f"{short}{int(level)}" if level is not None else short


def channel_labels(dcfg: dict) -> list[str]:
    return [channel_label(s["name"], s["level"]) for s in channel_specs(dcfg)]


def display_channel(cfg: dict) -> int:
    """Channel index used for figures and headline (physical-unit) metrics."""
    return int(cfg.get("eval", {}).get("display_channel", 0))


# Checkpoint-name tag per geo encoder ("" for the default hash grid, so
# existing checkpoint names like diffusion_geo.pt / diffusion_geo_hpx.pt are
# unchanged).
_ENCODER_TAG = {"hash": "", "healpix": "_hpx", "xyz": "_xyz",
                "sinusoidal": "_sin", "static": "_static",
                "hash_static": "_combo",
                "hash_compact": "_hashcompact", "xyz_static": "_xyzstatic",
                "sinusoidal_static": "_sinstatic"}


def geo_suffix(cfg: dict) -> str:
    """Checkpoint-name suffix identifying the geo conditioning: '' when geo is
    disabled, else '_geo' + the encoder tag (e.g. '_geo', '_geo_hpx',
    '_geo_static'), plus '_gated' when noise-dependent level gating is on."""
    g = cfg.get("geo", {})
    if not g.get("enabled", False):
        return ""
    encoder = g.get("encoder", "hash")
    if encoder not in _ENCODER_TAG:
        raise ValueError(f"unknown geo encoder: {encoder}")
    suffix = "_geo" + _ENCODER_TAG[encoder]
    if g.get("level_gating", False):
        suffix += "_gated"
    return suffix


def _var_tag(dcfg: dict) -> str:
    """Short dataset tag: channel label for single-variable runs, data.name
    (or '<C>ch') for multi-channel runs."""
    specs = channel_specs(dcfg) if (dcfg.get("variables") or dcfg.get("variable")) else []
    if len(specs) > 1:
        return dcfg.get("name") or f"{len(specs)}ch"
    if specs:
        return channel_label(specs[0]["name"], specs[0]["level"])
    return ""


def run_name(cfg: dict, *parts: str) -> str:
    """Canonical wandb run name: short variable + identity parts.

    Callers pass the checkpoint stem (which already encodes model kind, geo,
    encoder, mean type, and seed) plus any extra tags; empty parts are
    skipped. Example: run_name(cfg, 'diffusion_geo_hpx', 'resumed')
    -> 't2m-diffusion_geo_hpx-resumed'."""
    return "-".join(p for p in (_var_tag(cfg.get("data", {})), *parts) if p)


def init_wandb(cfg: dict, job_type: str, extra_config: dict | None = None,
               name: str | None = None):
    """Start a wandb run when cfg['wandb'].enabled is true.

    Opt-in: returns (None, None) when disabled so callers can guard with
    `if run is not None`. Returns (run, wandb_module) when enabled — the module
    is handed back so callers can build wandb.Image() etc. without re-importing.
    Name precedence: explicit wandb.name in the config > `name` argument >
    auto default (variable-geo-job)."""
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled"):
        return None, None
    import wandb
    config = {**cfg, **(extra_config or {})}
    name = wcfg.get("name") or name
    if not name and "data" in cfg:
        geo_tag = "geo" if cfg.get("geo", {}).get("enabled") else "base"
        name = f"{_var_tag(cfg['data'])}-{geo_tag}-{job_type}"
    run = wandb.init(
        project=wcfg.get("project", "era5-diffusion-downscaling"),
        entity=wcfg.get("entity"),
        name=name,
        job_type=job_type,
        config=config,
    )
    return run, wandb
