"""Checkpoint loading and reconstruction for transport-based SR models."""

from __future__ import annotations

from pathlib import Path

import torch

from data.degrade import coarsen, degrade
from models.transport import build_transport, build_transport_model


def load_transport(ckpt_path, device, use_ema=True):
    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    method = ck["method"]
    model = build_transport_model(cfg, method)
    state = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    model.load_state_dict(state)
    model.eval().to(device)
    return model, build_transport(cfg, method), cfg, method


@torch.no_grad()
def reconstruct_transport(model, process, hf_norm, ratio, cfg, method,
                          coords=None, steps=None, solver=None, sampler=None,
                          stochasticity=None, projection=None):
    tc = cfg.get("transport", {})
    low_res = degrade(hf_norm, ratio)
    coarse = coarsen(hf_norm, ratio)
    kwargs = dict(
        coords=coords,
        steps=tc.get("sample_steps", 100) if steps is None else steps,
        solver=tc.get("solver", "heun") if solver is None else solver,
        project=tc.get("projection", "final") if projection is None else projection,
        coarse=coarse,
        ratio=ratio,
    )
    if method == "stochastic_interpolant":
        si = tc.get("stochastic_interpolant", {})
        kwargs.update(
            sampler=si.get("sampler", "ode") if sampler is None else sampler,
            stochasticity=(si.get("stochasticity", 0.1)
                           if stochasticity is None else stochasticity),
        )
    return process.sample(model, low_res, **kwargs)
