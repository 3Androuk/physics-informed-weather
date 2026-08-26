"""Checkpoint loading and reconstruction for transport-based SR models."""

from __future__ import annotations

from pathlib import Path

import torch

from data.degrade import coarsen, degrade, upsample_nearest
from models.transport import build_transport, build_transport_model


def load_transport(ckpt_path, device, use_ema=True):
    """Load a transport checkpoint.

    Returns (model, process, cfg, method, residual). `residual` is None for
    full-field models; for residual-mode models it is a dict carrying the scale
    the residual was normalized by and the frozen mean it was trained against
    (None mean_model => bicubic).
    """
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    method = ck["method"]
    model = build_transport_model(cfg, method)
    state = ck["ema"] if use_ema and "ema" in ck else ck["model"]
    model.load_state_dict(state)
    model.eval().to(device)

    residual = None
    if ck.get("residual"):
        mean_model, mean_geo = None, False
        if ck.get("mean_ckpt"):
            from sample.reconstruct import load_directmap  # noqa: PLC0415
            mean_model, mean_cfg = load_directmap(
                ckpt_path.parent / ck["mean_ckpt"], device)
            mean_geo = mean_cfg.get("geo", {}).get("enabled", False)
        residual = {"res_std": ck.get("res_std", 1.0),
                    "mean_model": mean_model, "mean_geo": mean_geo,
                    # Exposed so callers can build the coords payload the mean
                    # needs when the transport model itself is not geo-conditioned.
                    "mean_geo_cfg": mean_cfg["geo"] if mean_geo else None,
                    # Sampling mirrors training: project the mean identically,
                    # and stay in ker A for null-space checkpoints.
                    "consistent_mean": ck.get("consistent_mean", False),
                    "null_space": ck.get("null_space", False)}
    return model, build_transport(cfg, method), cfg, method, residual


@torch.no_grad()
def _mean_field(hf_norm, ratio, coords, residual):
    """Deterministic mean: frozen learned regression, or bicubic.

    consistent_mean checkpoints project it onto {x: coarsen(x) == observation}
    — the orthogonal projection m = mu + A+(y - A mu), which cannot increase
    the mean's error and matches how the residual target was formed."""
    if residual["mean_model"] is not None:
        x = degrade(hf_norm, ratio)
        m = residual["mean_model"]
        mean = m(x, None, coords) if residual["mean_geo"] else m(x)
    else:
        lo = coarsen(hf_norm, ratio)
        mean = torch.nn.functional.interpolate(
            lo, size=hf_norm.shape[-2:], mode="bicubic", align_corners=False)
    if residual.get("consistent_mean"):
        from models.transport import project_data_consistency
        mean = project_data_consistency(mean, coarsen(hf_norm, ratio), ratio)
    return mean


@torch.no_grad()
def reconstruct_transport(model, process, hf_norm, ratio, cfg, method,
                          coords=None, steps=None, solver=None, sampler=None,
                          stochasticity=None, projection=None, residual=None):
    tc = cfg.get("transport", {})
    low_res = degrade(hf_norm, ratio)
    coarse = coarsen(hf_norm, ratio)
    project = tc.get("projection", "final") if projection is None else projection

    # Residual mode: the transported variable is the normalized residual, so the
    # data-consistency constraint coarsen(x) == coarse does NOT hold for it —
    # it holds for the composed field. Disable projection inside the sampler and
    # apply it once to mean + res_std * residual instead.
    cond_field, compose = low_res, None
    if residual is not None:
        cond_field = _mean_field(hf_norm, ratio, coords, residual)
        compose, project = project, "none"
        if residual.get("null_space"):
            # Consistency of the composite is structural (mean is consistent,
            # residual stays in ker A); the final compose projection remains
            # only as float-roundoff cleanup.
            kwargs_null = ratio
        else:
            kwargs_null = None
    else:
        kwargs_null = None

    kwargs = dict(
        coords=coords,
        steps=tc.get("sample_steps", 100) if steps is None else steps,
        solver=tc.get("solver", "heun") if solver is None else solver,
        project=project,
        coarse=coarse,
        ratio=ratio,
        null_ratio=kwargs_null,
    )
    if method == "stochastic_interpolant":
        si = tc.get("stochastic_interpolant", {})
        kwargs.update(
            sampler=si.get("sampler", "ode") if sampler is None else sampler,
            stochasticity=(si.get("stochasticity", 0.1)
                           if stochasticity is None else stochasticity),
        )
    out = process.sample(model, cond_field, **kwargs)
    if residual is None:
        return out
    out = cond_field + residual["res_std"] * out
    if compose and compose != "none":
        out = out + upsample_nearest(coarse - coarsen(out, ratio), out.shape[-2:])
    return out
