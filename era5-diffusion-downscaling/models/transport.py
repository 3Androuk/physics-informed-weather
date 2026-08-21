"""Conditional flow-matching and stochastic-interpolant models.

Both methods transport a standard-normal source field to the conditional ERA5
high-resolution distribution.  The upsampled low-resolution observation is a
conditioning channel; optional geographic features use the same encoder as the
DDPM experiments.

Time is represented continuously in [0, 1].  ``time_scale`` expands it before
the existing sinusoidal UNet embedding so that the embedding does not collapse
to a tiny section of its wavelength range.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from data.degrade import coarsen, upsample_nearest
from models.unet import build_unet


class ConditionalTransportUNet(nn.Module):
    """UNet vector field conditioned on LF input and optional geography."""

    def __init__(self, base_unet: nn.Module, time_scale: float = 1000.0,
                 geo_encoder: nn.Module | None = None, level_gate=None):
        super().__init__()
        self.unet = base_unet
        self.geo = geo_encoder
        self.gate = level_gate
        self.time_scale = float(time_scale)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond):
        low_res, coords = cond
        chans = [x_t, low_res]
        if self.geo is not None:
            if coords is None:
                raise ValueError("geo-conditioned transport model needs coordinates")
            from models.geo_encoding import checkpointed_embed
            emb = checkpointed_embed(self.geo, coords)
            if self.gate is not None:
                # transport time runs 0 (noise) -> 1 (data): t IS the signal fraction
                emb = self.gate(emb, t.float().clamp(0.0, 1.0))
            chans.append(emb.permute(0, 3, 1, 2).contiguous())
        return self.unet(torch.cat(chans, dim=1), t * self.time_scale)


def build_transport_model(cfg: dict, method: str) -> ConditionalTransportUNet:
    """Build a conditional vector field; SI has velocity and score heads."""
    if method not in {"flow", "stochastic_interpolant"}:
        raise ValueError(f"unknown transport method: {method}")
    geo_on = cfg.get("geo", {}).get("enabled", False)
    if geo_on:
        from models.geo_encoding import build_geo_encoder, build_level_gate
        geo_encoder = build_geo_encoder(cfg)
        geo_channels = geo_encoder.output_dim
        level_gate = build_level_gate(cfg)
    else:
        geo_encoder, geo_channels, level_gate = None, 0, None

    # Current field + the C-channel LF conditioning + optional geo embedding.
    # Stochastic interpolants predict [velocity, scaled_score].
    model_cfg = {**cfg, "unet": dict(cfg["unet"])}
    model_cfg["unet"]["out_channels"] = (
        2 * cfg["unet"]["out_channels"]
        if method == "stochastic_interpolant" else cfg["unet"]["out_channels"]
    )
    base = build_unet(model_cfg, use_time=True,
                      extra_in_channels=cfg["unet"]["in_channels"] + geo_channels)
    return ConditionalTransportUNet(
        base, cfg.get("transport", {}).get("time_scale", 1000.0), geo_encoder,
        level_gate=level_gate,
    )


def _batch_time(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rand(x.shape[0], device=x.device) * (1.0 - 2.0 * eps) + eps


def _expand(t: torch.Tensor) -> torch.Tensor:
    return t.view(-1, 1, 1, 1)


class FlowMatching:
    """Conditional flow matching on x_t = (1-t) z + t x_data."""

    def __init__(self, time_epsilon: float = 1e-4):
        self.time_epsilon = float(time_epsilon)
        if not 0.0 <= self.time_epsilon < 0.5:
            raise ValueError("time_epsilon must be in [0, 0.5)")

    def training_loss(self, model: nn.Module, target: torch.Tensor,
                      low_res: torch.Tensor, coords=None, return_details=False):
        t = _batch_time(target, self.time_epsilon)
        z = torch.randn_like(target)
        xt = (1.0 - _expand(t)) * z + _expand(t) * target
        velocity = target - z
        pred = model(xt, t, (low_res, coords))
        per_sample = (pred - velocity).float().pow(2).mean(dim=(1, 2, 3))
        loss = per_sample.mean()
        return (loss, per_sample.detach(), t.detach()) if return_details else loss

    @torch.no_grad()
    def sample(self, model: nn.Module, low_res: torch.Tensor, coords=None,
               steps: int = 100, solver: str = "heun", project: str = "none",
               coarse: torch.Tensor | None = None, ratio: int | None = None,
               noise: torch.Tensor | None = None):
        return integrate_transport(model, low_res, coords, steps, solver,
                                   project, coarse, ratio, noise=noise)


class StochasticInterpolant:
    """Noisy interpolant with joint velocity and score regression.

    I_t = (1-t) z + t x_data + gamma sin(pi t) epsilon.

    The second model head predicts ``sigma_t * score_t``.  Scaling keeps the
    regression target bounded as the conditional variance vanishes near t=1.
    The learned probability-flow velocity can be sampled deterministically, or
    combined with the score in an SDE that preserves the same marginals.
    """

    def __init__(self, gamma: float = 0.5, score_weight: float = 1.0,
                 time_epsilon: float = 1e-4):
        self.gamma = float(gamma)
        self.score_weight = float(score_weight)
        self.time_epsilon = float(time_epsilon)
        if self.gamma < 0 or self.score_weight < 0:
            raise ValueError("gamma and score_weight must be non-negative")
        if not 0.0 <= self.time_epsilon < 0.5:
            raise ValueError("time_epsilon must be in [0, 0.5)")

    def path(self, target: torch.Tensor, t: torch.Tensor):
        z, eps = torch.randn_like(target), torch.randn_like(target)
        te = _expand(t)
        bridge = self.gamma * torch.sin(math.pi * te)
        bridge_dot = self.gamma * math.pi * torch.cos(math.pi * te)
        gaussian_part = (1.0 - te) * z + bridge * eps
        sigma = torch.sqrt((1.0 - te).square() + bridge.square()).clamp_min(1e-6)
        xt = te * target + gaussian_part
        velocity = target - z + bridge_dot * eps
        scaled_score = -gaussian_part / sigma
        return xt, velocity, scaled_score

    def training_loss(self, model: nn.Module, target: torch.Tensor,
                      low_res: torch.Tensor, coords=None, return_details=False):
        t = _batch_time(target, self.time_epsilon)
        xt, velocity, scaled_score = self.path(target, t)
        pred_v, pred_s = model(xt, t, (low_res, coords)).chunk(2, dim=1)
        v_per = (pred_v - velocity).float().pow(2).mean(dim=(1, 2, 3))
        s_per = (pred_s - scaled_score).float().pow(2).mean(dim=(1, 2, 3))
        per_sample = v_per + self.score_weight * s_per
        loss = per_sample.mean()
        if not return_details:
            return loss
        details = {"velocity": v_per.detach(), "score": s_per.detach()}
        return loss, per_sample.detach(), t.detach(), details

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        bridge = self.gamma * torch.sin(math.pi * t)
        return torch.sqrt((1.0 - t).square() + bridge.square()).clamp_min(1e-6)

    @torch.no_grad()
    def sample(self, model: nn.Module, low_res: torch.Tensor, coords=None,
               steps: int = 100, solver: str = "heun", sampler: str = "ode",
               stochasticity: float = 0.1, project: str = "none",
               coarse: torch.Tensor | None = None, ratio: int | None = None,
               noise: torch.Tensor | None = None):
        if sampler == "ode":
            return integrate_transport(model, low_res, coords, steps, solver,
                                       project, coarse, ratio, split_velocity=True,
                                       noise=noise)
        if sampler != "sde":
            raise ValueError("sampler must be 'ode' or 'sde'")
        return self._sample_sde(model, low_res, coords, steps, stochasticity,
                                project, coarse, ratio, noise)

    def _sample_sde(self, model, low_res, coords, steps, stochasticity,
                    project, coarse, ratio, noise=None):
        _validate_sampling(steps, "euler", project, coarse, ratio)
        if stochasticity < 0:
            raise ValueError("stochasticity must be non-negative")
        # Only the INITIAL state is shareable across overlapping tiles; the
        # per-step dW increments stay independent (overlap-blending averages
        # the residual disagreement).
        x = _initial_noise(low_res, noise)
        dt = 1.0 / steps
        for i in range(steps):
            t_value = i / steps
            t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
            pred_v, pred_scaled_score = model(x, t, (low_res, coords)).chunk(2, dim=1)
            sigma = self.sigma(t).view(-1, 1, 1, 1)
            score = pred_scaled_score / sigma
            # lambda(t) vanishes at both endpoints.  Adding lambda*score to the
            # probability-flow drift and sqrt(2 lambda)dW preserves marginals.
            rate = float(stochasticity) * 4.0 * t_value * (1.0 - t_value)
            x = x + dt * (pred_v + rate * score)
            if rate > 0:
                x = x + math.sqrt(2.0 * rate * dt) * torch.randn_like(x)
            if project == "each":
                x = project_data_consistency(x, coarse, ratio)
        if project == "final":
            x = project_data_consistency(x, coarse, ratio)
        return x


def _velocity(model, x, t, cond, split_velocity):
    out = model(x, t, cond)
    return out.chunk(2, dim=1)[0] if split_velocity else out


def _validate_sampling(steps, solver, project, coarse, ratio):
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'")
    if project not in {"none", "final", "each"}:
        raise ValueError("project must be 'none', 'final', or 'each'")
    if project != "none" and (coarse is None or ratio is None):
        raise ValueError("data-consistency projection needs coarse and ratio")


def _initial_noise(low_res: torch.Tensor, noise: torch.Tensor | None) -> torch.Tensor:
    """Fresh Gaussian noise, or a caller-supplied field (e.g. tiles cropped
    from one global noise field so overlapping tiles agree)."""
    if noise is None:
        return torch.randn_like(low_res)
    if noise.shape != low_res.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != input shape "
                         f"{tuple(low_res.shape)}")
    return noise.to(device=low_res.device, dtype=low_res.dtype)


@torch.no_grad()
def integrate_transport(model: nn.Module, low_res: torch.Tensor, coords=None,
                        steps: int = 100, solver: str = "heun",
                        project: str = "none", coarse=None, ratio=None,
                        split_velocity: bool = False,
                        noise: torch.Tensor | None = None):
    """Euler/Heun integration of a learned probability-flow ODE, t=0 -> 1."""
    _validate_sampling(steps, solver, project, coarse, ratio)
    x = _initial_noise(low_res, noise)
    dt = 1.0 / steps
    cond = (low_res, coords)
    for i in range(steps):
        t0 = torch.full((x.shape[0],), i / steps, device=x.device, dtype=x.dtype)
        v0 = _velocity(model, x, t0, cond, split_velocity)
        if solver == "euler":
            x = x + dt * v0
        else:
            proposal = x + dt * v0
            t1 = torch.full_like(t0, (i + 1) / steps)
            v1 = _velocity(model, proposal, t1, cond, split_velocity)
            x = x + 0.5 * dt * (v0 + v1)
        if project == "each":
            x = project_data_consistency(x, coarse, ratio)
    if project == "final":
        x = project_data_consistency(x, coarse, ratio)
    return x


def project_data_consistency(x: torch.Tensor, coarse: torch.Tensor,
                             ratio: int) -> torch.Tensor:
    """Exact block-average projection: coarsen(output, ratio) == observation."""
    return x + upsample_nearest(coarse - coarsen(x, ratio), x.shape[-2:])


def build_transport(cfg: dict, method: str):
    tc = cfg.get("transport", {})
    if method == "flow":
        return FlowMatching(tc.get("time_epsilon", 1e-4))
    if method == "stochastic_interpolant":
        sc = tc.get("stochastic_interpolant", {})
        return StochasticInterpolant(
            gamma=sc.get("gamma", 0.5),
            score_weight=sc.get("score_weight", 1.0),
            time_epsilon=tc.get("time_epsilon", 1e-4),
        )
    raise ValueError(f"unknown transport method: {method}")
