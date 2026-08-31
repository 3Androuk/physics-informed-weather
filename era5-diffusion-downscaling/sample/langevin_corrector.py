"""Null-space Langevin corrector over a frozen unconditional diffusion prior.

Post-hoc correction of a finished reconstruction: unadjusted Langevin steps
restricted to ker A (the subspace the coarse observation leaves free), using
the trained epsilon-network as the score of the sigma-smoothed model density.
Consistency is preserved exactly at every step because both the drift and the
injected noise are projected by P = I - A†A; the invariant target (up to
discretization and score error) is the model posterior restricted to
{x: coarsen(x) = y}. See docs/spectral_posterior_corrector.md for the
derivation, the guarantees, and the caveats.

Score estimate. The net expects inputs at noise level t; for a clean-scale
state x we re-noise, x_in = sqrt(abar)*x + sqrt(1-abar)*eps, and use

    s(x) ~= -(sqrt(abar)/sqrt(1-abar)) * eps_theta(x_in, t_eps)

— a stochastic estimate of the smoothed-density score (exact in expectation
for the Gaussian family; a fresh eps per step makes this stochastic-gradient
Langevin). t_eps small: the target is the sigma-smoothed posterior, the
honest compromise for score reliability near the data manifold.

Preconditioning. C = I (isotropic) or C = P F^-1 S F P (spectral), with S the
stationary spectrum from data.estimate_spectral_covariance. The drift uses C,
the noise uses C^(1/2) (elementwise sqrt of S in Fourier space), so the
discretized SDE is dX = C P s dtau + sqrt(2C) P dW. Step size follows the
Song et al. corrector signal-to-noise rule per sample, or a fixed --delta.
"""

from __future__ import annotations

import torch

from data.degrade import coarsen
from models.transport import nullspace_project, project_data_consistency


def load_spectral_power(path, device=None):
    """(1, C, H, W//2+1) covariance eigenvalues from the estimator's artifact."""
    import numpy as np
    with np.load(path) as data:
        power = torch.as_tensor(np.array(data["power"], dtype="float32"))
    if power.ndim == 2:
        power = power.unsqueeze(0)
    return power.unsqueeze(0).to(device) if device else power.unsqueeze(0)


def _spectral_apply(x: torch.Tensor, power: torch.Tensor, exponent: float):
    """F^-1 S^exponent F x for a stored rFFT spectrum S (per channel)."""
    freq = torch.fft.rfft2(x)
    return torch.fft.irfft2(freq * power.pow(exponent).to(freq.device),
                            s=x.shape[-2:])


@torch.no_grad()
def langevin_correct(model, diffusion, x, coarse, ratio, steps,
                     t_eps: int = 50, snr: float = 0.16, delta: float | None = None,
                     cond=None, power: torch.Tensor | None = None,
                     generator: torch.Generator | None = None,
                     t_schedule=None):
    """Run `steps` ker-A Langevin steps on a (consistent) reconstruction.

    Args:
        model: frozen epsilon-network (geo-conditioned nets get `cond`).
        diffusion: GaussianDiffusion (for the abar schedule).
        x: (N, C, H, W) clean-scale reconstruction (projected or not — the
            state is re-projected onto {coarsen(x) = coarse} on entry, and
            every update stays in ker A afterwards).
        coarse: (N, C, H/r, W/r) the observation the state must keep matching.
        ratio: block-averaging ratio r.
        steps: number of corrector steps (0 returns the projected input).
        t_eps: DDPM timestep of the score estimate (small = low smoothing).
        snr: Song-style per-sample adaptive step size; ignored when `delta`
            is given (fixed step).
        power: (1, C, H, W//2+1) spectrum for the spectral preconditioner;
            None = isotropic (C = I).
    Returns:
        (N, C, H, W) corrected state, exactly consistent with `coarse`.
    """
    x = project_data_consistency(x, coarse, ratio)
    if steps == 0:
        return x
    # Fixed-t Langevin targets p_t, NOT p_0 -- a systematically smoothed
    # distribution. t_schedule (one t per step, decreasing) anneals the
    # stationary target down toward p_0 instead.
    if t_schedule is not None:
        ts = [int(t) for t in t_schedule]
        if len(ts) != int(steps):
            raise ValueError(f"t_schedule has {len(ts)} entries for {steps} steps")
    else:
        ts = [int(t_eps)] * int(steps)

    def randn():
        if generator is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=generator,
                           device=generator.device if hasattr(generator, "device")
                           else "cpu", dtype=x.dtype).to(x.device)

    for t_k in ts:
        sa = diffusion.sqrt_abar[t_k]
        som = diffusion.sqrt_one_minus_abar[t_k]
        t_batch = torch.full((x.shape[0],), float(t_k), device=x.device)
        x_in = sa * x + som * randn()
        eps_hat = model(x_in, t_batch) if cond is None else model(x_in, t_batch, cond)
        score = -(sa / som) * eps_hat

        drift = nullspace_project(score, ratio)
        if power is not None:
            drift = nullspace_project(_spectral_apply(drift, power, 1.0), ratio)
        noise = randn()
        if power is not None:
            noise = _spectral_apply(noise, power, 0.5)
        noise = nullspace_project(noise, ratio)

        if delta is not None:
            step = torch.as_tensor(float(delta), device=x.device)
            step = step.view(1, 1, 1, 1).expand(x.shape[0], 1, 1, 1)
        else:
            # Song et al. corrector rule, per sample: delta = 2 (snr |z|/|s|)^2.
            flat = lambda v: v.flatten(1).norm(dim=1).clamp_min(1e-12)  # noqa: E731
            step = (2.0 * (snr * flat(noise) / flat(drift)) ** 2).view(-1, 1, 1, 1)
        x = x + step * drift + torch.sqrt(2.0 * step) * noise

    # Updates are all in ker A; this only clears accumulated float roundoff.
    return project_data_consistency(x, coarse, ratio)


def check_consistency(x, coarse, ratio):
    """Max abs coarse-consistency violation (diagnostic)."""
    return float((coarsen(x, ratio) - coarse).abs().max())
