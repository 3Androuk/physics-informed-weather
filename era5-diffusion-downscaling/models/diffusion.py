"""DDPM training objective + guided DDIM sampler.

Implements:
  - the standard DDPM noise-prediction loss (Ho et al. 2020), Eq. (2) of the
    paper, used to train on high-fidelity patches only;
  - the guided, intermediate-start DDIM sampler of Algorithm 2 with the
    physics-residual term dropped (w = 0) and sigma = 0, i.e. the physics-agnostic
    core of Shu et al. (2023).

Index convention: alphas_cumprod has length T+1 with abar[0] = 1 (clean data)
and abar[t] = prod_{s=1..t}(1 - beta_s). Training/sampling use t in {1, ..., T};
the backward chain ends at t = 0 where abar = 1, recovering x_0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_beta_schedule(schedule: str, t: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, t, dtype=torch.float64)
    if schedule == "cosine":
        # Nichol & Dhariwal cosine schedule.
        steps = t + 1
        s = 0.008
        x = torch.linspace(0, t, steps, dtype=torch.float64)
        ac = torch.cos(((x / t) + s) / (1 + s) * torch.pi / 2) ** 2
        ac = ac / ac[0]
        betas = 1 - (ac[1:] / ac[:-1])
        return betas.clamp(max=0.999)
    raise ValueError(f"unknown beta schedule: {schedule}")


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_schedule="linear",
                 beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.timesteps = timesteps
        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar = torch.cat([torch.ones(1, dtype=torch.float64), abar])  # length T+1, abar[0]=1
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", abar.float())
        self.register_buffer("sqrt_abar", abar.sqrt().float())
        self.register_buffer("sqrt_one_minus_abar", (1.0 - abar).sqrt().float())

    # ── forward process ──
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) noise. t: (N,) ints in [1, T]."""
        sa = self.sqrt_abar[t].view(-1, 1, 1, 1)
        som = self.sqrt_one_minus_abar[t].view(-1, 1, 1, 1)
        return sa * x0 + som * noise

    def training_loss(self, model: nn.Module, x0: torch.Tensor, cond=None,
                      return_details: bool = False):
        """DDPM simple loss: predict the injected noise (Eq. 2).

        `cond` (optional) is extra conditioning forwarded to the model — e.g. the
        per-pixel geographic coordinates (B, H, W, d) for a geo-conditioned UNet.
        Conditioning is never noised; only the atmospheric field x0 is.

        With return_details=True also returns the detached per-sample losses
        and the sampled timesteps (for logging loss-vs-t diagnostics).
        """
        n = x0.shape[0]
        t = torch.randint(1, self.timesteps + 1, (n,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t.float()) if cond is None else model(x_t, t.float(), cond)
        loss = F.mse_loss(pred, noise)
        if not return_details:
            return loss
        per_sample = (pred.detach() - noise).float().pow(2).mean(dim=(1, 2, 3))
        return loss, per_sample, t

    # ── guided DDIM sampling (Algorithm 2, physics term dropped, sigma=0) ──
    @torch.no_grad()
    def guided_reconstruct(
        self,
        model: nn.Module,
        x_guidance: torch.Tensor,
        t_steps,
        K: int = 1,
        eta: float = 0.0,
        stride: int = 1,
        progress: bool = False,
        cond=None,
        project: bool = False,
        lf: torch.Tensor = None,
        ratio: int = None,
    ) -> torch.Tensor:
        """Reconstruct a high-fidelity field from a noise-mixed LF guidance.

        With project=True (requires `lf`, the (N,1,h,w) coarse observation, and
        `ratio`), every step's x0 estimate is projected onto the constraint
        coarsen(x0) == lf: the model keeps its invented fine scales but the
        block averages are pinned to the observed input (ILVR-style data
        consistency). Without it the input is consulted only once, at the
        noise-mixing initialization, and the chain is free to drift.

        Args:
            model: trained noise predictor eps_theta(x_t, t).
            x_guidance: (N, 1, H, W) low-fidelity guidance x^(g), already
                normalized and on the HF grid (nearest-upsampled, optionally
                smoothed). The SAME model/guidance handles any ratio.
            t_steps: list of per-outer-loop start timesteps (len == K). Each is
                the max timestep t for that backward chain; the chain runs over
                tau = [0..t] (stride `stride`), ending at x_0.
            K: number of outer recursive-refinement loops (paper's K).
            eta: DDIM stochasticity (0 = deterministic, as in the paper).
            stride: take every `stride`-th timestep in the subsequence.

        Returns:
            (N, 1, H, W) reconstructed field (normalized units).
        """
        assert len(t_steps) == K, "t_steps must have one entry per outer loop K"
        if project:
            assert lf is not None and ratio is not None, "project=True needs lf and ratio"
            from data.degrade import coarsen, upsample_nearest
            hw = x_guidance.shape[-2:]
        x_g = x_guidance
        device = x_guidance.device
        iterator = range(K)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="outer K")

        for k in iterator:
            t0 = int(t_steps[k])
            # Increasing subsequence tau = {0, ..., t0}.
            seq = list(range(0, t0 + 1, stride))
            if seq[-1] != t0:
                seq.append(t0)

            # Noise mixing + intermediate start: x_t = sqrt(abar_t) x_g + sqrt(1-abar_t) eps.
            eps = torch.randn_like(x_g)
            sa = self.sqrt_abar[t0]
            som = self.sqrt_one_minus_abar[t0]
            x = sa * x_g + som * eps

            for i in reversed(range(1, len(seq))):
                ti, tprev = seq[i], seq[i - 1]
                a_i = self.alphas_cumprod[ti]
                a_prev = self.alphas_cumprod[tprev]

                t_batch = torch.full((x.shape[0],), ti, device=device, dtype=torch.float32)
                eps_theta = model(x, t_batch) if cond is None else model(x, t_batch, cond)

                x0_pred = (x - (1 - a_i).sqrt() * eps_theta) / a_i.sqrt()
                if project:
                    x0_pred = x0_pred + upsample_nearest(lf - coarsen(x0_pred, ratio), hw)
                sigma = eta * (
                    ((1 - a_prev) / (1 - a_i)).clamp(min=0).sqrt()
                    * (1 - a_i / a_prev).clamp(min=0).sqrt()
                )
                dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps_theta
                x = a_prev.sqrt() * x0_pred + dir_xt
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)

            x_g = x  # recursive refinement: this reconstruction guides the next loop
        return x_g

    @torch.no_grad()
    def sample_unconditional(self, model: nn.Module, shape, device, n_steps=100, eta=0.0,
                             cond=None):
        """Full DDIM sampling from pure noise (sanity check: plausible fields).

        `cond` (optional, (N,H,W,d)) supplies per-pixel coords for a
        geo-conditioned model — samples then show the learned prior at those
        fixed locations.
        """
        seq = torch.linspace(0, self.timesteps, n_steps + 1).round().long().tolist()
        seq = sorted(set(min(s, self.timesteps) for s in seq))
        x = torch.randn(shape, device=device)
        for i in reversed(range(1, len(seq))):
            ti, tprev = seq[i], seq[i - 1]
            a_i = self.alphas_cumprod[ti]
            a_prev = self.alphas_cumprod[tprev]
            t_batch = torch.full((shape[0],), ti, device=device, dtype=torch.float32)
            eps_theta = model(x, t_batch) if cond is None else model(x, t_batch, cond)
            x0_pred = (x - (1 - a_i).sqrt() * eps_theta) / a_i.sqrt()
            sigma = eta * (((1 - a_prev) / (1 - a_i)).clamp(min=0).sqrt()
                           * (1 - a_i / a_prev).clamp(min=0).sqrt())
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps_theta
            x = a_prev.sqrt() * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x


def build_diffusion(cfg: dict) -> GaussianDiffusion:
    d = cfg["diffusion"]
    return GaussianDiffusion(
        timesteps=d["timesteps"],
        beta_schedule=d["beta_schedule"],
        beta_start=d["beta_start"],
        beta_end=d["beta_end"],
    )
