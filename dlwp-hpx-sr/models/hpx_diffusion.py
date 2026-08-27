"""DDPM training objective + DDIM sampler on the HEALPix mesh.

The math is the sibling era5-diffusion-downscaling project's models/diffusion.py
(Ho et al. 2020 noise-prediction loss; DDIM sampler of Song et al. 2021 as used
by Shu et al. 2023 with the physics term dropped), carried over unchanged so the
two studies stay numerically comparable. Only the tensor rank differs: fields
here are (B, 12, C, F, F) HEALPix faces rather than (B, C, H, W) patches, so the
schedule coefficients broadcast over one extra axis.

What the sphere buys, and why this is not just the sibling model on new data:

  * **No tiles.** One sample is the whole globe, so there is no tiled or fused
    reconstruction, no overlap blending, no shared-noise bookkeeping, and no
    seams to suppress. Sampling cost is (steps x one global forward pass).

  * **The data-consistency projection is EXACT and global.** In nested
    ordering, coarsening a face by `ratio` is exactly average-pooling it, so
    replacing each coarse block's mean with the observed value

        x0 <- x0 + upsample_nearest(lf - coarsen(x0, ratio), ratio)

    yields coarsen(x0) == lf identically, everywhere on the sphere, in one
    operation. On the lat-lon patch pipeline the equivalent projection had to
    be applied per tile and then reconciled across tile boundaries.

Index convention (as in the sibling): alphas_cumprod has length T+1 with
abar[0] = 1 (clean data); training and sampling use t in {1, ..., T}.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen_faces, upsample_nearest_faces  # noqa: E402


def make_beta_schedule(schedule: str, t: int, beta_start: float,
                       beta_end: float) -> torch.Tensor:
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


def project_faces(x0: torch.Tensor, lf: torch.Tensor, ratio: int) -> torch.Tensor:
    """Pin block averages to the observation: coarsen(result, ratio) == lf.

    Exact on the mesh (see module docstring). x0: (B,12,C,F,F) estimate,
    lf: (B,12,C,F/r,F/r) observed coarse field.
    """
    return x0 + upsample_nearest_faces(lf - coarsen_faces(x0, ratio), ratio)


class HPXGaussianDiffusion(nn.Module):
    """DDPM/DDIM over (B, 12, C, F, F) HEALPix face tensors."""

    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        self.timesteps = timesteps
        betas = make_beta_schedule(beta_schedule, timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar = torch.cat([torch.ones(1, dtype=torch.float64), abar])  # abar[0]=1
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", abar.float())
        self.register_buffer("sqrt_abar", abar.sqrt().float())
        self.register_buffer("sqrt_one_minus_abar", (1.0 - abar).sqrt().float())

    # ── forward process ──────────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) noise. t: (B,) ints in [1, T]."""
        sa = self.sqrt_abar[t].view(-1, 1, 1, 1, 1)
        som = self.sqrt_one_minus_abar[t].view(-1, 1, 1, 1, 1)
        return sa * x0 + som * noise

    def training_loss(self, model: nn.Module, x0: torch.Tensor, cond=None,
                      return_details: bool = False):
        """DDPM simple loss: predict the injected noise.

        `cond` is conditioning forwarded to the model unchanged (here the
        deterministic mean field). It is never noised — only x0 is.
        """
        n = x0.shape[0]
        t = torch.randint(1, self.timesteps + 1, (n,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = model(x_t, t.float()) if cond is None else model(x_t, t.float(), cond)
        loss = F.mse_loss(pred, noise)
        if not return_details:
            return loss
        per_sample = (pred.detach() - noise).float().pow(2).mean(dim=(1, 2, 3, 4))
        return loss, per_sample, t

    # ── DDIM sampling ────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, model: nn.Module, mean_field: torch.Tensor, lf: torch.Tensor,
               ratio: int, n_steps: int = 100, eta: float = 0.0,
               project: bool = True, cond=None, generator=None,
               progress: bool = False) -> torch.Tensor:
        """Reconstruct a full field from pure noise, conditioned on `mean_field`.

        The chain models the RESIDUAL x0 = y - mean_field; the returned field is
        the composed reconstruction mean_field + x0. With project=True every
        step's x0 estimate is corrected so that the composed field's coarse
        block averages equal the observation `lf` exactly.

        Args:
            mean_field: (B,12,C,F,F) deterministic mean prediction (conditioning).
            lf: (B,12,C,F/r,F/r) observed coarse field.
            n_steps: DDIM subsequence length (<= timesteps).
            eta: DDIM stochasticity; 0 = deterministic, as in the paper.
        Returns:
            (B,12,C,F,F) reconstruction in the same (normalized) units.
        """
        device = mean_field.device
        seq = torch.linspace(0, self.timesteps, n_steps + 1).round().long().tolist()
        seq = sorted(set(min(s, self.timesteps) for s in seq))
        if cond is None:
            cond = (mean_field,)

        x = torch.randn(mean_field.shape, device=device, generator=generator,
                        dtype=mean_field.dtype)
        iterator = reversed(range(1, len(seq)))
        if progress:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc="ddim")

        for i in iterator:
            ti, tprev = seq[i], seq[i - 1]
            a_i = self.alphas_cumprod[ti]
            a_prev = self.alphas_cumprod[tprev]
            t_batch = torch.full((x.shape[0],), ti, device=device, dtype=torch.float32)
            eps_theta = model(x, t_batch, cond)

            x0_pred = (x - (1 - a_i).sqrt() * eps_theta) / a_i.sqrt()
            if project:
                # project the COMPOSED field, then return to residual space
                composed = project_faces(mean_field + x0_pred, lf, ratio)
                x0_pred = composed - mean_field
            sigma = eta * (
                ((1 - a_prev) / (1 - a_i)).clamp(min=0).sqrt()
                * (1 - a_i / a_prev).clamp(min=0).sqrt())
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps_theta
            x = a_prev.sqrt() * x0_pred + dir_xt
            if eta > 0:
                x = x + sigma * torch.randn(x.shape, device=device,
                                            generator=generator, dtype=x.dtype)

        out = mean_field + x
        return project_faces(out, lf, ratio) if project else out


def build_diffusion(cfg: dict) -> HPXGaussianDiffusion:
    d = cfg["diffusion"]
    return HPXGaussianDiffusion(
        timesteps=d["timesteps"],
        beta_schedule=d["beta_schedule"],
        beta_start=d["beta_start"],
        beta_end=d["beta_end"],
    )
