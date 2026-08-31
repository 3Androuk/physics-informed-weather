"""CPU tests for DPS likelihood guidance in the guided DDIM sampler."""

import unittest

import torch
import torch.nn as nn

from data.degrade import coarsen
from models.diffusion import GaussianDiffusion


class ZeroNet(nn.Module):
    """eps = 0: x0_hat == x_t / sqrt(abar). Even with no gradient through the
    network, DPS still acts via x0_hat's direct dependence on x_t."""

    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, t, cond=None):
        return self.conv(x)


class DPSTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.diffusion = GaussianDiffusion(timesteps=20)
        self.x_g = torch.randn(3, 1, 16, 16)
        self.lf = coarsen(self.x_g, 4)
        self.init = torch.randn(1, *self.x_g.shape)

    def _run(self, model, **kw):
        return self.diffusion.guided_reconstruct(
            model, self.x_g, t_steps=[10], K=1, eta=0.0,
            init_noise=self.init, lf=self.lf, ratio=4, **kw)

    def test_scale_zero_matches_default_path(self):
        base = self._run(ConvNet())
        dps0 = self._run(ConvNet(), dps_scale=0.0)
        self.assertTrue(torch.equal(base, dps0))

    def test_requires_lf_and_ratio(self):
        with self.assertRaises(AssertionError):
            self.diffusion.guided_reconstruct(
                ConvNet(), self.x_g, t_steps=[10], K=1, dps_scale=0.5)

    def test_guidance_reduces_coarse_residual(self):
        # Same init noise, deterministic chain: the DPS kicks must pull the
        # reconstruction toward coarse consistency relative to no guidance.
        free = self._run(ZeroNet())
        guided = self._run(ZeroNet(), dps_scale=0.5)
        err_free = (coarsen(free, 4) - self.lf).norm()
        err_guided = (coarsen(guided, 4) - self.lf).norm()
        self.assertLess(float(err_guided), float(err_free))
        self.assertTrue(torch.isfinite(guided).all())

    def test_gradient_flows_through_the_network(self):
        # With a conv net the DPS gradient includes the denoiser Jacobian, so
        # the kick differs from the ZeroNet-style direct term alone: outputs
        # must change when guidance is on.
        base = self._run(ConvNet())
        guided = self._run(ConvNet(), dps_scale=0.5)
        self.assertGreater(float((guided - base).abs().max()), 1e-6)
        self.assertTrue(torch.isfinite(guided).all())

    def test_composes_with_projection_and_stays_exact(self):
        out = self._run(ConvNet(), dps_scale=0.5, project=True)
        self.assertLess(float((coarsen(out, 4) - self.lf).abs().max()), 1e-5)

    def test_no_grad_leaks_outside_sampler(self):
        out = self._run(ConvNet(), dps_scale=0.5)
        self.assertFalse(out.requires_grad)


class ReconstructWrapperTests(unittest.TestCase):
    def test_reconstruct_diffusion_builds_lf_for_dps(self):
        from sample.reconstruct import reconstruct_diffusion
        torch.manual_seed(2)
        diffusion = GaussianDiffusion(timesteps=20)
        hf = torch.randn(2, 1, 16, 16)
        out = reconstruct_diffusion(diffusion, ZeroNet(), hf, 4,
                                    {"t_steps": [8], "K": 1}, dps_scale=0.5)
        self.assertEqual(out.shape, hf.shape)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
