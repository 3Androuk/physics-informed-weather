"""CPU tests for the null-space Langevin corrector."""

import unittest

import torch
import torch.nn as nn

from data.degrade import coarsen
from models.diffusion import GaussianDiffusion
from sample.langevin_corrector import (_spectral_apply, langevin_correct,
                                       load_spectral_power)


class RandomNet(nn.Module):
    """Arbitrary epsilon-net stand-in (wrong on purpose)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, t):
        return self.conv(x)


class GaussianNet(nn.Module):
    """Exact eps-net for x0 ~ N(0, I): E[eps | x_t] = sqrt(1-abar) * x_t."""

    def __init__(self, som):
        super().__init__()
        self.som = som

    def forward(self, x, t):
        return self.som * x


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.diffusion = GaussianDiffusion(timesteps=100)
        self.x = torch.randn(3, 1, 16, 16)
        self.coarse = torch.randn(3, 1, 4, 4)

    def test_zero_steps_is_projection_only(self):
        out = langevin_correct(RandomNet(), self.diffusion, self.x,
                               self.coarse, 4, steps=0)
        self.assertLess(float((coarsen(out, 4) - self.coarse).abs().max()), 1e-6)

    def test_isotropic_steps_preserve_consistency(self):
        out = langevin_correct(RandomNet(), self.diffusion, self.x,
                               self.coarse, 4, steps=12, t_eps=10)
        self.assertLess(float((coarsen(out, 4) - self.coarse).abs().max()), 1e-5)
        self.assertTrue(torch.isfinite(out).all())

    def test_spectral_steps_preserve_consistency(self):
        ky = torch.fft.fftfreq(16)[:, None]
        kx = torch.fft.rfftfreq(16)[None, :]
        power = (0.05 + 1.0 / (0.01 + 30.0 * (kx.square() + ky.square())))
        power = power.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W//2+1)
        out = langevin_correct(RandomNet(), self.diffusion, self.x,
                               self.coarse, 4, steps=12, t_eps=10, power=power)
        self.assertLess(float((coarsen(out, 4) - self.coarse).abs().max()), 1e-5)
        self.assertTrue(torch.isfinite(out).all())

    def test_fixed_delta_path(self):
        out = langevin_correct(RandomNet(), self.diffusion, self.x,
                               self.coarse, 4, steps=5, t_eps=10, delta=1e-3)
        self.assertLess(float((coarsen(out, 4) - self.coarse).abs().max()), 1e-5)


class GaussianEquilibriumTests(unittest.TestCase):
    """With the exact score of a standard-normal prior, the ker-A component
    must equilibrate near unit variance and zero mean."""

    def test_kerA_statistics_approach_target(self):
        torch.manual_seed(1)
        diffusion = GaussianDiffusion(timesteps=100)
        t_eps = 10
        som = float(diffusion.sqrt_one_minus_abar[t_eps])
        net = GaussianNet(som)
        x = torch.zeros(16, 1, 16, 16)          # start far inside the mode
        coarse = torch.zeros(16, 1, 8, 8)       # consistency: zero block means
        out = langevin_correct(net, diffusion, x, coarse, 2, steps=400,
                               t_eps=t_eps, delta=0.05)
        self.assertLess(float(coarsen(out, 2).abs().max()), 1e-4)
        std = float(out.std())
        # target ~ sqrt(1/abar) ~ 1.006 at t_eps=10 (plus SGLD inflation);
        # generous band — this is a statistical test, not a numerics one
        self.assertGreater(std, 0.6)
        self.assertLess(std, 1.5)
        self.assertLess(abs(float(out.mean())), 0.15)


class SpectralHelperTests(unittest.TestCase):
    def test_spectral_apply_roundtrip(self):
        torch.manual_seed(2)
        # A valid spectrum must carry the Hermitian symmetry of a real field's
        # periodogram (ky-symmetric in the kx=0/Nyquist columns) — arbitrary
        # random rFFT arrays don't, and irfft2 would silently symmetrize.
        power = 0.1 + torch.fft.rfft2(torch.randn(1, 1, 16, 16)).abs().square()
        x = torch.randn(2, 1, 16, 16)
        # S^{1/2} twice == S once
        twice = _spectral_apply(_spectral_apply(x, power, 0.5), power, 0.5)
        once = _spectral_apply(x, power, 1.0)
        self.assertTrue(torch.allclose(twice, once, atol=1e-4))

    def test_load_spectral_power_shape(self):
        import tempfile
        from pathlib import Path

        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cov.npz"
            np.savez(path, power=np.random.rand(1, 16, 9).astype("float32"),
                     image_size=np.array([16, 16]))
            power = load_spectral_power(path)
            self.assertEqual(tuple(power.shape), (1, 1, 16, 9))


if __name__ == "__main__":
    unittest.main()
