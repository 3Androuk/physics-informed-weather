"""CPU unit tests for the spectral Weather-DDNM linear algebra."""

import unittest

import torch
import torch.nn as nn

from data.degrade import coarsen, upsample_nearest
from models.diffusion import GaussianDiffusion
from sample.weather_ddnm import SpectralCovarianceProjector


class WeatherDDNMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.size = (16, 16)
        self.ratio = 4
        self.estimate = torch.randn(3, 1, *self.size)
        self.observation = torch.randn(3, 1, 4, 4)

    def test_white_covariance_equals_ordinary_pseudoinverse(self):
        projector = SpectralCovarianceProjector(
            torch.ones(1, 16, 9), self.size)
        actual = projector.project(self.estimate, self.observation, self.ratio)
        expected = self.estimate + upsample_nearest(
            self.observation - coarsen(self.estimate, self.ratio), self.size)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_colored_covariance_projection_is_exact(self):
        ky = torch.fft.fftfreq(16)[:, None]
        kx = torch.fft.rfftfreq(16)[None, :]
        power = 0.05 + 1.0 / (1.0 + 80.0 * (kx.square() + ky.square()))
        projector = SpectralCovarianceProjector(power, self.size)
        projected = projector.project(self.estimate, self.observation, self.ratio)
        self.assertTrue(torch.isfinite(projected).all())
        self.assertTrue(torch.allclose(
            coarsen(projected, self.ratio), self.observation,
            atol=2e-6, rtol=2e-6))

    def test_covariance_lift_is_exact(self):
        power = 0.1 + torch.rand(1, 16, 9)
        projector = SpectralCovarianceProjector(power, self.size)
        lifted = projector.lift(self.observation, self.ratio)
        self.assertEqual(lifted.shape, self.estimate.shape)
        self.assertTrue(torch.allclose(
            coarsen(lifted, self.ratio), self.observation,
            atol=2e-6, rtol=2e-6))

    def test_invalid_spectrum_is_rejected(self):
        with self.assertRaises(ValueError):
            SpectralCovarianceProjector(torch.zeros(1, 16, 9), self.size)

    def test_sampler_integration_uses_weather_projection_and_init(self):
        class ZeroNoise(nn.Module):
            def forward(self, x, timestep):
                return torch.zeros_like(x)

        projector = SpectralCovarianceProjector(
            0.1 + torch.rand(1, 16, 9), self.size)
        diffusion = GaussianDiffusion(timesteps=4)
        guidance = upsample_nearest(self.observation, self.size)
        initial_noise = torch.zeros(1, *guidance.shape)
        output = diffusion.guided_reconstruct(
            ZeroNoise(), guidance, t_steps=[2], K=1, project=True,
            lf=self.observation, ratio=self.ratio,
            covariance_projector=projector, covariance_init=True,
            init_noise=initial_noise,
        )
        self.assertTrue(torch.allclose(
            coarsen(output, self.ratio), self.observation,
            atol=2e-6, rtol=2e-6))


if __name__ == "__main__":
    unittest.main()
