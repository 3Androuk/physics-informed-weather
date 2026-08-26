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

    def test_correction_is_smooth_where_ordinary_ddnm_is_blocky(self):
        # The mechanism: nearest-upsampling spreads the coarse residual as a
        # piecewise-constant field, injecting spurious power at and above the
        # block scale. The covariance correction must not.
        from eval.metrics import radial_power_spectrum
        size, ratio = (32, 32), 8
        ky = torch.fft.fftfreq(32)[:, None]
        kx = torch.fft.rfftfreq(32)[None, :]
        power = 1.0 / (0.01 + 40.0 * (kx.square() + ky.square()))
        projector = SpectralCovarianceProjector(power, size)
        residual = torch.randn(16, 1, 4, 4)
        ordinary = upsample_nearest(residual, size)
        smooth = projector.correction(residual, ratio)
        block_k = size[0] // ratio
        _, e_ord = radial_power_spectrum(ordinary)
        _, e_cov = radial_power_spectrum(smooth)
        self.assertLess(e_cov[block_k:].sum(), 0.1 * e_ord[block_k:].sum())

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


class LocalizationTests(unittest.TestCase):
    """Gaspari-Cohn tapering of the covariance kernel (periodicity fix)."""

    def setUp(self):
        from sample.weather_ddnm import localize_spectrum
        self.localize = localize_spectrum
        self.size = (32, 32)
        ky = torch.fft.fftfreq(32)[:, None]
        kx = torch.fft.rfftfreq(32)[None, :]
        # long-correlation covariance: without tapering it wraps the patch
        self.power = 1.0 / (0.002 + 20.0 * (kx.square() + ky.square()))

    def test_gaspari_cohn_shape(self):
        from sample.weather_ddnm import _gaspari_cohn
        r = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        g = _gaspari_cohn(r)
        self.assertAlmostEqual(float(g[0]), 1.0, places=6)
        self.assertEqual(float(g[-1]), 0.0)          # compact support
        self.assertEqual(float(g[4]), 0.0)           # exactly zero at 2
        self.assertTrue((g[:-2] >= g[1:-1] - 1e-6).all())  # non-increasing
        self.assertTrue((g >= 0).all())

    def test_localized_kernel_has_compact_support(self):
        radius = 4.0
        tapered = self.localize(self.power, self.size, radius)
        kernel = torch.fft.irfft2(tapered, s=self.size)[0]
        dy = torch.arange(32).float(); dy = torch.minimum(dy, 32 - dy)
        dist = torch.sqrt(dy[:, None] ** 2 + dy[None, :] ** 2)
        outside = kernel.abs()[dist > 2 * radius]
        self.assertLess(float(outside.max()), 1e-5 * float(kernel.abs().max()))

    def test_localized_covariance_stays_positive_definite(self):
        # Gaspari-Cohn is PD, so the Schur product keeps the spectrum positive
        # and the projector's own validation must accept it.
        tapered = self.localize(self.power, self.size, 6.0)
        self.assertTrue(torch.isfinite(tapered).all())
        self.assertTrue((tapered > 0).all())
        projector = SpectralCovarianceProjector(tapered, self.size)
        estimate = torch.randn(4, 1, *self.size)
        observation = torch.randn(4, 1, 4, 4)
        projected = projector.project(estimate, observation, 8)
        self.assertTrue(torch.allclose(coarsen(projected, 8), observation,
                                       atol=2e-6, rtol=2e-6))

    def test_localization_confines_the_correction(self):
        # A single coarse-cell residual must not push mass to the far edge.
        residual = torch.zeros(1, 1, 4, 4)
        residual[0, 0, 0, 0] = 1.0
        wide = SpectralCovarianceProjector(self.power, self.size)
        narrow = SpectralCovarianceProjector(
            self.localize(self.power, self.size, 3.0), self.size)
        far = (slice(None), slice(None), slice(14, 18), slice(14, 18))
        wide_far = wide.correction(residual, 8)[far].abs().mean()
        narrow_far = narrow.correction(residual, 8)[far].abs().mean()
        self.assertLess(float(narrow_far), float(wide_far))

    def test_large_radius_is_a_near_no_op(self):
        # Gaspari-Cohn decays from 1 immediately, so "no-op" needs a radius far
        # beyond the grid's own diameter, not merely larger than it.
        tapered = self.localize(self.power, self.size, 2048.0)
        rel = (tapered - self.power).abs().max() / self.power.max()
        self.assertLess(float(rel), 0.02)


class EstimationLeakageTests(unittest.TestCase):
    """Detrending and tapering in data.estimate_spectral_covariance."""

    def test_planar_detrend_removes_a_plane_exactly(self):
        from data.estimate_spectral_covariance import _planar_detrend
        import numpy as np
        h = w = 16
        yy, xx = np.mgrid[0:h, 0:w]
        plane = (3.0 + 0.5 * yy - 0.25 * xx)[None, None]
        out = _planar_detrend(plane.astype(np.float64))
        self.assertLess(float(np.abs(out).max()), 1e-10)

    def test_detrend_and_window_reduce_leakage(self):
        # A ramp plus a single sinusoid: leakage shows up as power at
        # wavenumbers where the true field has none.
        from data.estimate_spectral_covariance import estimate
        import numpy as np
        h = w = 32
        yy, xx = np.mgrid[0:h, 0:w]
        signal = np.sin(2 * np.pi * 4 * xx / w)
        patches = (signal + 0.8 * yy / h)[None, None].astype(np.float32)
        idx = np.arange(1)
        raw = estimate(patches, 0.0, 1.0, idx, 1, detrend=False, window="none")
        clean = estimate(patches, 0.0, 1.0, idx, 1, detrend=True, window="hann")
        # power away from the ramp (k_y=0) and the tone (k_x=4) is leakage
        mask = np.ones((h, w // 2 + 1), dtype=bool)
        mask[:2, :] = False
        mask[:, 3:6] = False
        self.assertLess(clean[0][mask].sum(), 0.5 * raw[0][mask].sum())

if __name__ == "__main__":
    unittest.main()
