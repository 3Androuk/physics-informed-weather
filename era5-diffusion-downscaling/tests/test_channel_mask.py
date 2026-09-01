"""CPU tests: channel-wise inpainting mask (observed=) + forecast helpers."""

import unittest

import numpy as np
import torch
import torch.nn as nn

from data.degrade import coarsen, upsample_nearest
from models.diffusion import GaussianDiffusion


class ConvNet(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x, t, cond=None):
        return self.conv(x)


class MaskedGuidedReconstructTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.diffusion = GaussianDiffusion(timesteps=20)
        self.x_g = torch.randn(2, 3, 16, 16)
        self.lf = coarsen(self.x_g, 4)
        self.init = torch.randn(1, *self.x_g.shape)
        self.obs = torch.tensor([True, True, False])

    def _run(self, **kw):
        return self.diffusion.guided_reconstruct(
            ConvNet(), self.x_g, t_steps=[10], K=1, eta=0.0,
            init_noise=self.init, project=True, lf=self.lf, ratio=4, **kw)

    def test_all_observed_matches_default(self):
        self.assertTrue(torch.equal(
            self._run(), self._run(observed=torch.ones(3, dtype=torch.bool))))

    def test_observed_channels_stay_consistent_unobserved_are_free(self):
        out = self._run(observed=self.obs)
        err = (coarsen(out, 4) - self.lf).abs()
        self.assertLess(float(err[:, :2].max()), 1e-5)
        self.assertGreater(float(err[:, 2].max()), 1e-3)  # generated, not pinned

    def test_unobserved_guidance_and_lf_are_ignored(self):
        garbage_g = self.x_g.clone(); garbage_g[:, 2] = 999.0
        garbage_lf = self.lf.clone(); garbage_lf[:, 2] = -999.0
        clean_g = self.x_g.clone(); clean_g[:, 2] = 0.0
        a = self.diffusion.guided_reconstruct(
            ConvNet(), garbage_g, t_steps=[10], K=1, init_noise=self.init,
            project=True, lf=garbage_lf, ratio=4, observed=self.obs)
        b = self.diffusion.guided_reconstruct(
            ConvNet(), clean_g, t_steps=[10], K=1, init_noise=self.init,
            project=True, lf=self.lf, ratio=4, observed=self.obs)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_wrong_mask_length_rejected(self):
        with self.assertRaises(AssertionError):
            self._run(observed=torch.ones(4, dtype=torch.bool))


class TiledMaskTests(unittest.TestCase):
    def test_full_field_masked_projection(self):
        from sample.full_field import reconstruct_full_tiled_diffusion
        torch.manual_seed(1)
        diffusion = GaussianDiffusion(timesteps=8)
        coarse = torch.randn(1, 3, 8, 12)
        lf = upsample_nearest(coarse, (32, 48))
        obs = torch.tensor([True, False, True])
        out = reconstruct_full_tiled_diffusion(
            diffusion, ConvNet(), lf, coarse, 4, {"K": 1, "t_steps": [4]},
            tile=16, overlap=8, project_steps=True, observed=obs)
        err = (coarsen(out, 4) - coarse).abs()
        self.assertLess(float(err[:, [0, 2]].max()), 1e-4)
        self.assertGreater(float(err[:, 1].max()), 1e-3)


class ForecastHelperTests(unittest.TestCase):
    def test_coarse_grid_ratio6(self):
        from data.download_forecast import coarse_grid
        lat = np.linspace(60, -60, 481)
        lon = np.arange(0, 360, 0.25)
        lat_c, lon_c = coarse_grid(lat, lon, 6)
        self.assertEqual(len(lat_c), 80)     # 480 kept
        self.assertEqual(len(lon_c), 240)
        self.assertAlmostEqual(float(lat_c[0]), float(lat[:6].mean()), places=10)

    def test_observed_mask(self):
        from data.download_forecast import observed_mask
        specs = [{"name": "2m_temperature", "level": None},
                 {"name": "total_column_water_vapour", "level": None},
                 {"name": "geopotential", "level": 500}]
        m = observed_mask(specs, {"2m_temperature", "geopotential"})
        self.assertEqual(m.tolist(), [True, False, True])

    def test_rmse_per_channel_weighting(self):
        from eval.downscale_forecast import rmse_per_channel
        truth = torch.zeros(1, 2, 8, 8)
        pred = torch.zeros(1, 2, 8, 8)
        pred[:, 0, -2:] = 1.0                # polar-row error, channel 0
        pred[:, 1, :2] = 1.0                 # equator-row error, channel 1
        lat = np.linspace(0, 80, 8)
        unw = rmse_per_channel(pred, truth)
        w = rmse_per_channel(pred, truth, lat)
        self.assertLess(w[0], unw[0])        # polar error discounted
        self.assertGreater(w[1], unw[1])     # equatorial error upweighted


if __name__ == "__main__":
    unittest.main()
