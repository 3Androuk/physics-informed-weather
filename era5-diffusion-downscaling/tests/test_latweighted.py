"""CPU tests for latitude-weighted RMSE (WeatherBench2-style area weighting)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from eval.metrics import (l2_norm, l2_norm_weighted, latitude_weights,
                          patch_latitudes)


class LatitudeWeightTests(unittest.TestCase):
    def test_weights_have_unit_mean(self):
        w = latitude_weights(np.linspace(-60, 60, 32))
        self.assertAlmostEqual(float(w.mean()), 1.0, places=10)
        self.assertTrue((w > 0).all())

    def test_weights_decrease_poleward(self):
        w = latitude_weights(np.array([0.0, 30.0, 60.0, 80.0]))
        self.assertTrue((np.diff(w) < 0).all())

    def test_equator_grid_matches_unweighted(self):
        # A patch at the equator has near-constant cos(lat): weighted RMSE
        # must agree closely with the plain one.
        torch.manual_seed(0)
        pred, truth = torch.randn(4, 1, 8, 8), torch.randn(4, 1, 8, 8)
        lat = np.linspace(-0.5, 0.5, 8)
        self.assertAlmostEqual(l2_norm_weighted(pred, truth, lat),
                               l2_norm(pred, truth), places=3)

    def test_weighting_discounts_polar_error(self):
        # Error concentrated in the poleward rows must score LOWER weighted
        # than unweighted; concentrated at the equator, higher.
        lat = np.linspace(0.0, 80.0, 16)
        truth = torch.zeros(1, 1, 16, 16)
        polar = torch.zeros(1, 1, 16, 16)
        polar[..., -4:, :] = 1.0
        equatorial = torch.zeros(1, 1, 16, 16)
        equatorial[..., :4, :] = 1.0
        self.assertLess(l2_norm_weighted(polar, truth, lat),
                        l2_norm(polar, truth))
        self.assertGreater(l2_norm_weighted(equatorial, truth, lat),
                           l2_norm(equatorial, truth))

    def test_per_patch_latitudes(self):
        torch.manual_seed(1)
        pred, truth = torch.randn(3, 1, 8, 8), torch.randn(3, 1, 8, 8)
        lat = np.stack([np.linspace(a, a + 10, 8) for a in (0.0, 30.0, 60.0)])
        val = l2_norm_weighted(pred, truth, lat)
        self.assertTrue(np.isfinite(val))
        self.assertGreater(val, 0.0)

    def test_shape_mismatch_rejected(self):
        pred = torch.randn(2, 1, 8, 8)
        with self.assertRaises(ValueError):
            l2_norm_weighted(pred, pred.clone(), np.linspace(0, 10, 5))

    def test_patch_latitudes_roundtrip_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(patch_latitudes(d, 4, 8))    # legacy dir
            np.save(d / "test_origins.npy", np.array([[0, 0], [4, 2]]))
            np.savez(d / "coords_full.npz",
                     lat=np.linspace(-90, 90, 32), lon=np.linspace(0, 360, 32))
            lat = patch_latitudes(d, 2, 8)
            self.assertEqual(lat.shape, (2, 8))
            self.assertTrue(np.all(lat[1] > lat[0]))       # second patch is north


if __name__ == "__main__":
    unittest.main()
