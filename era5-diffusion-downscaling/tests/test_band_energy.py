"""CPU tests for the block-pyramid band decomposition (H2CD gate 1)."""

import unittest

import torch

from data.degrade import coarsen
from eval.band_energy import block_bands


class BlockBandTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.x = torch.randn(4, 1, 32, 32)

    def test_bands_sum_to_identity(self):
        _, bands = block_bands(self.x, 16)
        recon = sum(bands)
        self.assertTrue(torch.allclose(recon, self.x, atol=1e-5))

    def test_energy_additivity_orthogonality(self):
        # Orthogonal projections: band energies must sum to total energy.
        _, bands = block_bands(self.x, 16)
        total = float(self.x.square().sum())
        parts = sum(float(b.square().sum()) for b in bands)
        self.assertLess(abs(parts - total) / total, 1e-5)

    def test_details_have_zero_parent_mean(self):
        # detail_{r/2 -> r} must vanish under coarsening at r (and coarser).
        labels, bands = block_bands(self.x, 16)
        for lab, b in zip(labels[:-1], bands[:-1]):
            r = int(lab.split("to")[1])
            self.assertLess(float(coarsen(b, r).abs().max()), 1e-5, lab)

    def test_free_bands_are_exactly_kerA(self):
        # For ratio r, the sum of bands finer than r has zero coarse content,
        # and the remaining bands reproduce coarsen(x, r) exactly.
        labels, bands = block_bands(self.x, 16)
        r = 4
        fine = sum(b for lab, b in zip(labels, bands)
                   if lab.startswith("detail_") and int(lab.split("to")[1]) <= r)
        rest = self.x - fine
        self.assertLess(float(coarsen(fine, r).abs().max()), 1e-5)
        self.assertTrue(torch.allclose(coarsen(rest, r), coarsen(self.x, r),
                                       atol=1e-5))

    def test_multichannel(self):
        x = torch.randn(2, 20, 32, 32)
        _, bands = block_bands(x, 8)
        self.assertTrue(torch.allclose(sum(bands), x, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
