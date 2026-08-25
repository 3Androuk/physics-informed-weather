"""CPU unit tests for the diagnostic battery (eval.diagnose + coherence)."""

import unittest

import numpy as np

from eval.diagnose import accumulate_error_map
from eval.metrics import radial_coherence


class CoherenceTests(unittest.TestCase):
    def test_self_coherence_is_one(self):
        rng = np.random.default_rng(0)
        f = rng.normal(size=(4, 16, 16))
        _, coh = radial_coherence(f, f)
        self.assertTrue(np.allclose(coh, 1.0, atol=1e-9))

    def test_independent_fields_are_incoherent(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(8, 32, 32))
        b = rng.normal(size=(8, 32, 32))
        _, coh = radial_coherence(a, b)
        # averaged over samples + annulus, uncorrelated fields stay low
        self.assertLess(np.median(coh[1:]), 0.3)

    def test_smoothed_field_loses_fine_coherence_only(self):
        # A low-passed copy stays coherent at coarse scales and decorrelates
        # at fine ones (relative to its own self-coherence of 1).
        rng = np.random.default_rng(1)
        f = rng.normal(size=(6, 32, 32))
        k = np.ones((5, 5)) / 25.0
        sm = np.stack([
            np.real(np.fft.ifft2(np.fft.fft2(x) * np.fft.fft2(k, s=x.shape)))
            for x in f])
        # re-center: the corner-anchored kernel translates by (2, 2), and the
        # coherence estimator (correctly) penalizes misregistration
        sm = np.roll(sm, (-2, -2), axis=(-2, -1))
        _, coh = radial_coherence(sm, f)
        self.assertGreater(np.mean(coh[1:5]), 0.7)
        self.assertGreater(np.mean(coh[1:5]), np.mean(coh[-5:]) + 0.2)


class ErrMapTests(unittest.TestCase):
    def test_overlap_average_and_uncovered_nan(self):
        errs = np.ones((2, 4, 4))
        errs[1] *= 3.0
        origins = np.array([[0, 0], [0, 2]])
        m = accumulate_error_map(errs, origins, (4, 8))
        self.assertEqual(m[0, 0], 1.0)     # only patch 0
        self.assertEqual(m[0, 3], 2.0)     # overlap: mean of 1 and 3
        self.assertEqual(m[0, 5], 3.0)     # only patch 1
        self.assertTrue(np.isnan(m[0, 7]))  # never covered


if __name__ == "__main__":
    unittest.main()
