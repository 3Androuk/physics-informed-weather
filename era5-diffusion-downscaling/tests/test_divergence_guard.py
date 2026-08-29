"""DivergenceGuard, tested against the real trajectories that motivated it.

Job 6169725 (20-var no-geo baseline, 4x GH200, global batch 128, lr 1e-4)
trained healthily to epoch 89 and then collapsed to the trivial solution,
running 108 further epochs at full cost. The same run had ALSO spiked at epoch
10 and recovered completely, so the guard has to separate the two. Both traces
below are the observed numbers, not invented ones.
"""

import math
import unittest

from utils import DivergenceGuard, build_divergence_guard

# Observed val losses, job 6169725. Epoch 10's 0.18781 is 6.4x the then-best
# 0.02914 and recovered by epoch 11 — a single-epoch trigger would kill this.
EARLY_SPIKE = [
    (1, 0.11369), (2, 0.07449), (3, 0.05794), (4, 0.04693), (5, 0.04258),
    (6, 0.03691), (7, 0.03346), (8, 0.03099), (9, 0.02914), (10, 0.18781),
    (11, 0.03108), (12, 0.02802), (13, 0.02642), (14, 0.02545),
]
# ... and the actual collapse, which never recovered.
COLLAPSE = [(86, 0.01346), (89, 0.01261), (92, 0.99980), (95, 0.99974),
            (98, 0.99974), (100, 0.99974)]
# The healthy HEALPix arm (job 6160937) for contrast.
HEALTHY = [(178, 0.01099), (197, 0.01092), (198, 0.01087), (199, 0.01095),
           (200, 0.01087)]


class DivergenceGuardTests(unittest.TestCase):
    def _run(self, guard, trace):
        for epoch, val in trace:
            reason = guard.update(val, epoch)
            if reason:
                return epoch, reason
        return None, None

    def test_survives_the_real_transient_spike(self):
        """Epoch 10's 6.4x spike recovered; killing there would be a false kill."""
        g = DivergenceGuard()
        epoch, reason = self._run(g, EARLY_SPIKE)
        self.assertIsNone(epoch, f"false positive at epoch {epoch}: {reason}")

    def test_catches_the_real_collapse(self):
        g = DivergenceGuard()
        self._run(g, EARLY_SPIKE)
        epoch, reason = self._run(g, COLLAPSE)
        self.assertIsNotNone(epoch, "the guard missed the collapse it exists for")
        self.assertLessEqual(epoch, 98, "caught too late to save meaningful cost")
        self.assertIn("consecutive", reason)

    def test_quiet_on_a_healthy_run(self):
        g = DivergenceGuard()
        self.assertIsNone(self._run(g, HEALTHY)[0])

    def test_patience_requires_consecutive_epochs(self):
        """Alternating bad/good epochs must not accumulate strikes."""
        g = DivergenceGuard(factor=10.0, patience=3, min_epochs=0)
        g.update(0.01, 1)
        for e in range(2, 12, 2):
            self.assertIsNone(g.update(5.0, e))      # bad
            self.assertIsNone(g.update(0.01, e + 1))  # recovered -> reset

    def test_early_epochs_are_exempt(self):
        """Before min_epochs 'best so far' is not yet meaningful."""
        g = DivergenceGuard(min_epochs=5, patience=1)
        for e in range(1, 5):
            self.assertIsNone(g.update(50.0, e))

    def test_non_finite_loss_aborts(self):
        g = DivergenceGuard(patience=2, min_epochs=0)
        g.update(0.01, 1)
        self.assertIsNone(g.update(float("nan"), 2))
        self.assertIsNotNone(g.update(float("inf"), 3))

    def test_best_tracks_the_minimum_not_the_latest(self):
        g = DivergenceGuard(factor=10.0, patience=1, min_epochs=0)
        g.update(0.010, 1)
        g.update(0.020, 2)          # worse, but nowhere near 10x -> best stays 0.010
        self.assertAlmostEqual(g.best, 0.010)
        self.assertIsNotNone(g.update(0.11, 3))   # 11x the best

    def test_build_from_config(self):
        self.assertIsNone(build_divergence_guard({"divergence": {"enabled": False}}))
        g = build_divergence_guard({"divergence": {"factor": 4.0, "patience": 2,
                                                   "min_epochs": 1}})
        self.assertEqual((g.factor, g.patience, g.min_epochs), (4.0, 2, 1))
        self.assertIsInstance(build_divergence_guard({}), DivergenceGuard)  # on by default

    def test_cost_saved_on_the_real_run(self):
        """Sanity-check the guard would actually have paid for itself."""
        g = DivergenceGuard()
        self._run(g, EARLY_SPIKE)
        epoch, _ = self._run(g, COLLAPSE)
        saved = 200 - epoch
        self.assertGreater(saved, 100,
                           "should save >100 of the 200 epochs on job 6169725")


if __name__ == "__main__":
    unittest.main()
