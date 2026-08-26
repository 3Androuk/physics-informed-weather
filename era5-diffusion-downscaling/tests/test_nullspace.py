"""CPU tests for the null-space residual transport (--null-space / --consistent-mean).

The two theorems the design rests on are tested directly:
  1. Pythagoras — projecting the mean onto {x: Ax=y} cannot increase its error
     to any target satisfying the constraint.
  2. Structural consistency — with a P-projected source and P-projected
     velocities, A R(t) = 0 along the whole trajectory for ANY network, so
     coarsen(mean + residual) == observation without any corrective step.
"""

import unittest

import torch

from data.degrade import coarsen
from models.transport import (FlowMatching, StochasticInterpolant,
                              build_transport_model, integrate_transport,
                              nullspace_project, project_data_consistency)


def tiny_cfg():
    return {
        "patches": {"size": 16},
        "unet": {"in_channels": 1, "out_channels": 1, "base_channels": 8,
                 "channel_mults": [1, 2], "num_res_blocks": 1,
                 "time_emb_dim": 16, "attn_resolutions": [8], "dropout": 0.0,
                 "groupnorm_groups": 4},
        "geo": {"enabled": False},
        "transport": {"time_scale": 100.0, "time_epsilon": 1e-3},
    }


class ProjectorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(4, 1, 16, 16)
        self.ratio = 4

    def test_projection_lands_in_ker_A(self):
        p = nullspace_project(self.x, self.ratio)
        self.assertLess(float(coarsen(p, self.ratio).abs().max()), 1e-6)

    def test_idempotent(self):
        p = nullspace_project(self.x, self.ratio)
        pp = nullspace_project(p, self.ratio)
        self.assertTrue(torch.allclose(p, pp, atol=1e-6))

    def test_identity_on_ker_A(self):
        p = nullspace_project(self.x, self.ratio)
        self.assertTrue(torch.allclose(nullspace_project(p, self.ratio), p,
                                       atol=1e-6))

    def test_orthogonal_decomposition(self):
        # x = Px + A†Ax with the two parts orthogonal.
        p = nullspace_project(self.x, self.ratio)
        r = self.x - p
        self.assertLess(abs(float((p * r).sum())), 1e-3)


class ConsistentMeanTests(unittest.TestCase):
    def test_projected_mean_is_exactly_consistent(self):
        torch.manual_seed(1)
        truth = torch.randn(4, 1, 16, 16)
        y = coarsen(truth, 4)
        mu = truth + 0.3 * torch.randn_like(truth)  # imperfect regression
        m = project_data_consistency(mu, y, 4)
        self.assertLess(float((coarsen(m, 4) - y).abs().max()), 1e-6)

    def test_pythagoras_never_worse(self):
        # ||x - m||^2 = ||x - mu||^2 - ||m - mu||^2 for any x with Ax = y.
        torch.manual_seed(2)
        for _ in range(5):
            truth = torch.randn(2, 1, 16, 16)
            y = coarsen(truth, 4)
            mu = truth + torch.randn_like(truth)
            m = project_data_consistency(mu, y, 4)
            before = float((truth - mu).square().sum())
            after = float((truth - m).square().sum())
            moved = float((m - mu).square().sum())
            self.assertLessEqual(after, before + 1e-4)
            self.assertAlmostEqual(after + moved, before, delta=1e-2)

    def test_residual_target_lies_in_ker_A(self):
        torch.manual_seed(3)
        truth = torch.randn(2, 1, 16, 16)
        y = coarsen(truth, 4)
        m = project_data_consistency(torch.randn_like(truth), y, 4)
        residual = truth - m
        self.assertLess(float(coarsen(residual, 4).abs().max()), 1e-6)


class StructuralConsistencyTests(unittest.TestCase):
    """The headline theorem: exact consistency for an UNTRAINED network."""

    def setUp(self):
        torch.manual_seed(4)
        self.ratio = 4
        self.mean = torch.randn(2, 1, 16, 16)

    def test_flow_trajectory_stays_in_ker_A(self):
        model = build_transport_model(tiny_cfg(), "flow")  # random weights
        out = integrate_transport(model, self.mean, steps=8, solver="heun",
                                  null_ratio=self.ratio)
        self.assertLess(float(coarsen(out, self.ratio).abs().max()), 1e-5)

    def test_si_ode_and_sde_stay_in_ker_A(self):
        model = build_transport_model(tiny_cfg(), "stochastic_interpolant")
        process = StochasticInterpolant()
        for sampler in ("ode", "sde"):
            out = process.sample(model, self.mean, steps=8, sampler=sampler,
                                 null_ratio=self.ratio)
            self.assertLess(float(coarsen(out, self.ratio).abs().max()), 1e-5,
                            sampler)

    def test_composite_is_exactly_consistent(self):
        # mean consistent + residual in ker A => coarsen(mean + s*R) == y,
        # with a random untrained network and no corrective projection.
        truth = torch.randn(2, 1, 16, 16)
        y = coarsen(truth, self.ratio)
        m = project_data_consistency(torch.randn_like(truth), y, self.ratio)
        model = build_transport_model(tiny_cfg(), "flow")
        r = integrate_transport(model, m, steps=6, null_ratio=self.ratio)
        composite = m + 0.37 * r
        self.assertLess(float((coarsen(composite, self.ratio) - y).abs().max()),
                        1e-5)

    def test_euler_solver_too(self):
        model = build_transport_model(tiny_cfg(), "flow")
        out = integrate_transport(model, self.mean, steps=8, solver="euler",
                                  null_ratio=self.ratio)
        self.assertLess(float(coarsen(out, self.ratio).abs().max()), 1e-5)


class NullspaceTrainingTests(unittest.TestCase):
    def test_flow_loss_finite_and_backprops(self):
        torch.manual_seed(5)
        model = build_transport_model(tiny_cfg(), "flow")
        target = nullspace_project(torch.randn(2, 1, 16, 16), 4)
        loss = FlowMatching().training_loss(model, target,
                                            torch.randn(2, 1, 16, 16),
                                            null_ratio=4)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_si_loss_finite(self):
        torch.manual_seed(6)
        model = build_transport_model(tiny_cfg(), "stochastic_interpolant")
        target = nullspace_project(torch.randn(2, 1, 16, 16), 4)
        loss = StochasticInterpolant().training_loss(
            model, target, torch.randn(2, 1, 16, 16), null_ratio=4)
        self.assertTrue(torch.isfinite(loss))

    def test_si_path_stays_in_ker_A(self):
        torch.manual_seed(7)
        target = nullspace_project(torch.randn(4, 1, 16, 16), 4)
        t = torch.rand(4)
        xt, velocity, scaled_score = StochasticInterpolant().path(
            target, t, null_ratio=4)
        for name, field in (("xt", xt), ("velocity", velocity),
                            ("score", scaled_score)):
            self.assertLess(float(coarsen(field, 4).abs().max()), 1e-5, name)


class StemTests(unittest.TestCase):
    def test_checkpoint_stems(self):
        from train.train_transport import _checkpoint_stem
        cfg = {"geo": {"enabled": False}, "seed": 42}
        self.assertEqual(_checkpoint_stem("flow", cfg, False, True, False,
                                          consistent_mean=True),
                         "flow_matching_res_cm")
        self.assertEqual(_checkpoint_stem("flow", cfg, False, True, False,
                                          consistent_mean=True, null_space=True),
                         "flow_matching_res_ns")
        self.assertEqual(_checkpoint_stem("stochastic_interpolant", cfg, False,
                                          True, True, null_space=True),
                         "stochastic_interpolant_res_ns_lm")


if __name__ == "__main__":
    unittest.main()
