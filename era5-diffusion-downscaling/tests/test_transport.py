"""CPU unit tests for transport paths, losses, and samplers."""

import unittest

import torch
import torch.nn as nn

from data.degrade import coarsen, degrade
from models.transport import (FlowMatching, StochasticInterpolant,
                              build_transport_model)


def tiny_config():
    return {
        "patches": {"size": 16},
        "unet": {
            "in_channels": 1, "out_channels": 1, "base_channels": 8,
            "channel_mults": [1, 2], "num_res_blocks": 1,
            "time_emb_dim": 16, "attn_resolutions": [8], "dropout": 0.0,
            "groupnorm_groups": 4,
        },
        "geo": {"enabled": False},
        "transport": {"time_scale": 100.0, "time_epsilon": 1e-3},
    }


class TransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.randn(2, 1, 16, 16)
        self.low_res = degrade(self.target, 4)

    def test_flow_loss_and_gradient(self):
        model = build_transport_model(tiny_config(), "flow")
        loss = FlowMatching().training_loss(model, self.target, self.low_res)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.parameters()))

    def test_stochastic_interpolant_heads_and_gradient(self):
        model = build_transport_model(tiny_config(), "stochastic_interpolant")
        process = StochasticInterpolant(gamma=0.4)
        loss, _, _, details = process.training_loss(
            model, self.target, self.low_res, return_details=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(details), {"velocity", "score"})
        self.assertEqual(model(self.target, torch.ones(2), (self.low_res, None)).shape,
                         (2, 2, 16, 16))

    def test_interpolant_endpoint(self):
        process = StochasticInterpolant(gamma=0.5)
        xt, _, _ = process.path(self.target, torch.ones(2))
        self.assertTrue(torch.allclose(xt, self.target, atol=1e-6))

    def test_flow_sample_has_exact_final_data_consistency(self):
        class ZeroField(nn.Module):
            def forward(self, x, t, cond):
                return torch.zeros_like(x)

        coarse = coarsen(self.target, 4)
        sample = FlowMatching().sample(
            ZeroField(), self.low_res, steps=2, solver="heun",
            project="final", coarse=coarse, ratio=4,
        )
        self.assertTrue(torch.allclose(coarsen(sample, 4), coarse, atol=1e-6))

    def test_stochastic_ode_and_sde_shapes(self):
        class ZeroTwoHead(nn.Module):
            def forward(self, x, t, cond):
                return torch.cat([torch.zeros_like(x), torch.zeros_like(x)], dim=1)

        process = StochasticInterpolant()
        model = ZeroTwoHead()
        ode = process.sample(model, self.low_res, steps=2, sampler="ode")
        sde = process.sample(model, self.low_res, steps=2, sampler="sde")
        self.assertEqual(ode.shape, self.target.shape)
        self.assertEqual(sde.shape, self.target.shape)
        self.assertTrue(torch.isfinite(sde).all())


if __name__ == "__main__":
    unittest.main()
