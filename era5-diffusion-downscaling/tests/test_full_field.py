"""CPU unit tests for full-field tiled reconstruction and SDPA attention."""

import unittest

import torch
import torch.nn as nn

from data.degrade import coarsen, degrade
from models.diffusion import build_diffusion
from models.transport import FlowMatching, StochasticInterpolant, build_transport_model
from models.unet import AttnBlock, build_unet
from sample.full_field import (blend_window, crop_geo_tiles, crop_tiles,
                               crop_to_multiple,
                               reconstruct_full_tiled_diffusion,
                               reconstruct_full_tiled_directmap,
                               reconstruct_full_tiled_transport, stitch_tiles,
                               tile_origins)


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
        "diffusion": {"timesteps": 8, "beta_schedule": "linear",
                      "beta_start": 1e-4, "beta_end": 2e-2},
        "transport": {"time_scale": 100.0, "time_epsilon": 1e-3},
    }


class TileMathTests(unittest.TestCase):
    def test_origins_cover_and_align(self):
        starts = tile_origins(96, 32, 24, align=8)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 96 - 32)          # right-aligned last tile
        self.assertTrue(all(s % 8 == 0 for s in starts))
        covered = set()
        for s in starts:
            covered.update(range(s, s + 32))
        self.assertEqual(covered, set(range(96)))      # full coverage

    def test_origins_reject_misaligned(self):
        with self.assertRaises(ValueError):
            tile_origins(100, 32, 24, align=8)         # 100 % 8 != 0
        with self.assertRaises(ValueError):
            tile_origins(16, 32, 24, align=8)          # tile > length

    def test_crop_to_multiple(self):
        x = torch.randn(1, 1, 481, 1440)
        self.assertEqual(crop_to_multiple(x, 16).shape, (1, 1, 480, 1440))

    def test_blend_window(self):
        win = blend_window(32, 8)
        self.assertEqual(win.shape, (1, 1, 32, 32))
        self.assertTrue((win > 0).all() and (win <= 1).all())
        self.assertTrue(torch.allclose(win[0, 0, 16, 16], torch.tensor(1.0)))

    def test_stitch_exact_when_tiles_agree(self):
        """Blending is exact wherever tiles agree — the window normalization
        cannot distort a consistent reconstruction (incl. shared noise)."""
        full = torch.randn(1, 2, 48, 64)
        out = stitch_tiles(lambda origins: crop_tiles(full, origins, 16),
                           full, tile=16, overlap=8, align=4, batch=3)
        self.assertTrue(torch.allclose(out, full, atol=1e-6))

    def test_crop_geo_tiles_shapes(self):
        origins = [(0, 0), (4, 8)]
        hash_coords = torch.randn(32, 48, 3)
        self.assertEqual(crop_geo_tiles(hash_coords, origins, 16).shape, (2, 16, 16, 3))
        hpx = torch.randn(5, 32, 48, 8)
        self.assertEqual(crop_geo_tiles(hpx, origins, 16).shape, (2, 5, 16, 16, 8))
        self.assertIsNone(crop_geo_tiles(None, origins, 16))


class TiledReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_config()
        self.hf = torch.randn(1, 1, 32, 48)
        self.ratio = 4
        self.lf = degrade(self.hf, self.ratio)
        self.coarse = coarsen(self.hf, self.ratio)

    def test_tiled_transport_shape_and_consistency(self):
        model = build_transport_model(self.cfg, "flow")
        out = reconstruct_full_tiled_transport(
            model, FlowMatching(), self.lf, self.coarse, self.ratio, self.cfg,
            "flow", tile=16, overlap=8, batch=4, steps=2, solver="euler",
            generator=torch.Generator().manual_seed(0))
        self.assertEqual(out.shape, self.hf.shape)
        self.assertTrue(torch.isfinite(out).all())
        # global projection: stitched field coarsens back to the observation
        self.assertTrue(torch.allclose(coarsen(out, self.ratio), self.coarse, atol=1e-5))

    def test_tiled_transport_deterministic_given_seed(self):
        model = build_transport_model(self.cfg, "flow")
        outs = [reconstruct_full_tiled_transport(
            model, FlowMatching(), self.lf, self.coarse, self.ratio, self.cfg,
            "flow", tile=16, overlap=8, batch=4, steps=2, solver="euler",
            generator=torch.Generator().manual_seed(7)) for _ in range(2)]
        self.assertTrue(torch.allclose(outs[0], outs[1]))

    def test_tiled_si_sde(self):
        model = build_transport_model(self.cfg, "stochastic_interpolant")
        out = reconstruct_full_tiled_transport(
            model, StochasticInterpolant(), self.lf, self.coarse, self.ratio,
            self.cfg, "stochastic_interpolant", tile=16, overlap=8, batch=4,
            steps=2, sampler="sde", generator=torch.Generator().manual_seed(0))
        self.assertEqual(out.shape, self.hf.shape)
        self.assertTrue(torch.allclose(coarsen(out, self.ratio), self.coarse, atol=1e-5))

    def test_tiled_transport_project_each(self):
        model = build_transport_model(self.cfg, "flow")
        out = reconstruct_full_tiled_transport(
            model, FlowMatching(), self.lf, self.coarse, self.ratio, self.cfg,
            "flow", tile=16, overlap=8, batch=4, steps=2, solver="euler",
            project_each=True, generator=torch.Generator().manual_seed(0))
        self.assertEqual(out.shape, self.hf.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(coarsen(out, self.ratio), self.coarse, atol=1e-5))

    def test_tiled_diffusion(self):
        model = build_unet(self.cfg, use_time=True)
        diffusion = build_diffusion(self.cfg)
        rc = {"K": 2, "t_steps": [3, 5]}
        out = reconstruct_full_tiled_diffusion(
            diffusion, model, self.lf, self.coarse, self.ratio, rc, tile=16,
            overlap=8, batch=4, project_steps=True,
            generator=torch.Generator().manual_seed(0))
        self.assertEqual(out.shape, self.hf.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(coarsen(out, self.ratio), self.coarse, atol=1e-5))

    def test_tiled_directmap(self):
        model = build_unet(self.cfg, use_time=False)
        out = reconstruct_full_tiled_directmap(model, self.lf, tile=16,
                                               overlap=8, batch=4)
        self.assertEqual(out.shape, self.hf.shape)
        self.assertTrue(torch.isfinite(out).all())


class SharedNoiseTests(unittest.TestCase):
    def test_transport_sampler_accepts_noise(self):
        """With a zero velocity field, the ODE output IS the initial noise —
        supplying it explicitly must round-trip exactly."""
        class ZeroField(nn.Module):
            def forward(self, x, t, cond):
                return torch.zeros_like(x)

        low = torch.randn(2, 1, 16, 16)
        noise = torch.randn_like(low)
        out = FlowMatching().sample(ZeroField(), low, steps=2, solver="euler",
                                    noise=noise)
        self.assertTrue(torch.allclose(out, noise))
        with self.assertRaises(ValueError):
            FlowMatching().sample(ZeroField(), low, steps=2, noise=noise[:1])

    def test_diffusion_accepts_init_noise(self):
        cfg = tiny_config()
        model = build_unet(cfg, use_time=True)
        diffusion = build_diffusion(cfg)
        x_g = torch.randn(2, 1, 16, 16)
        eps = torch.randn(2, 2, 1, 16, 16)   # (K, N, C, H, W)
        out = diffusion.guided_reconstruct(model, x_g, t_steps=[3, 5], K=2,
                                           init_noise=eps)
        self.assertEqual(out.shape, x_g.shape)
        with self.assertRaises(AssertionError):
            diffusion.guided_reconstruct(model, x_g, t_steps=[3, 5], K=2,
                                         init_noise=eps[:1])


class AttentionEquivalenceTests(unittest.TestCase):
    def test_sdpa_matches_explicit_softmax(self):
        torch.manual_seed(0)
        block = AttnBlock(16, 4).eval()
        x = torch.randn(2, 16, 8, 8)
        with torch.no_grad():
            out = block(x)
            # reference: the explicit (HW, HW) computation with the same weights
            n, c, h, w = x.shape
            q, k, v = block.qkv(block.norm(x)).chunk(3, dim=1)
            q = q.reshape(n, c, h * w).permute(0, 2, 1)
            k = k.reshape(n, c, h * w)
            v = v.reshape(n, c, h * w).permute(0, 2, 1)
            attn = torch.softmax(torch.bmm(q, k) * block.scale, dim=-1)
            ref = x + block.proj(torch.bmm(attn, v).permute(0, 2, 1).reshape(n, c, h, w))
        self.assertTrue(torch.allclose(out, ref, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
