"""CPU unit tests for the multi-channel (20-variable) pipeline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.dataset import Normalizer
from data.degrade import coarsen, degrade
from data.make_patches import crop_patches, crop_patches_to_disk
from models.residual import build_residual_model
from models.transport import FlowMatching, build_transport_model
from utils import channel_label, channel_labels, channel_specs, display_channel


def tiny_config(channels):
    return {
        "patches": {"size": 16},
        "unet": {
            "in_channels": channels, "out_channels": channels, "base_channels": 8,
            "channel_mults": [1, 2], "num_res_blocks": 1,
            "time_emb_dim": 16, "attn_resolutions": [8], "dropout": 0.0,
            "groupnorm_groups": 4,
        },
        "geo": {"enabled": False},
        "transport": {"time_scale": 100.0, "time_epsilon": 1e-3},
    }


class ChannelHelperTests(unittest.TestCase):
    def test_multi_channel_specs_and_labels(self):
        dcfg = {"variables": [
            {"name": "2m_temperature", "level": None},
            {"name": "geopotential", "level": 500},
            {"name": "specific_humidity", "level": 700},
        ]}
        self.assertEqual(channel_specs(dcfg), [
            {"name": "2m_temperature", "level": None},
            {"name": "geopotential", "level": 500},
            {"name": "specific_humidity", "level": 700},
        ])
        self.assertEqual(channel_labels(dcfg), ["t2m", "z500", "q700"])

    def test_legacy_single_variable(self):
        self.assertEqual(channel_specs({"variable": "2m_temperature", "level": None}),
                         [{"name": "2m_temperature", "level": None}])
        self.assertEqual(channel_label("geopotential", 500), "z500")
        self.assertEqual(channel_labels({"variable": "geopotential", "level": 500}),
                         ["z500"])

    def test_display_channel_default_and_config(self):
        self.assertEqual(display_channel({}), 0)
        self.assertEqual(display_channel({"eval": {"display_channel": 3}}), 3)


class NormalizerTests(unittest.TestCase):
    def test_per_channel_roundtrip(self):
        mean, std = np.array([1.0, -2.0, 30.0]), np.array([0.5, 4.0, 10.0])
        norm = Normalizer(mean, std)
        x = torch.randn(4, 3, 8, 8)
        self.assertTrue(torch.allclose(norm.decode(norm.encode(x)), x, atol=1e-5))
        # Channels are normalized independently.
        enc = norm.encode(torch.zeros(1, 3, 2, 2))
        expected = torch.tensor([-2.0, 0.5, -3.0]).view(1, 3, 1, 1).expand(1, 3, 2, 2)
        self.assertTrue(torch.allclose(enc, expected, atol=1e-6))

    def test_scalar_legacy_broadcasts(self):
        norm = Normalizer(5.0, 2.0)
        x = torch.randn(2, 4, 8, 8)  # any channel count
        self.assertTrue(torch.allclose(norm.decode(norm.encode(x)), x, atol=1e-5))

    def test_zero_std_guard(self):
        norm = Normalizer(np.zeros(2), np.array([0.0, 1.0]))
        self.assertTrue(torch.isfinite(norm.encode(torch.randn(1, 2, 4, 4))).all())


class CropPatchesTests(unittest.TestCase):
    def test_multi_channel_crops(self):
        rng = np.random.default_rng(0)
        fields = rng.normal(size=(3, 5, 32, 48)).astype(np.float32)
        patches, origins = crop_patches(fields, size=16, per_field=2, rng=rng)
        self.assertEqual(patches.shape, (6, 5, 16, 16))
        self.assertEqual(origins.shape, (6, 2))

    def test_legacy_3dim_promoted(self):
        rng = np.random.default_rng(0)
        fields = rng.normal(size=(3, 32, 48)).astype(np.float32)
        patches, _ = crop_patches(fields, size=16, per_field=2, rng=rng)
        self.assertEqual(patches.shape, (6, 1, 16, 16))

    def test_to_disk_shapes_and_per_channel_stats(self):
        rng = np.random.default_rng(0)
        fields = np.stack([  # channels with very different scales
            rng.normal(300.0, 5.0, size=(4, 32, 48)),
            rng.normal(0.0, 1.0, size=(4, 32, 48)),
            rng.normal(-50.0, 20.0, size=(4, 32, 48)),
        ], axis=1).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            np.save(tmp / "raw.npy", fields)
            shape, mean, std = crop_patches_to_disk(
                tmp / "raw.npy", size=16, per_field=3,
                rng=np.random.default_rng(1), out_path=tmp / "patches.npy",
                origins_path=tmp / "origins.npy", shuffle=True, desc="test")
            self.assertEqual(shape, (12, 3, 16, 16))
            patches = np.load(tmp / "patches.npy")
            origins = np.load(tmp / "origins.npy")
            self.assertEqual(patches.shape, (12, 3, 16, 16))
            self.assertEqual(origins.shape, (12, 2))
            # Streamed stats must match direct per-channel stats of the output.
            direct_mean = patches.mean(axis=(0, 2, 3))
            direct_std = patches.std(axis=(0, 2, 3))
            np.testing.assert_allclose(mean, direct_mean, rtol=1e-4)
            np.testing.assert_allclose(std, direct_std, rtol=1e-3)
            # Shuffle keeps patches aligned with their recorded origins: every
            # stored patch must be an exact crop of SOME field at its origin.
            for j in range(len(patches)):
                r, c = origins[j]
                match = any(
                    np.array_equal(patches[j], fields[f, :, r:r + 16, c:c + 16])
                    for f in range(fields.shape[0]))
                self.assertTrue(match, f"patch {j} does not match any field crop")


class MultiChannelModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_transport_model_20_channels(self):
        cfg = tiny_config(20)
        model = build_transport_model(cfg, "flow")
        x = torch.randn(2, 20, 16, 16)
        low_res = degrade(x, 4)
        out = model(x, torch.rand(2), (low_res, None))
        self.assertEqual(out.shape, (2, 20, 16, 16))

    def test_stochastic_interpolant_two_heads(self):
        cfg = tiny_config(20)
        model = build_transport_model(cfg, "stochastic_interpolant")
        x = torch.randn(2, 20, 16, 16)
        out = model(x, torch.rand(2), (degrade(x, 4), None))
        self.assertEqual(out.shape, (2, 40, 16, 16))

    def test_flow_loss_multi_channel(self):
        cfg = tiny_config(3)
        model = build_transport_model(cfg, "flow")
        target = torch.randn(2, 3, 16, 16)
        loss = FlowMatching().training_loss(model, target, degrade(target, 4))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_flow_sample_multi_channel_data_consistency(self):
        cfg = tiny_config(3)
        model = build_transport_model(cfg, "flow")
        target = torch.randn(2, 3, 16, 16)
        low_res = degrade(target, 4)
        coarse = coarsen(target, 4)
        sample = FlowMatching().sample(model, low_res, steps=2, solver="euler",
                                       project="final", coarse=coarse, ratio=4)
        self.assertEqual(sample.shape, target.shape)
        self.assertTrue(torch.allclose(coarsen(sample, 4), coarse, atol=1e-5))

    def test_residual_model_multi_channel(self):
        cfg = {**tiny_config(3), "diffusion": {}}
        model = build_residual_model(cfg)
        x_t = torch.randn(2, 3, 16, 16)
        mean_field = torch.randn(2, 3, 16, 16)
        out = model(x_t, torch.ones(2), (mean_field, None))
        self.assertEqual(out.shape, x_t.shape)


if __name__ == "__main__":
    unittest.main()
