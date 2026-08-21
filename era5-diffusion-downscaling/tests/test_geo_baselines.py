"""CPU unit tests for the geo-conditioning baseline encoders.

The comparison ladder for the learned location tables (hash/HEALPix):
  xyz        raw unit-sphere coordinates (trivial baseline)
  sinusoidal fixed multiscale Fourier basis (engineered-basis baseline)
  static     real physiographic fields (literature-standard strong null)
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.dataset import Normalizer, PatchDataset
from models.geo_encoding import (RawCoords, SinusoidalSphere, StaticFields,
                                 build_geo_encoder)
from models.transport import build_transport_model
from utils import geo_suffix


def geo_cfg(encoder, **extra):
    g = {"enabled": True, "encoder": encoder, "input_dim": 3, "altitude": None,
         "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 10,
         "base_resolution": 4, "finest_resolution": 16,
         "healpix_n_levels": 3, "healpix_nside_max": 4,
         "sinusoidal_n_frequencies": 2,
         "static_fields": ["geopotential_at_surface", "land_sea_mask"]}
    g.update(extra)
    return {
        "patches": {"size": 16},
        "unet": {"in_channels": 1, "out_channels": 1, "base_channels": 8,
                 "channel_mults": [1, 2], "num_res_blocks": 1,
                 "time_emb_dim": 16, "attn_resolutions": [8], "dropout": 0.0,
                 "groupnorm_groups": 4},
        "geo": g,
        "transport": {"time_scale": 100.0, "time_epsilon": 1e-3},
    }


class EncoderTests(unittest.TestCase):
    def test_raw_coords_is_identity(self):
        enc = RawCoords(input_dim=3)
        coords = torch.rand(2, 8, 8, 3)
        self.assertEqual(enc.output_dim, 3)
        self.assertTrue(torch.equal(enc(coords), coords))
        self.assertEqual(sum(p.numel() for p in enc.parameters()), 0)

    def test_sinusoidal_shape_determinism_no_params(self):
        enc = SinusoidalSphere(input_dim=3, n_frequencies=2)
        coords = torch.rand(2, 8, 8, 3)
        out = enc(coords)
        self.assertEqual(enc.output_dim, 12)  # 3 * 2 * 2
        self.assertEqual(out.shape, (2, 8, 8, 12))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(out.abs().max() <= 1.0 + 1e-6)
        self.assertTrue(torch.equal(out, enc(coords)))  # deterministic
        self.assertEqual(sum(p.numel() for p in enc.parameters()), 0)

    def test_sinusoidal_distinguishes_locations(self):
        enc = SinusoidalSphere(input_dim=3, n_frequencies=2)
        a = enc(torch.tensor([[0.1, 0.5, 0.9]]))
        b = enc(torch.tensor([[0.6, 0.2, 0.3]]))
        self.assertFalse(torch.allclose(a, b))

    def test_static_fields_is_identity(self):
        enc = StaticFields(n_fields=2)
        payload = torch.rand(2, 8, 8, 2)
        self.assertEqual(enc.output_dim, 2)
        self.assertTrue(torch.equal(enc(payload), payload))

    def test_build_geo_encoder_dispatch(self):
        expected_dims = {"hash": 8, "healpix": 6, "xyz": 3,
                         "sinusoidal": 12, "static": 2, "hash_static": 10}
        for encoder, dim in expected_dims.items():
            enc = build_geo_encoder(geo_cfg(encoder))
            self.assertEqual(enc.output_dim, dim, encoder)
        with self.assertRaises(ValueError):
            build_geo_encoder(geo_cfg("nope"))

    def test_hash_static_combo_splits_payload(self):
        # First d channels feed the hash grid; the trailing S static channels
        # pass through untouched into the tail of the embedding.
        enc = build_geo_encoder(geo_cfg("hash_static"))
        payload = torch.rand(2, 8, 8, 5)   # d=3 coords | S=2 static
        out = enc(payload)
        self.assertEqual(out.shape, (2, 8, 8, 10))
        self.assertTrue(torch.equal(out[..., -2:], payload[..., 3:]))


class LadderTests(unittest.TestCase):
    def test_pow2_ladder_backward_compatible(self):
        from models.geo_encoding import healpix_nside_ladder
        self.assertEqual(healpix_nside_ladder(8, 1, 128),
                         [1, 2, 4, 8, 16, 32, 64, 128])

    def test_integer_ring_ladder_matches_hash_band(self):
        from models.geo_encoding import healpix_nside_ladder
        self.assertEqual(healpix_nside_ladder(8, 8, 64),
                         [8, 11, 14, 20, 26, 35, 48, 64])

    def test_non_increasing_ladder_rejected(self):
        from models.geo_encoding import healpix_nside_ladder
        with self.assertRaises(AssertionError):
            healpix_nside_ladder(16, 8, 16)  # too many levels for the range


class SuffixTests(unittest.TestCase):
    def test_geo_suffix(self):
        self.assertEqual(geo_suffix({"geo": {"enabled": False}}), "")
        self.assertEqual(geo_suffix({}), "")
        for encoder, tag in (("hash", "_geo"), ("healpix", "_geo_hpx"),
                             ("xyz", "_geo_xyz"), ("sinusoidal", "_geo_sin"),
                             ("static", "_geo_static"),
                             ("hash_static", "_geo_combo")):
            self.assertEqual(geo_suffix({"geo": {"enabled": True,
                                                 "encoder": encoder}}), tag)
        with self.assertRaises(ValueError):
            geo_suffix({"geo": {"enabled": True, "encoder": "nope"}})

    def test_checkpointed_embed_is_exact(self):
        # Recompute-in-backward must be bitwise identical in output and
        # numerically identical in the table gradients.
        from models.geo_encoding import checkpointed_embed
        torch.manual_seed(0)
        enc = build_geo_encoder(geo_cfg("hash"))
        coords = torch.rand(2, 8, 8, 3)
        plain = enc(coords)
        plain.square().sum().backward()
        g_plain = [p.grad.clone() for p in enc.parameters()]
        enc.zero_grad()
        ck = checkpointed_embed(enc, coords)
        self.assertTrue(torch.equal(ck, plain))
        ck.square().sum().backward()
        for gp, p in zip(g_plain, enc.parameters()):
            self.assertTrue(torch.allclose(gp, p.grad))
        # parameter-free encoders bypass the checkpoint machinery
        static = build_geo_encoder(geo_cfg("static"))
        payload = torch.rand(2, 8, 8, 2)
        self.assertTrue(torch.equal(checkpointed_embed(static, payload),
                                    static(payload)))

    def test_geo_suffix_gated(self):
        self.assertEqual(geo_suffix({"geo": {"enabled": True, "encoder": "hash",
                                             "level_gating": True}}),
                         "_geo_gated")
        self.assertEqual(geo_suffix({"geo": {"enabled": True,
                                             "encoder": "hash_static",
                                             "level_gating": True}}),
                         "_geo_combo_gated")


class LevelGateTests(unittest.TestCase):
    """Noise-dependent gating of leveled embeddings (--gated)."""

    def test_build_level_gate(self):
        from models.geo_encoding import build_level_gate
        self.assertIsNone(build_level_gate(geo_cfg("hash")))
        for encoder in ("hash", "healpix", "hash_static"):
            gate = build_level_gate(geo_cfg(encoder, level_gating=True))
            self.assertIsNotNone(gate, encoder)
            self.assertEqual(sum(p.numel() for p in gate.parameters()), 0)
        for encoder in ("xyz", "sinusoidal", "static"):
            with self.assertRaises(ValueError):
                build_level_gate(geo_cfg(encoder, level_gating=True))

    def test_gates_open_monotonically_with_signal(self):
        from models.geo_encoding import LevelGate
        gate = LevelGate(n_levels=4, n_features_per_level=2)
        emb = torch.ones(3, 4, 4, 8)
        # u=1 (clean data): every level essentially open
        out_clean = gate(emb, torch.ones(3))
        self.assertTrue((out_clean > 0.97).all())
        # mid-denoising: coarse levels more open than fine levels
        out_mid = gate(emb, torch.full((3,), 0.3))
        per_level = out_mid[0, 0, 0].view(4, 2)[:, 0]
        self.assertTrue((per_level[:-1] >= per_level[1:]).all())
        self.assertGreater(per_level[0], 0.9)   # coarsest: open
        self.assertLess(per_level[-1], 0.1)     # finest: shut at u=0.3
        # u=0 (pure noise): only the c_0=0 level is half open
        out_noise = gate(emb, torch.zeros(3))
        self.assertLess(out_noise[..., 2:].max(), 0.5)

    def test_static_tail_passes_ungated(self):
        from models.geo_encoding import LevelGate
        # hash_static: gate the first L*F channels, leave the S-field tail
        gate = LevelGate(n_levels=4, n_features_per_level=2, gated_dim=8)
        emb = torch.rand(2, 4, 4, 10)  # 8 hash + 2 static
        out = gate(emb, torch.zeros(2))
        self.assertTrue(torch.equal(out[..., 8:], emb[..., 8:]))
        self.assertFalse(torch.equal(out[..., :8], emb[..., :8]))

    def test_gated_models_forward(self):
        from models.geo_encoding import (GeoConditionedUNet, build_geo_encoder,
                                         build_level_gate)
        from models.unet import build_unet
        torch.manual_seed(0)
        # DDPM wrapper: t in [1, T], u = 1 - t/T
        cfg = geo_cfg("hash", level_gating=True)
        geo_enc = build_geo_encoder(cfg)
        base = build_unet(cfg, use_time=True,
                          extra_in_channels=geo_enc.output_dim)
        model = GeoConditionedUNet(base, geo_enc,
                                   level_gate=build_level_gate(cfg),
                                   ddpm_timesteps=100)
        x = torch.randn(2, 1, 16, 16)
        coords = torch.rand(2, 16, 16, 3)
        out = model(x, torch.tensor([1.0, 99.0]), coords)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        # transport: t IS the signal fraction; builder wires the gate itself
        tmodel = build_transport_model(cfg, "flow")
        self.assertIsNotNone(tmodel.gate)
        tout = tmodel(x, torch.rand(2), (torch.randn(2, 1, 16, 16), coords))
        self.assertEqual(tout.shape, x.shape)
        self.assertTrue(torch.isfinite(tout).all())

    def test_gated_residual_model_forward(self):
        from models.residual import build_residual_model
        torch.manual_seed(0)
        cfg = geo_cfg("hash", level_gating=True)
        cfg["diffusion"] = {"timesteps": 50, "beta_schedule": "cosine"}
        model = build_residual_model(cfg)
        self.assertIsNotNone(model.gate)
        self.assertEqual(model.ddpm_timesteps, 50)
        x = torch.randn(2, 1, 16, 16)
        mean_f = torch.randn(2, 1, 16, 16)
        coords = torch.rand(2, 16, 16, 3)
        out = model(x, torch.tensor([1.0, 49.0]), (mean_f, coords))
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())


class StaticDatasetTests(unittest.TestCase):
    def test_patch_dataset_static_payload(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            patches = rng.normal(size=(3, 1, 8, 8)).astype(np.float32)
            origins = np.array([[0, 0], [4, 8], [8, 16]], dtype=np.int32)
            np.save(tmp / "patches.npy", patches)
            np.save(tmp / "origins.npy", origins)
            np.savez(tmp / "coords_full.npz",
                     lat=np.linspace(-60, 60, 16), lon=np.linspace(0, 360, 24))
            static = rng.normal(size=(2, 16, 24)).astype(np.float32)
            np.savez(tmp / "static_fields.npz", fields=static,
                     names=np.array(["orog", "lsm"]),
                     mean=np.zeros(2), std=np.ones(2))

            ds = PatchDataset(tmp / "patches.npy", Normalizer(0.0, 1.0),
                              origins_path=tmp / "origins.npy",
                              coords_full_path=tmp / "coords_full.npz",
                              geo_encoder="static")
            x, payload = ds[1]
            self.assertEqual(tuple(payload.shape), (8, 8, 2))
            # channels-last crop matches the (S, H, W) source at the origin
            np.testing.assert_allclose(payload.numpy(),
                                       static[:, 4:12, 8:16].transpose(1, 2, 0))

            combo = PatchDataset(tmp / "patches.npy", Normalizer(0.0, 1.0),
                                 origins_path=tmp / "origins.npy",
                                 coords_full_path=tmp / "coords_full.npz",
                                 geo_encoder="hash_static")
            _, cp = combo[1]
            self.assertEqual(tuple(cp.shape), (8, 8, 5))  # 3 coords + 2 static
            np.testing.assert_allclose(cp[..., 3:].numpy(),
                                       static[:, 4:12, 8:16].transpose(1, 2, 0))
            self.assertTrue((cp[..., :3] >= 0).all() and (cp[..., :3] <= 1).all())
            del ds, combo  # release mmaps before the tempdir is removed


class ConditionedModelTests(unittest.TestCase):
    """Every encoder must plug into the conditioned models via the same
    extra-channels path."""

    def _payload(self, encoder, batch=2, size=16):
        if encoder == "static":
            return torch.rand(batch, size, size, 2)
        if encoder == "hash_static":
            return torch.rand(batch, size, size, 5)  # 3 coords + 2 static
        if encoder == "healpix":
            # per-level indices must stay inside each level's 12*Nside^2 cells
            from models.geo_encoding import healpix_nside_ladder
            nsides = healpix_nside_ladder(3, 1, 4)
            idx = torch.stack([
                torch.randint(0, 12 * ns * ns, (batch, size, size, 4)).float()
                for ns in nsides], dim=1)                     # (B, L, s, s, 4)
            w = torch.softmax(torch.rand(batch, 3, size, size, 4), dim=-1)
            return torch.cat([idx, w], dim=-1)
        return torch.rand(batch, size, size, 3)  # hash / xyz / sinusoidal

    def test_transport_model_with_each_encoder(self):
        torch.manual_seed(0)
        for encoder in ("hash", "healpix", "xyz", "sinusoidal", "static",
                        "hash_static"):
            cfg = geo_cfg(encoder)
            model = build_transport_model(cfg, "flow")
            x = torch.randn(2, 1, 16, 16)
            low_res = torch.randn(2, 1, 16, 16)
            out = model(x, torch.rand(2), (low_res, self._payload(encoder)))
            self.assertEqual(out.shape, x.shape, encoder)
            self.assertTrue(torch.isfinite(out).all(), encoder)

    def test_geo_conditioned_unet_with_baselines(self):
        from models.geo_encoding import GeoConditionedUNet
        from models.unet import build_unet
        torch.manual_seed(0)
        for encoder in ("xyz", "sinusoidal", "static", "hash_static"):
            cfg = geo_cfg(encoder)
            geo_enc = build_geo_encoder(cfg)
            base = build_unet(cfg, use_time=True,
                              extra_in_channels=geo_enc.output_dim)
            model = GeoConditionedUNet(base, geo_enc)
            x = torch.randn(2, 1, 16, 16)
            out = model(x, torch.ones(2), self._payload(encoder))
            self.assertEqual(out.shape, x.shape, encoder)


if __name__ == "__main__":
    unittest.main()
