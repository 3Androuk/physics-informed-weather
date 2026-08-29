"""End-to-end CPU smoke run of train_diffusion.

Exists because of a real failure: a NameError on `stem` in main() killed a
4-GPU job 37 seconds in. py_compile does not catch unresolved names, the
divergence-guard unit tests exercise the class in isolation, and
test_distributed.py never invokes a trainer — so nothing in the suite executed
the code path that broke.

This runs the actual module as a subprocess on a tiny synthetic dataset for a
couple of epochs. It is slower than the rest of the suite and worth it: it
catches NameErrors, bad config keys, checkpoint-path mistakes and wiring
regressions before they reach a billed GPU node.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

CFG = """
seed: 42
paths: {{raw_dir: "{d}/raw", patch_dir: "{d}/patches", ckpt_dir: "{d}/ckpt",
         results_dir: "{d}/res", log_dir: "{d}/res/tb"}}
data: {{variable: 2m_temperature, level: null}}
patches: {{size: 16, per_field: 2, lat_range: [-60.0, 60.0]}}
normalize: {{method: zscore}}
diffusion: {{timesteps: 20, beta_schedule: linear, beta_start: 1.0e-4, beta_end: 2.0e-2}}
unet: {{in_channels: 1, out_channels: 1, base_channels: 8, channel_mults: [1, 2],
        num_res_blocks: 1, time_emb_dim: 16, attn_resolutions: [8], dropout: 0.0,
        groupnorm_groups: 4}}
geo: {{enabled: false}}
train: {{batch_size: 4, epochs: 2, lr: 1.0e-4, weight_decay: 0.0, grad_clip: 1.0,
         amp: false, amp_dtype: off, ema_decay: 0.9, num_workers: 0, log_every: 1,
         val_patches: 4, ckpt_every_epochs: 1, sample_every_epochs: 99,
         divergence: {{enabled: true, factor: 10.0, patience: 3, min_epochs: 5}}}}
sample: {{ddim_eta: 0.0, guidance_strength: 0.0, interp: nearest,
          reconstructions: [{{ratio: 4, K: 1, t_steps: [5], smooth_sigma: 0.0}}]}}
eval: {{n_test_patches: 4, display_channel: 0, spectrum_bins: null, hist_bins: 10}}
wandb: {{enabled: false}}
"""


class TrainerSmokeTest(unittest.TestCase):
    def test_train_diffusion_runs_and_writes_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(0)
            for split, n in (("train", 8), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(CFG.format(d=d))

            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0,
                             f"trainer failed\nSTDOUT:\n{r.stdout[-2500:]}"
                             f"\nSTDERR:\n{r.stderr[-2500:]}")

            ck = dd / "ckpt"
            rolling = ck / "diffusion.pt"
            best = ck / "diffusion_best.pt"
            self.assertTrue(rolling.exists(), f"no rolling checkpoint; got {list(ck.iterdir())}")
            self.assertTrue(best.exists(), f"no best checkpoint; got {list(ck.iterdir())}")
            self.assertIn("Best (val", r.stdout)

    def test_resume_from_the_rolling_checkpoint(self):
        """--resume is passed unconditionally by scripts/isambard_train.sbatch."""
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(1)
            for split, n in (("train", 8), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(CFG.format(d=d))
            base = [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg)]

            first = subprocess.run(base, cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(first.returncode, 0, first.stderr[-2000:])
            # ... and again with --resume, which must not crash on an existing ckpt
            second = subprocess.run(base + ["--resume"], cwd=REPO,
                                    capture_output=True, text=True, timeout=900)
            self.assertEqual(second.returncode, 0,
                             f"--resume failed\nSTDERR:\n{second.stderr[-2500:]}")

    def test_resume_on_a_fresh_run_is_not_an_error(self):
        """The launcher always passes --resume, including the very first run."""
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d)
            (dd / "patches").mkdir(parents=True)
            rng = np.random.default_rng(2)
            for split, n in (("train", 8), ("test", 4)):
                np.save(dd / "patches" / f"{split}_patches.npy",
                        rng.standard_normal((n, 1, 16, 16)).astype("float32"))
            np.savez(dd / "patches" / "norm_stats.npz",
                     mean=np.float32(0.0), std=np.float32(1.0), size=16)
            cfg = dd / "cfg.yaml"
            cfg.write_text(CFG.format(d=d))
            r = subprocess.run(
                [sys.executable, "-m", "train.train_diffusion", "--config", str(cfg), "--resume"],
                cwd=REPO, capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("starting fresh", r.stdout)


if __name__ == "__main__":
    unittest.main()
