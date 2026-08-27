"""Mixed-precision resolution: config/CLI -> (enabled, torch dtype).

The back-compat rule is the one that matters. `autocast("cuda")` defaults to
fp16, so a pre-existing `amp: true` config means FP16; reading it as bf16 would
silently change the numerics of runs that already exist.
"""

import unittest
from unittest import mock

import torch

from utils import add_perf_args, apply_perf_overrides, resolve_amp


class ResolveAmpTests(unittest.TestCase):
    def test_absent_means_off(self):
        self.assertEqual(resolve_amp({}, "cuda"), (False, None))

    def test_legacy_amp_true_is_fp16_not_bf16(self):
        """A config written before amp_dtype existed must keep its numerics."""
        enabled, dtype = resolve_amp({"amp": True}, "cuda")
        self.assertTrue(enabled)
        self.assertIs(dtype, torch.float16)

    def test_legacy_amp_false_is_off(self):
        self.assertEqual(resolve_amp({"amp": False}, "cuda"), (False, None))

    def test_explicit_bf16(self):
        with mock.patch.object(torch.cuda, "is_bf16_supported", lambda: True):
            enabled, dtype = resolve_amp({"amp_dtype": "bf16"}, "cuda")
        self.assertTrue(enabled)
        self.assertIs(dtype, torch.bfloat16)

    def test_amp_dtype_overrides_legacy_amp(self):
        with mock.patch.object(torch.cuda, "is_bf16_supported", lambda: True):
            _, dtype = resolve_amp({"amp": True, "amp_dtype": "bf16"}, "cuda")
        self.assertIs(dtype, torch.bfloat16)
        self.assertEqual(resolve_amp({"amp": True, "amp_dtype": "off"}, "cuda"),
                         (False, None))

    def test_cpu_disables_everything(self):
        self.assertEqual(resolve_amp({"amp_dtype": "bf16"}, "cpu"), (False, None))

    def test_bf16_refused_when_unsupported(self):
        with mock.patch.object(torch.cuda, "is_bf16_supported", lambda: False):
            with self.assertRaises(RuntimeError):
                resolve_amp({"amp_dtype": "bf16"}, "cuda")

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            resolve_amp({"amp_dtype": "tf32"}, "cuda")


class PerfOverrideTests(unittest.TestCase):
    def _args(self, argv):
        import argparse
        ap = argparse.ArgumentParser()
        add_perf_args(ap)
        return ap.parse_args(argv)

    def test_cli_lands_in_the_right_section(self):
        cfg = {"train": {"batch_size": 20, "num_workers": 4},
               "directmap": {"batch_size": 20}}
        args = self._args(["--batch-size", "64", "--num-workers", "16",
                           "--amp-dtype", "bf16"])

        apply_perf_overrides(cfg, args, "train")
        self.assertEqual(cfg["train"]["batch_size"], 64)
        self.assertEqual(cfg["train"]["num_workers"], 16)
        self.assertEqual(cfg["train"]["amp_dtype"], "bf16")
        self.assertNotIn("amp_dtype", cfg["directmap"])

        apply_perf_overrides(cfg, args, "directmap")
        self.assertEqual(cfg["directmap"]["batch_size"], 64)
        self.assertEqual(cfg["directmap"]["amp_dtype"], "bf16")

    def test_absent_flags_leave_config_untouched(self):
        cfg = {"train": {"batch_size": 20, "num_workers": 4}}
        apply_perf_overrides(cfg, self._args([]), "train")
        self.assertEqual(cfg["train"], {"batch_size": 20, "num_workers": 4})


if __name__ == "__main__":
    unittest.main()
