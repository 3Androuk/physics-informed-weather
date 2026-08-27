"""Correctness tests for the sphere-native residual diffusion model.

No data, no GPU. The important one is test_projection_is_exact: the claim that
on the HEALPix mesh the data-consistency projection is exact and global (not an
approximation reconciled across tiles, as in the sibling patch pipeline) is the
central argument for doing diffusion on the sphere, so it is checked to
floating-point tolerance rather than assumed.

Run with pytest (`python -m pytest tests/ -q`) or directly
(`python -m tests.test_diffusion`).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen_faces, degrade_faces  # noqa: E402
from models.hpx_diffusion import (HPXGaussianDiffusion, make_beta_schedule,  # noqa: E402
                                  project_faces)
from models.hpx_residual import MeanField, build_residual_model  # noqa: E402
from models.hpx_unet import build_model, count_params  # noqa: E402

NSIDE, RATIO = 8, 4

CFG = {
    "hpx": {"nside": NSIDE},
    "sr": {"ratio": RATIO},
    "model": {"in_channels": 1, "out_channels": 1, "channels": [8, 16],
              "dilations": [1, 2], "blocks_per_level": 1, "expansion": 2,
              "gelu_cap": 10.0, "global_residual": False,
              "grad_checkpoint": False},
    "diffusion": {"timesteps": 50, "beta_schedule": "cosine",
                  "beta_start": 1e-4, "beta_end": 2e-2},
}


def _field(b=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, 12, 1, NSIDE, NSIDE, generator=g)


def test_projection_is_exact():
    """coarsen(project(x, lf)) == lf, everywhere, to float tolerance."""
    y, x0 = _field(seed=1), _field(seed=2)
    lf = coarsen_faces(y, RATIO)
    projected = project_faces(x0, lf, RATIO)
    back = coarsen_faces(projected, RATIO)
    assert torch.allclose(back, lf, atol=1e-5), (back - lf).abs().max().item()
    # and it is a genuine correction, not a no-op
    assert not torch.allclose(projected, x0)


def test_projection_preserves_fine_structure():
    """The projection only shifts block means; within-block detail survives."""
    x0 = _field(seed=3)
    lf = coarsen_faces(_field(seed=4), RATIO)
    p = project_faces(x0, lf, RATIO)
    # the difference is constant within every coarse block
    d = p - x0
    assert torch.allclose(d, degrade_faces(d, RATIO), atol=1e-5)


def test_beta_schedules_are_valid():
    for name in ("linear", "cosine"):
        betas = make_beta_schedule(name, 100, 1e-4, 2e-2)
        assert betas.shape == (100,)
        assert (betas > 0).all() and (betas < 1).all()
    d = HPXGaussianDiffusion(**CFG["diffusion"])
    # abar[0] == 1 (clean data) and decreases monotonically
    assert abs(float(d.alphas_cumprod[0]) - 1.0) < 1e-6
    assert (d.alphas_cumprod[1:] <= d.alphas_cumprod[:-1] + 1e-6).all()
    assert len(d.alphas_cumprod) == d.timesteps + 1


def test_q_sample_matches_closed_form():
    d = HPXGaussianDiffusion(**CFG["diffusion"])
    x0, noise = _field(seed=5), _field(seed=6)
    t = torch.tensor([1, d.timesteps])
    xt = d.q_sample(x0, t, noise)
    for i, ti in enumerate(t.tolist()):
        want = (d.sqrt_abar[ti] * x0[i] + d.sqrt_one_minus_abar[ti] * noise[i])
        assert torch.allclose(xt[i], want, atol=1e-6)
    # t=1 stays close to the data; t=T is dominated by noise
    assert (xt[0] - x0[0]).abs().mean() < (xt[1] - x0[1]).abs().mean()


def test_time_conditioning_changes_output():
    """The timestep embedding must actually reach the blocks."""
    torch.manual_seed(0)
    model = build_model(CFG, use_time=True, extra_in_channels=0)
    x = _field(seed=7)
    a = model(x, torch.tensor([1.0, 1.0]))
    b = model(x, torch.tensor([40.0, 40.0]))
    assert not torch.allclose(a, b, atol=1e-6)
    # and each sample's own t is used (not just the first)
    c = model(x, torch.tensor([1.0, 40.0]))
    assert torch.allclose(c[0], a[0], atol=1e-6)
    assert torch.allclose(c[1], b[1], atol=1e-6)


def test_untimed_model_unchanged():
    """The deterministic SR path still builds and runs without a timestep."""
    model = build_model(CFG)
    out = model(_field(seed=8))
    assert out.shape == (2, 12, 1, NSIDE, NSIDE)
    assert count_params(model) > 0


def test_residual_model_forward_and_loss():
    torch.manual_seed(0)
    model = build_residual_model(CFG)
    d = HPXGaussianDiffusion(**CFG["diffusion"])
    x0, mean = _field(seed=9), _field(seed=10)
    loss = d.training_loss(model, x0, cond=(mean,))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_sampling_shape_and_data_consistency():
    """A sampled field must satisfy the observation constraint exactly."""
    torch.manual_seed(0)
    model = build_residual_model(CFG)
    d = HPXGaussianDiffusion(**CFG["diffusion"])
    y = _field(seed=11)
    lf = coarsen_faces(y, RATIO)
    mean = degrade_faces(y, RATIO)
    out = d.sample(model, mean, lf, RATIO, n_steps=5, project=True)
    assert out.shape == y.shape and torch.isfinite(out).all()
    back = coarsen_faces(out, RATIO)
    assert torch.allclose(back, lf, atol=1e-4), (back - lf).abs().max().item()


def test_sampling_is_seeded_and_stochastic():
    torch.manual_seed(0)
    model = build_residual_model(CFG)
    d = HPXGaussianDiffusion(**CFG["diffusion"])
    y = _field(seed=12)
    lf, mean = coarsen_faces(y, RATIO), degrade_faces(y, RATIO)
    kw = dict(n_steps=4, project=True)
    a = d.sample(model, mean, lf, RATIO, generator=torch.Generator().manual_seed(7), **kw)
    b = d.sample(model, mean, lf, RATIO, generator=torch.Generator().manual_seed(7), **kw)
    c = d.sample(model, mean, lf, RATIO, generator=torch.Generator().manual_seed(8), **kw)
    assert torch.allclose(a, b, atol=1e-6)      # same seed -> same field
    assert not torch.allclose(a, c, atol=1e-6)  # different seed -> ensemble member


def test_bilinear_mean_field():
    """The Phase-A mean is a real seam-aware upsample of the coarse field."""
    mf = MeanField("bilinear", RATIO, nside=NSIDE)
    y = _field(seed=13)
    out = mf(degrade_faces(y, RATIO))
    assert out.shape == y.shape and torch.isfinite(out).all()
    # bilinear preserves the coarse block means only approximately, but must
    # stay in range and be smoother than the blocky nearest input
    nearest = degrade_faces(y, RATIO)
    tv = lambda z: (z[..., 1:, :] - z[..., :-1, :]).abs().mean()
    assert tv(out) < tv(nearest) or np.isclose(float(tv(out)), float(tv(nearest)))


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"{name}: OK")
        except Exception as e:  # noqa: BLE001 - report all failures at the end
            failed.append(name)
            print(f"{name}: FAILED ({type(e).__name__}: {e})")
    if failed:
        raise SystemExit(f"\n{len(failed)}/{len(tests)} tests failed: {failed}")
    print(f"\n{len(tests)} tests passed")


if __name__ == "__main__":
    main()
