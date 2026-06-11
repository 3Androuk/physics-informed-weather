"""Offline wiring check on tiny synthetic data (no GCS download required).

Exercises every code path with a small UNet/short schedule so it runs in
seconds. Verifies shapes, finiteness, a gradient step, and the guided sampler.

Run:
    python sanity_check.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.dataset import Normalizer, PatchDataset  # noqa: E402
from data.degrade import coarsen, degrade, gaussian_smooth, upsample_nearest  # noqa: E402
from data.make_patches import crop_patches  # noqa: E402
from eval.metrics import (l2_norm, radial_power_spectrum,  # noqa: E402
                          spectrum_log_l1, value_histogram)
from models.diffusion import build_diffusion  # noqa: E402
from models.unet import build_unet  # noqa: E402
from sample.reconstruct import (reconstruct_bicubic, reconstruct_diffusion,  # noqa: E402
                                reconstruct_directmap)
from utils import load_config  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond, extra=""):
    results.append((name, cond))
    print(f"[{PASS if cond else FAIL}] {name} {extra}")


def tiny_cfg():
    cfg = load_config("config/default.yaml")
    cfg["patches"]["size"] = 32
    cfg["unet"].update(base_channels=16, channel_mults=[1, 2, 4, 8],
                       attn_resolutions=[4], groupnorm_groups=8, time_emb_dim=32)
    cfg["diffusion"]["timesteps"] = 50
    return cfg


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    cfg = tiny_cfg()
    S = cfg["patches"]["size"]

    # 1. UNet with time conditioning.
    unet = build_unet(cfg, use_time=True).to(device)
    x = torch.randn(2, 1, S, S, device=device)
    t = torch.randint(1, 50, (2,), device=device).float()
    out = unet(x, t)
    check("UNet(use_time) forward shape", out.shape == x.shape, str(tuple(out.shape)))
    check("UNet output finite", torch.isfinite(out).all().item())
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"      tiny UNet params: {n_params:,}")

    # 2. UNet direct-map (no time).
    dm = build_unet(cfg, use_time=False).to(device)
    out_dm = dm(x)
    check("UNet(no time) forward shape", out_dm.shape == x.shape)

    # 3. Diffusion training loss + gradient step.
    diffusion = build_diffusion(cfg).to(device)
    check("alphas_cumprod length T+1", diffusion.alphas_cumprod.numel() == 51)
    check("abar[0]==1", abs(diffusion.alphas_cumprod[0].item() - 1.0) < 1e-6)
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-3)
    loss = diffusion.training_loss(unet, x)
    loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in unet.parameters() if p.grad is not None)
    opt.step()
    check("training_loss scalar + finite", loss.dim() == 0 and torch.isfinite(loss).item(),
          f"loss={loss.item():.4f}")
    check("gradients flow", grad_norm > 0)

    # 4. q_sample shape.
    xt = diffusion.q_sample(x, t.long(), torch.randn_like(x))
    check("q_sample shape", xt.shape == x.shape)

    # 5. Guided reconstruction (short schedule).
    x_g = degrade(x, ratio=4)
    recon = diffusion.guided_reconstruct(unet, x_g, t_steps=[6, 4], K=2, eta=0.0)
    check("guided_reconstruct shape", recon.shape == x.shape)
    check("guided_reconstruct finite", torch.isfinite(recon).all().item())

    # 6. Unconditional sample.
    samp = diffusion.sample_unconditional(unet, (2, 1, S, S), device, n_steps=5)
    check("unconditional sample shape", samp.shape == (2, 1, S, S))

    # 7. Degradation operators.
    c4 = coarsen(x, 4)
    check("coarsen 4x shape", c4.shape == (2, 1, S // 4, S // 4))
    up = upsample_nearest(c4, S)
    check("nearest upsample shape", up.shape == x.shape)
    d8 = degrade(x, 8, smooth_sigma=5.0)
    check("degrade 8x + smooth shape/finite", d8.shape == x.shape and torch.isfinite(d8).all().item())
    sm = gaussian_smooth(x, 2.0)
    check("gaussian_smooth shape", sm.shape == x.shape)

    # 8. Patch cropping + normalizer + dataset round-trip.
    rng = np.random.default_rng(0)
    fields = rng.normal(54000, 3000, size=(3, 64, 96)).astype(np.float32)
    patches = crop_patches(fields, size=32, per_field=4, rng=rng)
    check("crop_patches shape", patches.shape == (12, 1, 32, 32))
    norm = Normalizer(float(patches.mean()), float(patches.std()))
    rt = norm.decode(norm.encode(torch.from_numpy(patches[0])))
    check("normalizer round-trip", torch.allclose(rt, torch.from_numpy(patches[0]), atol=1e-2))
    tmp = Path("datasets/_sanity_patches.npy")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    np.save(tmp, patches)
    pd = PatchDataset(tmp, norm)
    check("PatchDataset item shape", tuple(pd[0].shape) == (1, 32, 32))
    import gc
    del pd  # release the mmap handle before unlinking (Windows)
    gc.collect()
    try:
        tmp.unlink()
    except PermissionError:
        pass

    # 9. Reconstruction baselines (normalized space).
    bic = reconstruct_bicubic(x, 4)
    check("bicubic shape", bic.shape == x.shape)
    rc = {"K": 1, "t_steps": [5], "smooth_sigma": 0.0}
    rdiff = reconstruct_diffusion(diffusion, unet, x, 4, rc)
    check("reconstruct_diffusion shape", rdiff.shape == x.shape)
    rdm = reconstruct_directmap(dm, x, 8, smooth_sigma=5.0)
    check("reconstruct_directmap (OOD) shape", rdm.shape == x.shape)

    # 10. Metrics.
    a = np.random.randn(8, 32, 32).astype(np.float32)
    b = a + 0.1 * np.random.randn(8, 32, 32).astype(np.float32)
    l2 = l2_norm(a, b)
    k, e = radial_power_spectrum(a)
    sm1 = spectrum_log_l1(a, b)
    c, h = value_histogram(a, bins=20)
    check("l2_norm positive finite", l2 > 0 and np.isfinite(l2), f"l2={l2:.4f}")
    check("radial_power_spectrum shapes", k.shape == e.shape and len(k) == 17)
    check("spectrum_log_l1 finite", np.isfinite(sm1), f"={sm1:.4f}")
    check("value_histogram shapes", c.shape == h.shape == (20,))

    n_pass = sum(c for _, c in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILURES:", [n for n, c in results if not c])
        sys.exit(1)


if __name__ == "__main__":
    main()
