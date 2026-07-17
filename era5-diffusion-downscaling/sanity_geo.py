"""Offline wiring check for the geographic hash-grid conditioning experiment.

Verifies coordinate mapping (sphere continuity / no dateline seam), the hash-grid
encoder (shape, gradients, determinism, spatial sensitivity), the geo-conditioned
UNet, and a geo-conditioned diffusion train + guided-sampling step.

Run:
    python sanity_geo.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.diffusion import build_diffusion  # noqa: E402
from models.geo_encoding import (GeoConditionedUNet, MultiResHashGrid,  # noqa: E402
                                 build_patch_coords, latlon_to_unit_sphere)
from models.unet import build_unet  # noqa: E402
from utils import load_config  # noqa: E402

results = []


def check(name, cond, extra=""):
    results.append((name, cond))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


def tiny_cfg():
    cfg = load_config("config/default.yaml")
    cfg["patches"]["size"] = 32
    cfg["unet"].update(base_channels=16, channel_mults=[1, 2, 4, 8],
                       attn_resolutions=[4], groupnorm_groups=8, time_emb_dim=32)
    cfg["diffusion"]["timesteps"] = 50
    cfg["geo"].update(enabled=True, input_dim=3, n_levels=4, n_features_per_level=2,
                      log2_hashmap_size=14, base_resolution=8, finest_resolution=64)
    return cfg


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    cfg = tiny_cfg()
    S = cfg["patches"]["size"]

    # 1. Sphere mapping: no dateline seam, pole degeneracy.
    a = latlon_to_unit_sphere(np.array(0.0), np.array(179.9))
    b = latlon_to_unit_sphere(np.array(0.0), np.array(-179.9))
    seam = float(np.linalg.norm(a - b))
    check("dateline continuity (179.9 vs -179.9 close)", seam < 0.01, f"dist={seam:.4f}")
    npole = latlon_to_unit_sphere(np.array(90.0), np.array(123.0))
    check("north pole -> (0,0,1)", np.allclose(npole, [0, 0, 1], atol=1e-6))
    check("unit norm", abs(float(np.linalg.norm(a)) - 1.0) < 1e-6)

    # 2. Patch coords builder.
    lat_vec = np.linspace(35, 25, S, dtype=np.float32)   # decreasing (ERA5 order)
    lon_vec = np.linspace(100, 110, S, dtype=np.float32)
    coords = build_patch_coords(lat_vec, lon_vec)
    check("patch coords shape", coords.shape == (S, S, 3))
    check("patch coords in [0,1]", coords.min() >= 0 and coords.max() <= 1,
          f"[{coords.min():.3f},{coords.max():.3f}]")

    # 3. Hash-grid encoder: shape, determinism, gradients, spatial sensitivity.
    enc = MultiResHashGrid(input_dim=3, n_levels=4, n_features_per_level=2,
                           log2_hashmap_size=14, base_resolution=8,
                           finest_resolution=64).to(device)
    print(f"      geo embedding dim: {enc.output_dim}, table params: "
          f"{sum(p.numel() for p in enc.parameters()):,}")
    c = torch.from_numpy(coords).to(device)
    e1 = enc(c)
    e2 = enc(c)
    check("encoder output shape", tuple(e1.shape) == (S, S, enc.output_dim))
    check("encoder deterministic", torch.equal(e1, e2))
    check("encoder output finite", torch.isfinite(e1).all().item())
    # spatially varying (different pixels -> different embeddings)
    var = e1.reshape(-1, enc.output_dim).std(dim=0).mean().item()
    check("embedding varies across space", var > 0, f"mean-std={var:.4e}")
    loss = e1.pow(2).mean()
    loss.backward()
    gn = sum(p.grad.abs().sum().item() for p in enc.parameters() if p.grad is not None)
    check("encoder gradients flow", gn > 0)

    # 4. Geo-conditioned UNet.
    geo_enc = MultiResHashGrid(input_dim=3, n_levels=4, n_features_per_level=2,
                               log2_hashmap_size=14, base_resolution=8,
                               finest_resolution=64)
    base = build_unet(cfg, use_time=True, extra_in_channels=geo_enc.output_dim)
    model = GeoConditionedUNet(base, geo_enc).to(device)
    check("unet in_channels = 1 + emb", base.conv_in.in_channels == 1 + geo_enc.output_dim,
          f"in={base.conv_in.in_channels}")
    x = torch.randn(2, 1, S, S, device=device)
    t = torch.randint(1, 50, (2,), device=device).float()
    cb = torch.stack([c, c]).to(device)              # (2, S, S, 3)
    out = model(x, t, cb)
    check("geo-UNet forward shape (noise, 1ch)", out.shape == x.shape, str(tuple(out.shape)))

    # 5. Geo-conditioned diffusion training step.
    diffusion = build_diffusion(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dloss = diffusion.training_loss(model, x, cond=cb)
    dloss.backward()
    g_unet = sum(p.grad.abs().sum().item() for p in base.parameters() if p.grad is not None)
    g_geo = sum(p.grad.abs().sum().item() for p in geo_enc.parameters() if p.grad is not None)
    opt.step()
    check("geo diffusion loss finite", torch.isfinite(dloss).item(), f"loss={dloss.item():.4f}")
    check("gradients reach UNet", g_unet > 0)
    check("gradients reach geo tables", g_geo > 0)

    # 6. Geo-conditioned guided reconstruction.
    from data.degrade import degrade
    xg = degrade(x, ratio=4)
    rec = diffusion.guided_reconstruct(model, xg, t_steps=[6, 4], K=2, eta=0.0, cond=cb)
    check("geo guided_reconstruct shape", rec.shape == x.shape)
    check("geo guided_reconstruct finite", torch.isfinite(rec).all().item())

    n_pass = sum(c for _, c in results)
    print(f"\n{n_pass}/{len(results)} checks passed")
    if n_pass != len(results):
        print("FAILURES:", [n for n, ok in results if not ok])
        sys.exit(1)


if __name__ == "__main__":
    main()
