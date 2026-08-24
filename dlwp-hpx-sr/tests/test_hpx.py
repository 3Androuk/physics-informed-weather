"""Correctness tests for the HEALPix mesh, padding, remap, and model.

Run with pytest (`python -m pytest tests/ -q`) or directly
(`python -m tests.test_hpx`).

The padding tests validate the derived halo index maps against the true
sphere topology from astropy-healpix: every grid adjacency in the padded maps
must be a real pixel adjacency of the right class (orthogonal grid neighbours
must be edge-sharing on the sphere, diagonal ones diagonal). This catches any
sheared, rotated, or misplaced halo assignment. The only tolerated exceptions
are the 24 diagonal pairs flanking the eight 3-valent mesh vertices, where
only three faces meet and the two flanking halo pixels really are
edge-sharing on the sphere.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen_faces, degrade_faces, upsample_bilinear_faces  # noqa: E402
from hpx.mesh import (face_index_map, face_xy_to_nest, neighbour_table,  # noqa: E402
                      nest_to_face_xy, nest_to_faces, npix, pixel_lonlat_deg)
from hpx.padding import HEALPixPadding, build_pad_index  # noqa: E402
from hpx.remap import LatLonToHPX, hpx_to_latlon  # noqa: E402
from models.hpx_unet import HEALPixUNetSR, count_params  # noqa: E402


def test_face_xy_bijection():
    for nside in (1, 2, 8, 16):
        idx = face_index_map(nside)
        assert idx.shape == (12, nside, nside)
        assert np.array_equal(np.sort(idx.ravel()), np.arange(npix(nside)))
        f, x, y = nest_to_face_xy(idx, nside)
        assert np.array_equal(idx, face_xy_to_nest(f, x, y, nside))


def test_face_grid_adjacency_is_sphere_adjacency():
    nside = 8
    idx = face_index_map(nside)
    nb = neighbour_table(nside)
    nbsets = [set(int(v) for v in nb[:, p] if v >= 0) for p in range(npix(nside))]
    for f in range(12):
        for yy in range(nside):
            for xx in range(nside):
                p = int(idx[f, yy, xx])
                for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    y2, x2 = yy + dy, xx + dx
                    if 0 <= y2 < nside and 0 <= x2 < nside:
                        assert int(idx[f, y2, x2]) in nbsets[p]


def _edge_diag_sets(nside):
    """Per-pixel edge-sharing vs diagonal neighbour sets (slot classes)."""
    nb = neighbour_table(nside)
    f, x, y = nest_to_face_xy(np.arange(npix(nside)), nside)
    p0 = int(face_xy_to_nest(0, nside // 2, nside // 2, nside))
    edge_slots = []
    for slot in range(8):
        q = int(nb[slot, p0])
        if (f[q] == f[p0]
                and abs(int(x[q]) - int(x[p0])) + abs(int(y[q]) - int(y[p0])) == 1):
            edge_slots.append(slot)
    assert len(edge_slots) == 4
    edge = [set(int(v) for s, v in enumerate(nb[:, p]) if v >= 0 and s in edge_slots)
            for p in range(npix(nside))]
    diag = [set(int(v) for s, v in enumerate(nb[:, p]) if v >= 0 and s not in edge_slots)
            for p in range(npix(nside))]
    return edge, diag


def test_pad_index_topology():
    for nside, pad in ((8, 1), (8, 2), (16, 1), (16, 4)):
        idx = build_pad_index(nside, pad)
        edge, diag = _edge_diag_sets(nside)
        S = nside + 2 * pad
        lo, hi = pad, pad + nside
        bad_diag_pairs = []
        for f in range(12):
            for i in range(S):
                for j in range(S):
                    p = idx[f, i, j]
                    if p < 0:
                        continue
                    for di, dj, ortho in ((0, 1, True), (1, 0, True),
                                          (1, 1, False), (1, -1, False)):
                        i2, j2 = i + di, j + dj
                        if not (0 <= i2 < S and 0 <= j2 < S):
                            continue
                        q = int(idx[f, i2, j2])
                        if q < 0:
                            continue
                        if ortho:
                            assert q in edge[p], (nside, pad, f, i, j, i2, j2)
                        elif q not in diag[p]:
                            # tolerated only when flanking a 3-valent corner:
                            # the pair must still be edge-sharing on the sphere
                            assert q in edge[p], (nside, pad, f, i, j, i2, j2)
                            bad_diag_pairs.append((f, i, j))
        # exactly two 3-valent corners per face -> 24 flanking pairs total
        assert len(bad_diag_pairs) == 24, len(bad_diag_pairs)

        # interior must be the identity face map
        assert np.array_equal(idx[:, lo:hi, lo:hi], face_index_map(nside))
        # strips (everything except corner blocks) must be fully resolved
        strip = np.zeros((S, S), dtype=bool)
        strip[lo:hi, :] = True
        strip[:, lo:hi] = True
        assert (idx[:, strip] >= 0).all()
        # unresolved cells only in the two 3-valent corner blocks per face
        assert (idx < 0).sum() == 12 * 2 * pad * pad


def test_padding_preserves_constant():
    for nside, pad in ((8, 1), (8, 2), (16, 2)):
        padder = HEALPixPadding(nside, pad)
        x = torch.ones(2 * 12, 3, nside, nside)
        out = padder(x)
        assert out.shape == (24, 3, nside + 2 * pad, nside + 2 * pad)
        assert torch.allclose(out, torch.ones_like(out))


def test_padding_matches_smooth_function():
    """Halo values must equal the field at the halo pixels' true positions."""
    nside, pad = 16, 2
    lon, lat = pixel_lonlat_deg(nside)
    lam, phi = np.radians(lon), np.radians(lat)
    field = (np.sin(2 * lam) * np.cos(phi) ** 2 + np.sin(phi)).astype(np.float32)
    faces = nest_to_faces(field, nside)

    padder = HEALPixPadding(nside, pad)
    out = padder(torch.from_numpy(faces).unsqueeze(1))  # (12, 1, S, S)
    # the face interior must come through the gather unchanged
    assert torch.equal(out[:, :, pad:-pad, pad:-pad],
                       torch.from_numpy(faces).unsqueeze(1))
    idx = build_pad_index(nside, pad)
    valid = idx >= 0
    got = out[:, 0].numpy()
    # every resolved cell must equal the field at its true pixel
    assert np.allclose(got[valid], field[np.clip(idx, 0, None)][valid], atol=1e-6)
    # unresolved corner cells are (transitively) convex combinations of the
    # valid cells around them, so they must lie inside the local value range
    S = nside + 2 * pad
    for f, i, j in zip(*np.nonzero(~valid)):
        i0, i1 = max(0, i - pad - 1), min(S, i + pad + 2)
        j0, j1 = max(0, j - pad - 1), min(S, j + pad + 2)
        near = got[f, i0:i1, j0:j1][valid[f, i0:i1, j0:j1]]
        assert near.min() - 1e-6 <= got[f, i, j] <= near.max() + 1e-6


def test_remap_roundtrip():
    nside = 32
    lat = np.linspace(-89.5, 89.5, 180)
    lon = np.arange(0.0, 360.0, 1.0)
    lon2, lat2 = np.meshgrid(lon, lat)
    lam, phi = np.radians(lon2), np.radians(lat2)
    field = (np.sin(3 * lam) * np.cos(phi) ** 2 + np.cos(phi) * np.sin(phi))

    remap = LatLonToHPX(lat, lon, nside)
    faces = remap(field[None])[0]
    assert faces.shape == (12, nside, nside)

    # forward: faces must match the analytic function at pixel centers
    plon, plat = pixel_lonlat_deg(nside)
    lam_p, phi_p = np.radians(plon), np.radians(plat)
    expect = np.sin(3 * lam_p) * np.cos(phi_p) ** 2 + np.cos(phi_p) * np.sin(phi_p)
    err = np.abs(faces - nest_to_faces(expect.astype(np.float32), nside))
    assert err.max() < 5e-3, err.max()

    # roundtrip back to lat-lon
    back = hpx_to_latlon(faces, lat, lon)
    assert back.shape == field.shape
    assert np.abs(back - field).max() < 5e-2


def test_degrade_is_healpix_coarsening():
    """Face avg-pooling must equal remapping at the coarser nside (smooth field)."""
    nside, ratio = 32, 4
    lon, lat = pixel_lonlat_deg(nside)
    lam, phi = np.radians(lon), np.radians(lat)
    field = (np.sin(lam) * np.cos(phi) + 0.5 * np.sin(phi)).astype(np.float32)
    faces = torch.from_numpy(nest_to_faces(field, nside))[None, :, None]

    lo = coarsen_faces(faces, ratio)
    assert lo.shape == (1, 12, 1, nside // ratio, nside // ratio)
    lon_c, lat_c = pixel_lonlat_deg(nside // ratio)
    lam_c, phi_c = np.radians(lon_c), np.radians(lat_c)
    expect_c = (np.sin(lam_c) * np.cos(phi_c) + 0.5 * np.sin(phi_c)).astype(np.float32)
    err = np.abs(lo[0, :, 0].numpy() - nest_to_faces(expect_c, nside // ratio))
    assert err.max() < 2e-2, err.max()

    up = degrade_faces(faces, ratio)
    assert up.shape == faces.shape
    bi = upsample_bilinear_faces(lo, ratio)
    assert bi.shape == faces.shape
    # bilinear must beat nearest on a smooth field
    assert (bi - faces).abs().mean() < (up - faces).abs().mean()


def test_model_forward_backward():
    torch.manual_seed(0)
    nside = 16
    model = HEALPixUNetSR(nside=nside, channels=(8, 12, 16), dilations=(1, 2, 2),
                          blocks_per_level=1, expansion=2)
    n = count_params(model)
    assert n > 0
    x = torch.randn(2, 12, 1, nside, nside)
    out = model(x)
    assert out.shape == (2, 12, 1, nside, nside)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    # gradient checkpointing must give the same output and finite grads
    model_ck = HEALPixUNetSR(nside=nside, channels=(8, 12, 16), dilations=(1, 2, 2),
                             blocks_per_level=1, expansion=2, grad_checkpoint=True)
    model_ck.load_state_dict(model.state_dict())
    model_ck.train()
    out_ck = model_ck(x)
    assert torch.allclose(out_ck, out, atol=1e-6)
    out_ck.square().mean().backward()
    grads_ck = [p.grad for p in model_ck.parameters() if p.grad is not None]
    assert grads_ck and all(torch.isfinite(g).all() for g in grads_ck)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    nside, ratio = 16, 4
    lon, lat = pixel_lonlat_deg(nside)
    lam, phi = np.radians(lon), np.radians(lat)
    fields = np.stack([
        np.sin(k * lam) * np.cos(phi) ** 2 + np.sin(phi) for k in (1, 2, 3, 4)
    ]).astype(np.float32)
    y = torch.from_numpy(nest_to_faces(fields, nside))[:, :, None]
    x = degrade_faces(y, ratio)

    model = HEALPixUNetSR(nside=nside, channels=(8, 12), dilations=(1, 2),
                          blocks_per_level=1, expansion=2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])


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
