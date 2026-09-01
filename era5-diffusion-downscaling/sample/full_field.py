"""Full-field reconstruction: tiled overlap-blend stitching of patch models.

Turns the patch-trained reconstructors into whole-field reconstructors. Two
strategies (both compared by eval/full_field.py):

  * DIRECT — run the fully-convolutional model on the entire field at once.
    No code here beyond what the samplers already support; the bottleneck
    self-attention sees far more tokens than at training time (a train/test
    mismatch), so treat its output with suspicion until validated.
  * TILED — reconstruct overlapping training-sized tiles and blend them with a
    smooth window (this module). Matches the training distribution exactly.
    Overlapping tiles crop their starting noise from ONE global noise field,
    so stochastic reconstructions agree where they overlap, and a final exact
    block-average projection re-pins the stitched field to the observed
    coarse input globally.

All functions operate in NORMALIZED units. Tile origins are snapped to
multiples of the degradation ratio so per-tile coarsening commutes with
cropping (a tile's coarse observation IS the crop of the global one).
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.degrade import coarsen, upsample_nearest  # noqa: E402


def crop_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    """Crop trailing (H, W) down to the nearest multiples of m (e.g. the raw
    481-row grid -> 480 so every ratio and the UNet's downsampling divide it)."""
    h, w = x.shape[-2:]
    return x[..., : (h // m) * m, : (w // m) * m]


def tile_origins(length: int, tile: int, stride: int, align: int = 1) -> list[int]:
    """Tile start positions covering [0, length), all multiples of `align`.

    The final tile is right-aligned to end exactly at `length` (its stride to
    the previous tile may be smaller)."""
    if tile % align or length % align:
        raise ValueError(f"tile {tile} and length {length} must be multiples of {align}")
    if tile > length:
        raise ValueError(f"tile {tile} larger than field {length}")
    stride = max(align, (stride // align) * align)
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)  # length - tile is a multiple of align
    return starts


def blend_window(tile: int, overlap: int, device=None, dtype=None) -> torch.Tensor:
    """(1, 1, tile, tile) separable cosine-ramp blending window.

    Flat in the interior, ramping smoothly over `overlap` pixels at each edge.
    Floored above zero so field-edge pixels covered by a single tile normalize
    to exactly that tile's value (out = sum w*tile / sum w)."""
    ramp = torch.ones(tile, device=device, dtype=dtype)
    if overlap > 0:
        t = torch.linspace(0.0, 1.0, overlap + 2, device=device, dtype=dtype)[1:-1]
        r = 0.5 - 0.5 * torch.cos(math.pi * t)
        ramp[:overlap] = r
        ramp[-overlap:] = r.flip(0)
    win = ramp[:, None] * ramp[None, :]
    return win.clamp_min(1e-4)[None, None]


def crop_tiles(full: torch.Tensor, origins, tile: int) -> torch.Tensor:
    """Crop tiles from a (1, C, H, W) tensor -> (B, C, tile, tile)."""
    return torch.stack([full[0, :, r:r + tile, c:c + tile] for r, c in origins])


def crop_geo_tiles(geo_full, origins, tile: int):
    """Per-tile crops of a geo payload.

    (H, W, d) hash-grid coords -> (B, tile, tile, d);
    (L, H, W, 8) packed HEALPix payload -> (B, L, tile, tile, 8)."""
    if geo_full is None:
        return None
    if geo_full.dim() == 3:
        return torch.stack([geo_full[r:r + tile, c:c + tile] for r, c in origins])
    if geo_full.dim() == 4:
        return torch.stack([geo_full[:, r:r + tile, c:c + tile, :] for r, c in origins])
    raise ValueError(f"unsupported geo payload rank {geo_full.dim()}")


def _global_noise(shape, like: torch.Tensor, generator=None) -> torch.Tensor:
    """Global noise field, drawn on CPU (device-independent for a given seed)."""
    return torch.randn(shape, generator=generator, dtype=like.dtype).to(like.device)


def _crop_coarse(coarse_full: torch.Tensor, origins, tile: int, ratio: int) -> torch.Tensor:
    """Crop the coarse observation for ratio-aligned tiles: (B, C, tile/r, tile/r)."""
    t = tile // ratio
    return torch.stack([coarse_full[0, :, r // ratio:r // ratio + t,
                                    c // ratio:c // ratio + t] for r, c in origins])


@torch.no_grad()
def stitch_tiles(fn, lf_full: torch.Tensor, tile: int, overlap: int,
                 align: int = 1, batch: int = 8) -> torch.Tensor:
    """Overlap-blend a full field from training-sized tile reconstructions.

    fn(origins) -> (B, C, tile, tile) reconstructions for a list of (row, col)
    tile origins; cropping the guidance/noise/geo per tile is the closure's
    job. lf_full: (1, C, H, W) sets the geometry and output buffer."""
    n, _, h, w = lf_full.shape
    assert n == 1, "stitching operates on one field at a time"
    rows = tile_origins(h, tile, tile - overlap, align)
    cols = tile_origins(w, tile, tile - overlap, align)
    origins = [(r, c) for r in rows for c in cols]
    win = blend_window(tile, overlap, lf_full.device, lf_full.dtype)[0]  # (1, t, t)
    out = None  # allocated from the first predictions (channels may differ, e.g. SI's 2C)
    wsum = torch.zeros(1, 1, h, w, device=lf_full.device, dtype=lf_full.dtype)
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        recs = fn(chunk)
        if out is None:
            out = torch.zeros(1, recs.shape[1], h, w,
                              device=recs.device, dtype=recs.dtype)
        for (r, c), rec in zip(chunk, recs):
            out[0, :, r:r + tile, c:c + tile] += rec * win
            wsum[0, :, r:r + tile, c:c + tile] += win
    return out / wsum


def _project_final(out: torch.Tensor, coarse_full: torch.Tensor, ratio: int,
                   observed=None) -> torch.Tensor:
    """Exact global block-average projection: coarsen(out) == observation.

    `observed` (optional bool per channel) restricts the projection to
    channels carrying a real observation — unobserved (generated) channels
    are left untouched."""
    corr = upsample_nearest(coarse_full - coarsen(out, ratio), out.shape[-2:])
    if observed is not None:
        obs = torch.as_tensor(observed, dtype=torch.bool,
                              device=out.device).view(1, -1, 1, 1)
        corr = corr * obs
    return out + corr


@torch.no_grad()
def reconstruct_full_tiled_diffusion(diffusion, model, lf_full, coarse_full, ratio,
                                     recon_cfg, eta=0.0, tile=128, overlap=32,
                                     batch=8, geo_full=None, project_steps=False,
                                     project_final=True, generator=None,
                                     covariance_projector=None, observed=None):
    """Tiled guided-diffusion reconstruction of one full field.

    lf_full: (1, C, H, W) noise-mixing guidance (globally degraded, normalized);
    coarse_full: (1, C, H/r, W/r) the observed coarse field. Each outer loop's
    mixing epsilon is cropped per tile from a global noise field. With
    project_steps=True the per-step ILVR projection also runs inside every
    tile (tile origins are ratio-aligned, so tile observations are exact crops
    of the global one).

    `covariance_projector` swaps that per-step projection for the Weather-DDNM
    covariance-aware one. It applies HERE (and not in the fused sampler)
    because tiles carry the patch geometry the covariance was estimated on;
    the projector validates the grid and raises if they disagree. The final
    global projection stays ordinary — it is a whole-field operator."""
    K = int(recon_cfg["K"])
    noise_full = _global_noise((K, *lf_full.shape), lf_full, generator)
    if covariance_projector is not None:
        if not project_steps:
            raise ValueError("covariance_projector requires project_steps=True")
        if tuple(covariance_projector.image_size) != (tile, tile):
            raise ValueError(
                f"covariance grid {covariance_projector.image_size} != tile "
                f"{(tile, tile)}: estimate the covariance at the tile size")

    def fn(origins):
        x_g = crop_tiles(lf_full, origins, tile)
        eps = torch.stack([crop_tiles(noise_full[k], origins, tile) for k in range(K)])
        coords = crop_geo_tiles(geo_full, origins, tile)
        lf_tiles = _crop_coarse(coarse_full, origins, tile, ratio) if project_steps else None
        return diffusion.guided_reconstruct(
            model, x_g, t_steps=recon_cfg["t_steps"], K=K, eta=eta, cond=coords,
            project=project_steps, lf=lf_tiles, ratio=ratio, init_noise=eps,
            covariance_projector=covariance_projector, observed=observed)

    out = stitch_tiles(fn, lf_full, tile, overlap, align=ratio, batch=batch)
    return (_project_final(out, coarse_full, ratio, observed=observed)
            if project_final else out)


@torch.no_grad()
def reconstruct_full_tiled_transport(model, process, lf_full, coarse_full, ratio,
                                     cfg, method, tile=128, overlap=32, batch=8,
                                     geo_full=None, steps=None, solver=None,
                                     sampler=None, stochasticity=None,
                                     project_final=True, project_each=False,
                                     generator=None):
    """Tiled flow-matching / stochastic-interpolant reconstruction of one full
    field. Every tile's ODE/SDE starts from a crop of one global noise field,
    so deterministic samplers agree exactly where tiles overlap.

    project_each=True additionally projects every integration step inside every
    tile onto its crop of the SHARED coarse observation — anchoring all tiles'
    low frequencies to the same field throughout, so neighboring tiles cannot
    drift apart (visible as tile-scale squares at weakly-constrained ratios)."""
    tc = cfg.get("transport", {})
    noise_full = _global_noise(lf_full.shape, lf_full, generator)
    kwargs = dict(
        steps=tc.get("sample_steps", 100) if steps is None else steps,
        solver=tc.get("solver", "heun") if solver is None else solver,
        project="none",  # global consistency is enforced after stitching
    )
    if method == "stochastic_interpolant":
        si = tc.get("stochastic_interpolant", {})
        kwargs.update(sampler=si.get("sampler", "ode") if sampler is None else sampler,
                      stochasticity=(si.get("stochasticity", 0.1)
                                     if stochasticity is None else stochasticity))

    def fn(origins):
        lows = crop_tiles(lf_full, origins, tile)
        coords = crop_geo_tiles(geo_full, origins, tile)
        noise = crop_tiles(noise_full, origins, tile)
        kw = dict(kwargs)
        if project_each:
            kw.update(project="each",
                      coarse=_crop_coarse(coarse_full, origins, tile, ratio),
                      ratio=ratio)
        return process.sample(model, lows, coords=coords, noise=noise, **kw)

    out = stitch_tiles(fn, lf_full, tile, overlap, align=ratio, batch=batch)
    return _project_final(out, coarse_full, ratio) if project_final else out


class _FusedTileModel:
    """MultiDiffusion-style per-step fusion wrapper (Bar-Tal et al., ICML 2023).

    Wraps a patch model so that ONE forward pass on a FULL field evaluates all
    overlapping tiles of the current state and blends their predictions with
    the stitching window. Plugged into the unchanged global sampling loops
    (guided DDIM / probability-flow ODE / SDE), this binds every tile to a
    single shared trajectory: the fusion happens at every step, not once at
    the end, so tiles cannot drift apart and no seams can form.

    Handles the three calling conventions of this codebase:
      model(x, t)                      plain diffusion UNet
      model(x, t, coords)              geo-conditioned UNet (coords cropped here)
      model(x, t, (low_res, coords))   transport UNet (low_res cropped from cond)
    """

    def __init__(self, model, tile, overlap, align, batch, geo_full=None):
        self.model = model
        self.tile, self.overlap = tile, overlap
        self.align, self.batch = align, batch
        self.geo_full = geo_full

    def __call__(self, x, t, cond=None):
        lf_full = cond[0] if isinstance(cond, (tuple, list)) else None

        def fn(origins):
            xt = crop_tiles(x, origins, self.tile)
            tb = t[:1].expand(len(origins))
            coords = crop_geo_tiles(self.geo_full, origins, self.tile)
            if lf_full is not None:
                lows = crop_tiles(lf_full, origins, self.tile)
                return self.model(xt, tb, (lows, coords))
            if coords is not None:
                return self.model(xt, tb, coords)
            return self.model(xt, tb)

        return stitch_tiles(fn, x, self.tile, self.overlap, self.align, self.batch)


@torch.no_grad()
def reconstruct_full_fused_diffusion(diffusion, model, lf_full, coarse_full, ratio,
                                     recon_cfg, eta=0.0, tile=128, overlap=32,
                                     batch=8, geo_full=None, project_steps=False,
                                     project_final=True, generator=None):
    """MultiDiffusion-fused guided reconstruction of one full field.

    A single global DDIM chain runs on the full field; every noise prediction
    is fused from overlapping tile evaluations (_FusedTileModel), and the
    per-step ILVR projection (project_steps) acts globally. Compute cost
    matches the tiled mode (all tiles, every step) — only the fusion point
    moves from the end of the chain into every step."""
    K = int(recon_cfg["K"])
    fused = _FusedTileModel(model, tile, overlap, align=ratio, batch=batch,
                            geo_full=geo_full)
    init_noise = _global_noise((K, *lf_full.shape), lf_full, generator)
    out = diffusion.guided_reconstruct(
        fused, lf_full, t_steps=recon_cfg["t_steps"], K=K, eta=eta,
        project=project_steps, lf=coarse_full if project_steps else None,
        ratio=ratio if project_steps else None, init_noise=init_noise)
    return _project_final(out, coarse_full, ratio) if project_final else out


@torch.no_grad()
def reconstruct_full_fused_transport(model, process, lf_full, coarse_full, ratio,
                                     cfg, method, tile=128, overlap=32, batch=8,
                                     geo_full=None, steps=None, solver=None,
                                     sampler=None, stochasticity=None,
                                     project_final=True, project_each=False,
                                     generator=None):
    """MultiDiffusion-fused flow-matching / stochastic-interpolant
    reconstruction: a single global ODE/SDE state, velocity (and score) fused
    from overlapping tile evaluations at every integration step; any per-step
    or final projection acts globally."""
    tc = cfg.get("transport", {})
    fused = _FusedTileModel(model, tile, overlap, align=ratio, batch=batch,
                            geo_full=geo_full)
    noise = _global_noise(lf_full.shape, lf_full, generator)
    kwargs = dict(
        steps=tc.get("sample_steps", 100) if steps is None else steps,
        solver=tc.get("solver", "heun") if solver is None else solver,
        project="each" if project_each else "none",
        coarse=coarse_full if project_each else None,
        ratio=ratio if project_each else None,
        noise=noise,
    )
    if method == "stochastic_interpolant":
        si = tc.get("stochastic_interpolant", {})
        kwargs.update(sampler=si.get("sampler", "ode") if sampler is None else sampler,
                      stochasticity=(si.get("stochasticity", 0.1)
                                     if stochasticity is None else stochasticity))
    out = process.sample(fused, lf_full, coords=None, **kwargs)
    return _project_final(out, coarse_full, ratio) if project_final else out


@torch.no_grad()
def reconstruct_full_tiled_directmap(model, lf_full, tile=128, overlap=32,
                                     batch=8, geo_full=None):
    """Tiled direct-map regression of one full field (deterministic — no noise
    sharing or projection needed)."""
    def fn(origins):
        x = crop_tiles(lf_full, origins, tile)
        coords = crop_geo_tiles(geo_full, origins, tile)
        return model(x, None, coords) if coords is not None else model(x)

    return stitch_tiles(fn, lf_full, tile, overlap, align=1, batch=batch)
