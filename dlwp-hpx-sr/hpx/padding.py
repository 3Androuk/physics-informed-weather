"""HEALPix cross-face halo padding — the core of the DLWP-HPX backbone.

Each of the 12 HEALPix faces is convolved as an ordinary 2D grid; before every
convolution the face is padded with a halo of pixels gathered from neighboring
faces so information flows across face boundaries (Karlbauer et al. 2024).

Rather than hard-coding the 12-face adjacency and per-edge rotations, the halo
index maps are DERIVED from the mesh topology (astropy-healpix `neighbours`)
by common-neighbour completion:

  1. Edge strips (halo rings beyond each of the four face edges) are filled
     ring by ring: an unassigned halo cell must be a sphere-neighbour of every
     already-assigned cell grid-adjacent to it, and must not duplicate a pixel
     already placed on this face. This determines every strip cell uniquely.
  2. Corner blocks (pad x pad, beyond each face corner) are filled the same
     way afterwards. At the eight 3-valent mesh vertices no real pixels exist
     beyond the corner ("missing wedge"): those cells stay unresolved and are
     filled at apply time by averaging their already-valid grid neighbours
     (the same treatment DLWP-HPX applies to its missing corners).

The construction is validated in tests/test_hpx.py: every grid adjacency in
the padded maps is checked against the true sphere topology, including the
orthogonal/diagonal distinction (which would catch any sheared or rotated
assignment).
"""

from functools import lru_cache

import numpy as np
import torch
from torch import nn

from .mesh import check_nside, face_index_map, neighbour_table, nest_to_face_xy

_ADJ8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _fill_region(face, cells, used, nbset, size):
    """Common-neighbour completion over `cells`; mutates face and used."""
    pending = [c for c in cells if face[c] < 0]
    changed = True
    while changed and pending:
        changed = False
        rest = []
        for (i, j) in pending:
            ctx = []
            for di, dj in _ADJ8:
                i2, j2 = i + di, j + dj
                if 0 <= i2 < size and 0 <= j2 < size and face[i2, j2] >= 0:
                    ctx.append(int(face[i2, j2]))
            if len(ctx) >= 2:
                cands = set(nbset[ctx[0]])
                for q in ctx[1:]:
                    cands &= nbset[q]
                cands -= used
                if len(cands) == 1:
                    face[i, j] = cands.pop()
                    used.add(int(face[i, j]))
                    changed = True
                    continue
            rest.append((i, j))
        pending = rest


@lru_cache(maxsize=None)
def build_pad_index(nside: int, pad: int) -> np.ndarray:
    """(12, S, S) int64 padded index map, S = nside + 2*pad.

    Entries are nested pixel indices into the flat (12*nside^2,) sphere array;
    -1 marks the unfillable cells at 3-valent corner blocks.
    """
    nside, pad = check_nside(nside), int(pad)
    if not 1 <= pad <= nside:
        raise ValueError(f"pad must be in [1, nside], got pad={pad} nside={nside}")
    nb = neighbour_table(nside)
    n_pix = 12 * nside * nside
    nbset = [frozenset(int(v) for v in nb[:, p] if v >= 0) for p in range(n_pix)]

    S = nside + 2 * pad
    lo, hi = pad, pad + nside  # interior box [lo, hi)
    idx = np.full((12, S, S), -1, dtype=np.int64)
    idx[:, lo:hi, lo:hi] = face_index_map(nside)

    for f in range(12):
        face = idx[f]
        used = set(int(v) for v in face.ravel() if v >= 0)
        # 1) edge strips, ring by ring (corner blocks stay empty until step 2,
        #    so strips can never be contaminated by corner assignments)
        for k in range(1, pad + 1):
            strips = (
                [(lo - k, j) for j in range(lo, hi)]
                + [(hi - 1 + k, j) for j in range(lo, hi)]
                + [(i, lo - k) for i in range(lo, hi)]
                + [(i, hi - 1 + k) for i in range(lo, hi)]
            )
            _fill_region(face, strips, used, nbset, S)
        # 2) corner blocks, cells nearest the face corner first
        corners = []
        for ci, cj in ((0, 0), (0, 1), (1, 0), (1, 1)):
            rows = range(lo - pad, lo) if ci == 0 else range(hi, hi + pad)
            cols = range(lo - pad, lo) if cj == 0 else range(hi, hi + pad)
            block = [(i, j) for i in rows for j in cols]
            block.sort(key=lambda ij: abs(ij[0] - (lo - 0.5 if ci == 0 else hi - 0.5))
                       + abs(ij[1] - (lo - 0.5 if cj == 0 else hi - 0.5)))
            corners += block
        _fill_region(face, corners, used, nbset, S)
        idx[f] = face
    idx.setflags(write=False)  # the array is cached (lru_cache); freeze it
    return idx


@lru_cache(maxsize=None)
def build_fill_generations(nside: int, pad: int):
    """Averaging instructions for the unresolved (-1) corner cells.

    Returns a tuple of generations; each generation is (dst, src, w) where
    dst: (M,) flat indices into the (12*S*S,) padded array to fill,
    src: (M, K) flat indices of the cells averaged into each dst,
    w:   (M, K) float32 averaging weights (rows sum to 1; ragged rows are
         padded by repeating a source with proportionally reduced weight).
    Generation g only references interior/strip cells or generations < g, so
    applying them in order is well-defined.
    """
    idx = build_pad_index(nside, pad)
    S = nside + 2 * pad
    valid = idx >= 0
    generations = []
    missing = [(f, i, j) for f, i, j in zip(*np.nonzero(~valid))]
    while missing:
        gen, rest = [], []
        newly = np.zeros_like(valid)
        for (f, i, j) in missing:
            srcs = []
            for di, dj in _ADJ8:
                i2, j2 = i + di, j + dj
                if 0 <= i2 < S and 0 <= j2 < S and valid[f, i2, j2]:
                    srcs.append((f * S + i2) * S + j2)
            if srcs:
                gen.append(((f * S + i) * S + j, srcs))
                newly[f, i, j] = True
            else:
                rest.append((f, i, j))
        if not gen:
            raise RuntimeError("corner fill did not converge")  # pragma: no cover
        K = max(len(s) for _, s in gen)
        dst = np.array([d for d, _ in gen], dtype=np.int64)
        src = np.zeros((len(gen), K), dtype=np.int64)
        w = np.zeros((len(gen), K), dtype=np.float32)
        for r, (_, s) in enumerate(gen):
            src[r, :len(s)] = s
            src[r, len(s):] = s[0]
            w[r, :len(s)] = 1.0 / len(s)
        generations.append((dst, src, w))
        valid |= newly
        missing = rest
    return tuple(generations)


class HEALPixPadding(nn.Module):
    """Pad the 12 faces with a `pad`-wide halo gathered from neighbor faces.

    Input:  (B*12, C, F, F)  — faces folded into the batch dimension.
    Output: (B*12, C, F+2*pad, F+2*pad)
    """

    def __init__(self, face_size: int, pad: int):
        super().__init__()
        self.face_size, self.pad = int(face_size), int(pad)
        self.S = self.face_size + 2 * self.pad
        idx = build_pad_index(self.face_size, self.pad)
        # The index map holds NESTED pixel indices; the tensor being gathered
        # from is the face array flattened row-major as (face, y, x), so
        # convert. -1 cells are clamped and then overwritten by the fills.
        f, x, y = nest_to_face_xy(np.clip(idx, 0, None), self.face_size)
        rowmajor = (f * self.face_size + y) * self.face_size + x
        self.register_buffer("gather_idx",
                             torch.from_numpy(rowmajor).reshape(-1),
                             persistent=False)
        self.n_fills = 0
        for g, (dst, src, w) in enumerate(build_fill_generations(self.face_size, self.pad)):
            self.register_buffer(f"fill_dst_{g}", torch.from_numpy(dst), persistent=False)
            self.register_buffer(f"fill_src_{g}", torch.from_numpy(src), persistent=False)
            self.register_buffer(f"fill_w_{g}", torch.from_numpy(w), persistent=False)
            self.n_fills += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, fs, _ = x.shape
        if n % 12 or fs != self.face_size:
            raise ValueError(f"expected (B*12, C, {self.face_size}, {self.face_size}), "
                             f"got {tuple(x.shape)}")
        b, S = n // 12, self.S
        flat = x.reshape(b, 12, c, fs * fs).transpose(1, 2).reshape(b, c, 12 * fs * fs)
        out = flat[:, :, self.gather_idx]  # (B, C, 12*S*S)
        for g in range(self.n_fills):
            dst = getattr(self, f"fill_dst_{g}")
            src = getattr(self, f"fill_src_{g}")
            w = getattr(self, f"fill_w_{g}").to(out.dtype)
            # .sum() is on autocast's fp32 list, so under bf16/fp16 autocast the
            # reduction comes back wider than `out`; index_copy needs a match.
            fill = (out[:, :, src] * w).sum(-1).to(out.dtype)
            out = out.index_copy(2, dst, fill)
        return (out.reshape(b, c, 12, S * S).transpose(1, 2)
                .reshape(b * 12, c, S, S))
