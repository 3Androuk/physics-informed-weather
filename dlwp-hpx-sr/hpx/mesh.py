"""HEALPix mesh indexing: nested pixel index <-> (face, x, y) face grids.

The HEALPix sphere tessellation has 12 base faces, each subdivided into an
nside x nside grid of equal-area pixels. In the NESTED ordering, a pixel index
is `face * nside^2 + interleave(x, y)`, so each face is a contiguous 2D array
and coarsening a face 2x2 -> 1 is exactly the HEALPix degrade to nside/2.
This is what lets DLWP-HPX (Karlbauer et al. 2024) run plain conv2d over the
12 faces, with a custom padding exchanging halos between neighboring faces.

Convention used throughout this project: face arrays are indexed [face, y, x]
where x/y are the de-interleaved even/odd bits of the in-face nested index.
"""

import numpy as np


def _spread_bits(x: np.ndarray) -> np.ndarray:
    """Spread the low 32 bits of x so bit i moves to bit 2i (uint64)."""
    x = np.asarray(x, dtype=np.uint64)
    x = (x | (x << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x << np.uint64(2))) & np.uint64(0x3333333333333333)
    x = (x | (x << np.uint64(1))) & np.uint64(0x5555555555555555)
    return x


def _compact_bits(x: np.ndarray) -> np.ndarray:
    """Inverse of _spread_bits: gather the even-position bits of x."""
    x = np.asarray(x, dtype=np.uint64) & np.uint64(0x5555555555555555)
    x = (x | (x >> np.uint64(1))) & np.uint64(0x3333333333333333)
    x = (x | (x >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x | (x >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
    x = (x | (x >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
    x = (x | (x >> np.uint64(16))) & np.uint64(0x00000000FFFFFFFF)
    return x


def check_nside(nside: int) -> int:
    nside = int(nside)
    if nside < 1 or nside & (nside - 1):
        raise ValueError(f"nside must be a power of 2, got {nside}")
    return nside


def npix(nside: int) -> int:
    return 12 * nside * nside


def face_xy_to_nest(face, x, y, nside: int) -> np.ndarray:
    """(face, x, y) -> nested pixel index."""
    nside = check_nside(nside)
    t = _spread_bits(x) | (_spread_bits(y) << np.uint64(1))
    return np.asarray(face, dtype=np.int64) * nside * nside + t.astype(np.int64)


def nest_to_face_xy(pix, nside: int):
    """Nested pixel index -> (face, x, y)."""
    nside = check_nside(nside)
    pix = np.asarray(pix, dtype=np.int64)
    face = pix // (nside * nside)
    t = (pix % (nside * nside)).astype(np.uint64)
    x = _compact_bits(t).astype(np.int64)
    y = _compact_bits(t >> np.uint64(1)).astype(np.int64)
    return face, x, y


def face_index_map(nside: int) -> np.ndarray:
    """(12, nside, nside) int64: nested pixel index at [face, y, x]."""
    nside = check_nside(nside)
    f, y, x = np.meshgrid(np.arange(12), np.arange(nside), np.arange(nside),
                          indexing="ij")
    return face_xy_to_nest(f, x, y, nside)


def nest_to_faces(arr: np.ndarray, nside: int) -> np.ndarray:
    """(..., npix) nested-ordered array -> (..., 12, nside, nside) faces."""
    return arr[..., face_index_map(nside)]


def faces_to_nest(faces: np.ndarray, nside: int) -> np.ndarray:
    """(..., 12, nside, nside) faces -> (..., npix) nested-ordered array."""
    idx = face_index_map(nside)
    out = np.empty(faces.shape[:-3] + (npix(nside),), dtype=faces.dtype)
    out[..., idx.ravel()] = faces.reshape(faces.shape[:-3] + (-1,))
    return out


def pixel_lonlat_deg(nside: int):
    """Pixel-center (lon, lat) in degrees, nested order. lon in [0, 360)."""
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=check_nside(nside), order="nested")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    return lon.deg % 360.0, lat.deg


def neighbour_table(nside: int) -> np.ndarray:
    """(8, npix) neighbour pixel indices in nested order; -1 where missing.

    Slots alternate diagonal/edge-sharing: even slots (0,2,4,6) are the four
    edge-sharing (orthogonal in the face grid) neighbours, odd slots the
    diagonal ones. 24 pixels (the face corners at the eight 3-valent mesh
    vertices) have one missing diagonal neighbour.
    """
    import warnings
    from astropy_healpix import HEALPix
    hp = HEALPix(nside=check_nside(nside), order="nested")
    with warnings.catch_warnings():
        # astropy emits a RuntimeWarning for the 24 pixels with a missing
        # neighbour; that is expected mesh topology.
        warnings.simplefilter("ignore", RuntimeWarning)
        return hp.neighbours(np.arange(hp.npix))
