"""Remapping between regular lat-lon grids and the HEALPix mesh.

lat-lon -> HEALPix: bilinear interpolation of the source grid at HEALPix
pixel centers, with periodic longitude handling and latitude clamped to the
grid's own range (cell-centered grids like WB2's conservative stores do not
include the poles; HEALPix pixels polewards of the last row reuse it).

HEALPix -> lat-lon (for figures/eval only): astropy-healpix spherical
bilinear interpolation from the 4 nearest HEALPix pixels.
"""

import numpy as np

from .mesh import check_nside, faces_to_nest, nest_to_faces, pixel_lonlat_deg


class LatLonToHPX:
    """Precomputed bilinear remap from a fixed (lat, lon) grid to HPX faces."""

    def __init__(self, lat: np.ndarray, lon: np.ndarray, nside: int):
        self.nside = check_nside(nside)
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
            raise ValueError("lat and lon must be strictly ascending")
        self.nlat, self.nlon = len(lat), len(lon)

        plon, plat = pixel_lonlat_deg(self.nside)

        # latitude: clamp outside the grid (no pole rows on cell-centered grids)
        plat = np.clip(plat, lat[0], lat[-1])
        iy1 = np.clip(np.searchsorted(lat, plat), 1, self.nlat - 1)
        iy0 = iy1 - 1
        wy = (plat - lat[iy0]) / (lat[iy1] - lat[iy0])

        # longitude: periodic
        lon0 = lon[0]
        period = 360.0
        step_last = (lon0 + period) - lon[-1]
        p = (plon - lon0) % period
        ix1 = np.searchsorted(lon - lon0, p)  # in [0, nlon]
        ix0 = (ix1 - 1) % self.nlon
        wrap = (ix1 == 0) | (ix1 == self.nlon)
        wx = np.where(
            wrap,
            ((p - (lon[-1] - lon0)) % period) / step_last,
            (p - (lon - lon0)[np.clip(ix1 - 1, 0, self.nlon - 1)])
            / np.diff(lon)[np.clip(ix1 - 1, 0, self.nlon - 2)],
        )
        ix1 = ix1 % self.nlon

        self.iy0, self.iy1 = iy0, iy1
        self.ix0, self.ix1 = ix0, ix1
        self.w00 = ((1 - wy) * (1 - wx)).astype(np.float32)
        self.w01 = ((1 - wy) * wx).astype(np.float32)
        self.w10 = (wy * (1 - wx)).astype(np.float32)
        self.w11 = (wy * wx).astype(np.float32)

    def __call__(self, fields: np.ndarray) -> np.ndarray:
        """(..., nlat, nlon) -> (..., 12, nside, nside) float32 faces."""
        f = np.asarray(fields)
        if f.shape[-2:] != (self.nlat, self.nlon):
            raise ValueError(f"expected trailing dims ({self.nlat}, {self.nlon}), "
                             f"got {f.shape[-2:]}")
        vals = (f[..., self.iy0, self.ix0] * self.w00
                + f[..., self.iy0, self.ix1] * self.w01
                + f[..., self.iy1, self.ix0] * self.w10
                + f[..., self.iy1, self.ix1] * self.w11).astype(np.float32)
        return nest_to_faces(vals, self.nside)


def hpx_to_latlon(faces: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """(..., 12, F, F) faces -> (..., nlat, nlon) via spherical bilinear interp."""
    import astropy.units as u
    from astropy_healpix import HEALPix

    faces = np.asarray(faces)
    nside = faces.shape[-1]
    hp = HEALPix(nside=check_nside(nside), order="nested")
    lon2, lat2 = np.meshgrid(np.asarray(lon), np.asarray(lat))
    flat = faces_to_nest(faces, nside).reshape(-1, hp.npix)
    out = np.stack([
        hp.interpolate_bilinear_lonlat(lon2 * u.deg, lat2 * u.deg, m)
        for m in flat
    ]).astype(np.float32)
    return out.reshape(faces.shape[:-3] + lon2.shape)
