"""Sphere-aware multiresolution hash-grid geographic encoding.

Encodes static geography (lat, lon, [altitude]) into a learned, multi-resolution
embedding (Müller et al. 2022, "Instant Neural Graphics Primitives" hash grid),
used here as a *positional / conditioning* signal — NOT as the atmosphere
representation itself. The embedding is concatenated as extra input channels to
the diffusion UNet so the noise predictor can exploit location-specific structure
(orography, land-sea contrast, latitude effects) when reconstructing.

Design choices motivated by the atmosphere being a sphere:
  - latitude/longitude are mapped to unit-sphere Cartesian (x, y, z) before
    hashing, so there is no discontinuity at the dateline and no pole
    over-sampling (a raw lat/lon grid would have both).
  - coarse levels index a dense table; fine levels use the spatial hash with a
    bounded table (collisions resolved by the downstream network), exactly as in
    Instant-NGP.

This is a static encoding: one table is shared across all timesteps / dates, so
unlike using a hash grid to represent a (time-varying) field, there is no
per-sample fitting.
"""

import numpy as np
import torch
import torch.nn as nn

# Large primes for the spatial hash (Instant-NGP); pi_1 = 1 by convention.
_PRIMES = [1, 2654435761, 805459861, 3674653429]


def latlon_to_unit_sphere(lat_deg, lon_deg):
    """(lat, lon) in degrees -> unit-sphere Cartesian (x, y, z) in [-1, 1].

    Accepts numpy arrays or torch tensors of matching shape; returns the same
    type, stacked along a new last dimension.
    """
    is_torch = isinstance(lat_deg, torch.Tensor)
    if is_torch:
        lat = torch.deg2rad(lat_deg); lon = torch.deg2rad(lon_deg)
        x = torch.cos(lat) * torch.cos(lon)
        y = torch.cos(lat) * torch.sin(lon)
        z = torch.sin(lat)
        return torch.stack([x, y, z], dim=-1)
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def build_patch_coords(lat_vec, lon_vec, altitude=None):
    """Per-pixel normalized coordinates for one patch.

    Args:
        lat_vec: (H,) latitudes in degrees for the patch rows.
        lon_vec: (W,) longitudes in degrees for the patch cols.
        altitude: optional scalar in [0, 1] appended as a 4th coordinate
            (constant for a single-level variable like Z500).

    Returns:
        (H, W, d) float32 coordinates in [0, 1], d = 3 (or 4 with altitude).
    """
    lat_g, lon_g = np.meshgrid(lat_vec, lon_vec, indexing="ij")  # (H, W)
    xyz = latlon_to_unit_sphere(lat_g, lon_g).astype(np.float32)  # (H, W, 3) in [-1,1]
    coords = (xyz + 1.0) * 0.5                                    # -> [0, 1]
    if altitude is not None:
        h, w = coords.shape[:2]
        alt = np.full((h, w, 1), float(altitude), dtype=np.float32)
        coords = np.concatenate([coords, alt], axis=-1)
    return coords


class MultiResHashGrid(nn.Module):
    """Multiresolution hash-grid encoder for d-dimensional coordinates in [0,1]^d."""

    def __init__(self, input_dim=3, n_levels=12, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()
        assert 1 <= input_dim <= 4, "input_dim must be in [1, 4]"
        self.d = input_dim
        self.L = n_levels
        self.F = n_features_per_level
        self.output_dim = n_levels * n_features_per_level
        T = 2 ** log2_hashmap_size

        if n_levels > 1:
            b = np.exp((np.log(finest_resolution) - np.log(base_resolution)) / (n_levels - 1))
        else:
            b = 1.0

        resolutions, is_dense, tables = [], [], nn.ParameterList()
        for l in range(n_levels):
            n = int(np.floor(base_resolution * (b ** l)))
            cells = (n + 1) ** self.d
            dense = cells <= T
            n_entries = cells if dense else T
            resolutions.append(n)
            is_dense.append(dense)
            tables.append(nn.Parameter(torch.empty(n_entries, self.F).uniform_(-1e-4, 1e-4)))
        self.resolutions = resolutions
        self.is_dense = is_dense
        self.tables = tables

        self.register_buffer("primes", torch.tensor(_PRIMES[: self.d], dtype=torch.long))
        # Corner offsets for d-linear interpolation: (2^d, d).
        corners = torch.tensor(
            [[(c >> i) & 1 for i in range(self.d)] for c in range(2 ** self.d)],
            dtype=torch.long,
        )
        self.register_buffer("corner_offsets", corners)

    def _index(self, corner, n, dense, n_entries):
        """Map integer corner coords (P, d) -> table indices (P,)."""
        if dense:
            stride = 1
            idx = torch.zeros(corner.shape[0], dtype=torch.long, device=corner.device)
            for i in range(self.d):
                idx = idx + corner[:, i] * stride
                stride *= (n + 1)
            return idx
        h = torch.zeros(corner.shape[0], dtype=torch.long, device=corner.device)
        for i in range(self.d):
            h = torch.bitwise_xor(h, corner[:, i] * self.primes[i])
        return h % n_entries

    def forward(self, coords):
        """coords: (..., d) in [0, 1] -> (..., L*F) embedding."""
        lead = coords.shape[:-1]
        x = coords.reshape(-1, self.d).clamp(0.0, 1.0)  # (P, d)
        p = x.shape[0]
        outs = []
        for l in range(self.L):
            n = self.resolutions[l]
            table = self.tables[l]
            n_entries = table.shape[0]
            dense = self.is_dense[l]
            pos = x * n                                  # (P, d) in [0, n]
            base = torch.floor(pos).long()               # (P, d)
            local = pos - base.float()                   # (P, d)
            feat = torch.zeros(p, self.F, device=x.device, dtype=table.dtype)
            for off in self.corner_offsets:              # (d,)
                corner = (base + off).clamp(0, n)        # (P, d)
                w = torch.ones(p, device=x.device, dtype=table.dtype)
                for i in range(self.d):
                    w = w * torch.where(off[i].bool(), local[:, i], 1.0 - local[:, i])
                idx = self._index(corner, n, dense, n_entries)
                feat = feat + w.unsqueeze(-1) * table[idx]
            outs.append(feat)
        out = torch.cat(outs, dim=-1)                    # (P, L*F)
        return out.reshape(*lead, self.output_dim)


class GeoConditionedUNet(nn.Module):
    """Wrap a base UNet to consume a per-pixel geographic embedding.

    The location embedding is encoded from static coordinates and concatenated
    as extra input channels. The noise prediction stays single-channel (it only
    predicts noise on the atmospheric field, never on the conditioning).
    """

    def __init__(self, base_unet: nn.Module, geo_encoder: MultiResHashGrid):
        super().__init__()
        self.unet = base_unet
        self.geo = geo_encoder

    def forward(self, x_t, t, coords):
        """x_t: (B,1,H,W); t: (B,); coords: (B,H,W,d) in [0,1]."""
        emb = self.geo(coords)                       # (B, H, W, E)
        emb = emb.permute(0, 3, 1, 2).contiguous()   # (B, E, H, W)
        return self.unet(torch.cat([x_t, emb], dim=1), t)


def build_geo_encoder(cfg: dict) -> MultiResHashGrid:
    g = cfg["geo"]
    return MultiResHashGrid(
        input_dim=g["input_dim"],
        n_levels=g["n_levels"],
        n_features_per_level=g["n_features_per_level"],
        log2_hashmap_size=g["log2_hashmap_size"],
        base_resolution=g["base_resolution"],
        finest_resolution=g["finest_resolution"],
    )
