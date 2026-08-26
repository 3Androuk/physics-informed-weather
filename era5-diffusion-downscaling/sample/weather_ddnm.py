"""Weather-covariance-aware DDNM projection for patch super-resolution.

For the block-average observation operator ``A`` and a stationary covariance
``C`` represented by a high-resolution power spectrum, this module applies

    K_C = C A^T (A C A^T)^-1

without forming a dense matrix.  Periodic covariance and aligned block
averaging make ``A C A^T`` a convolution on the coarse grid, so its inverse is
an elementwise division in coarse Fourier space.  The result is an exact
coarse-consistency projection up to floating-point roundoff; a final tiny
ordinary correction removes that roundoff.

The implementation is intentionally inference-only.  It does not change or
retrain the diffusion prior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.degrade import coarsen, upsample_nearest


def _gaspari_cohn(radius: torch.Tensor) -> torch.Tensor:
    """Gaspari-Cohn (1999) 5th-order correlation function, support on [0, 2].

    Positive definite and compactly supported — the standard localization
    kernel in ensemble data assimilation."""
    r = radius.clamp_min(0.0)
    r_safe = r.clamp_min(1e-12)
    near = (1.0 - 0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - (5.0 / 3.0) * r**2)
    far = ((1.0 / 12.0) * r**5 - 0.5 * r**4 + 0.625 * r**3
           + (5.0 / 3.0) * r**2 - 5.0 * r + 4.0 - 2.0 / (3.0 * r_safe))
    # Support is [0, 2): forcing the tail to exact zero keeps the tapered
    # kernel exactly compact instead of leaving polynomial roundoff behind.
    out = torch.where(r <= 1.0, near, torch.where(r < 2.0, far,
                                                  torch.zeros_like(r)))
    return out.clamp_min(0.0)


def localize_spectrum(power, image_size, radius: float, floor: float = 1e-6):
    """Taper a stationary covariance to compact spatial support.

    The projector applies ``C`` as a CIRCULAR convolution, but weather patches
    are not periodic: an untapered covariance with correlation length
    comparable to the patch spreads the data-consistency correction across the
    wrap-around edge. Multiplying the covariance KERNEL by a Gaspari-Cohn
    window of half-width ``radius`` pixels (support 2*radius) confines the
    correction to a physical neighborhood, so the periodic embedding becomes
    harmless. Because Gaspari-Cohn is itself positive definite, the Schur
    product theorem keeps the tapered covariance positive definite — the exact
    FFT algebra downstream is unchanged.

    Args:
        power: (C, H, W//2+1) covariance eigenvalues (rFFT layout).
        image_size: (H, W) of the high-resolution grid.
        radius: localization half-width in pixels; the kernel vanishes beyond
            2*radius. Values >= max(H, W) / 2 leave the covariance essentially
            untouched (no wrap to suppress).
        floor: relative floor re-applied after tapering (tapering can push
            eigenvalues to ~0 through roundoff).
    Returns:
        (C, H, W//2+1) tapered eigenvalues, finite and strictly positive.
    """
    power = torch.as_tensor(power, dtype=torch.float32)
    if power.ndim == 2:
        power = power.unsqueeze(0)
    h, w = int(image_size[0]), int(image_size[1])
    if radius <= 0:
        raise ValueError("localization radius must be positive")
    # Toroidal distance from the kernel origin keeps the taper circulant.
    dy = torch.arange(h).float()
    dx = torch.arange(w).float()
    dy = torch.minimum(dy, h - dy)
    dx = torch.minimum(dx, w - dx)
    dist = torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2)
    window = _gaspari_cohn(dist / float(radius))

    kernel = torch.fft.irfft2(power, s=(h, w)) * window
    tapered = torch.fft.rfft2(kernel).real
    scale = tapered.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
    return tapered.clamp_min(scale * floor)


class SpectralCovarianceProjector:
    """Single- or diagonal-multichannel stationary covariance projector.

    Args:
        power: covariance eigenvalues with shape ``(C,H,W//2+1)`` or
            ``(H,W//2+1)`` for one channel. Values must be finite and positive.
        image_size: ``(H,W)`` of the high-resolution patches.
        inverse_floor: relative floor applied to eigenvalues of ``A C A^T``.
            It is only a numerical safeguard; the saved covariance itself is
            expected to have already been shrunk/floored during estimation.

    This class supports a diagonal channel covariance.  A full cross-channel
    spectral covariance can be added later by replacing scalar Fourier
    multiplication/division with small Hermitian matrix operations per mode.
    T2M uses one channel, so the current implementation is exact for this task.
    """

    def __init__(self, power, image_size, inverse_floor: float = 1e-7):
        power = torch.as_tensor(power, dtype=torch.float32)
        if power.ndim == 2:
            power = power.unsqueeze(0)
        if power.ndim != 3:
            raise ValueError("power must have shape (C,H,W//2+1) or (H,W//2+1)")
        h, w = (int(image_size[0]), int(image_size[1]))
        if tuple(power.shape[-2:]) != (h, w // 2 + 1):
            raise ValueError(
                f"power shape {tuple(power.shape)} is incompatible with image size {(h, w)}")
        if not torch.isfinite(power).all() or (power <= 0).any():
            raise ValueError("spectral covariance must be finite and strictly positive")
        if inverse_floor <= 0:
            raise ValueError("inverse_floor must be positive")
        self.power = power
        self.image_size = (h, w)
        self.channels = power.shape[0]
        self.inverse_floor = float(inverse_floor)
        self._coarse_spectra = {}

    @classmethod
    def from_npz(cls, path, inverse_floor: float = 1e-7,
                 localization_radius: float | None = None):
        """Load the artifact written by ``data.estimate_spectral_covariance``.

        ``localization_radius`` (pixels) applies the Gaspari-Cohn taper at load
        time, so an artifact estimated without localization can still be used
        with it (and the setting ablated) without re-estimating."""
        path = Path(path)
        with np.load(path) as data:
            power = np.array(data["power"], dtype=np.float32)
            image_size = tuple(int(v) for v in data["image_size"])
        if localization_radius is not None:
            power = localize_spectrum(power, image_size, localization_radius)
        return cls(power, image_size, inverse_floor=inverse_floor)

    def to(self, device=None, dtype=None):
        """Move the saved spectrum; cached coarse spectra are rebuilt lazily."""
        dtype = dtype or self.power.dtype
        self.power = self.power.to(device=device, dtype=dtype)
        self._coarse_spectra.clear()
        return self

    def _check(self, x: torch.Tensor, ratio: int):
        if x.ndim != 4:
            raise ValueError("expected tensors with shape (N,C,H,W)")
        if tuple(x.shape[-2:]) != self.image_size:
            raise ValueError(
                f"tensor grid {tuple(x.shape[-2:])} != covariance grid {self.image_size}")
        if x.shape[1] != self.channels:
            raise ValueError(f"tensor has {x.shape[1]} channels; covariance has {self.channels}")
        h, w = self.image_size
        if ratio < 1 or h % ratio or w % ratio:
            raise ValueError(f"ratio {ratio} must divide covariance grid {(h, w)}")

    def apply_covariance(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the periodic stationary covariance ``C`` to a HR field."""
        self._check(x, 1)
        power = self.power.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        freq = torch.fft.rfft2(x)
        return torch.fft.irfft2(freq * power, s=self.image_size)

    @staticmethod
    def _adjoint(coarse: torch.Tensor, ratio: int, image_size) -> torch.Tensor:
        """Adjoint of non-overlapping average pooling: repeat then divide r^2."""
        return upsample_nearest(coarse, image_size) / float(ratio * ratio)

    def _coarse_spectrum(self, ratio: int, device, dtype):
        """Fourier eigenvalues of ``A C A^T`` for an aligned coarse grid."""
        key = (ratio, str(device), dtype)
        cached = self._coarse_spectra.get(key)
        if cached is not None:
            return cached

        h, w = self.image_size
        hc, wc = h // ratio, w // ratio
        impulse = torch.zeros(1, self.channels, hc, wc, device=device, dtype=dtype)
        impulse[:, :, 0, 0] = 1.0
        response = coarsen(
            self.apply_covariance(self._adjoint(impulse, ratio, self.image_size)), ratio)

        # A C A^T is real symmetric positive definite.  Its circulant
        # eigenvalues are the non-normalized FFT of its impulse response.
        spectrum = torch.fft.rfft2(response)[0].real
        scale = spectrum.amax(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(dtype).tiny)
        spectrum = spectrum.clamp_min(scale * self.inverse_floor)
        self._coarse_spectra[key] = spectrum
        return spectrum

    def correction(self, residual: torch.Tensor, ratio: int) -> torch.Tensor:
        """Return ``K_C residual`` where residual lives on the coarse grid."""
        h, w = self.image_size
        expected = (h // ratio, w // ratio)
        if tuple(residual.shape[-2:]) != expected:
            raise ValueError(
                f"coarse residual grid {tuple(residual.shape[-2:])} != {expected}")
        if residual.shape[1] != self.channels:
            raise ValueError("coarse residual channel count does not match covariance")
        denom = self._coarse_spectrum(ratio, residual.device, residual.dtype).unsqueeze(0)
        dual = torch.fft.irfft2(
            torch.fft.rfft2(residual) / denom, s=expected)
        return self.apply_covariance(self._adjoint(dual, ratio, self.image_size))

    def project(self, estimate: torch.Tensor, observation: torch.Tensor,
                ratio: int) -> torch.Tensor:
        """Project a HR estimate onto ``coarsen(x, ratio) == observation``."""
        self._check(estimate, ratio)
        residual = observation - coarsen(estimate, ratio)
        out = estimate + self.correction(residual, ratio)

        # Remove FFT roundoff so downstream checks see exact consistency.  This
        # term should be many orders smaller than the covariance correction.
        cleanup = observation - coarsen(out, ratio)
        return out + upsample_nearest(cleanup, self.image_size)

    def lift(self, observation: torch.Tensor, ratio: int) -> torch.Tensor:
        """Covariance conditional mean ``K_C y`` for a zero-mean HR prior."""
        h, w = self.image_size
        zeros = torch.zeros(
            observation.shape[0], observation.shape[1], h, w,
            device=observation.device, dtype=observation.dtype)
        return self.project(zeros, observation, ratio)
