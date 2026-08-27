"""DLWP-HPX backbone (Karlbauer et al. 2024) adapted to super-resolution.

The backbone elements taken from the paper:
  * all convolutions run on the 12 HEALPix faces with cross-face halo padding
    (hpx.padding), so the model sees a seamless sphere;
  * capped GELU activations (GELU clamped at a maximum value, default 10);
  * ConvNeXt-style residual blocks: a 3x3 (dilated) HEALPix convolution
    followed by a pointwise expansion MLP, added to a 1x1-projected skip;
  * a U-Net encoder-decoder over HEALPix resolutions: average-pool 2x
    downsampling (exactly HEALPix nside -> nside/2), transposed-conv
    upsampling, skip connections, and dilation increasing with depth.

Adapted for super-resolution: the model is a spatial map from a degraded t2m
field (upsampled onto the target HPX grid) to the high-resolution field, with
a global residual connection — the temporal/recurrent parts of DLWP-HPX
(GRU blocks) are dropped because the task is single-time-step.

The same backbone doubles as the noise predictor of the sphere-native
diffusion model (models/hpx_diffusion.py): with `use_time=True` a sinusoidal
timestep embedding is projected into every ConvNeXt block (DDPM-style), and
`extra_in_channels` makes room for conditioning channels concatenated to the
noisy input.

All internal tensors are (B*12, C, F, F); the public forward accepts
(B, 12, C, F, F).
"""

import math
import sys
from pathlib import Path

import torch
import torch.utils.checkpoint  # not auto-imported by `import torch` on all versions
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpx.padding import HEALPixPadding  # noqa: E402


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (same convention as the sibling UNet)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class CappedGELU(nn.Module):
    """GELU clamped from above, as used by DLWP-HPX (cap 10)."""

    def __init__(self, cap: float = 10.0):
        super().__init__()
        self.cap = float(cap)
        self.gelu = nn.GELU()

    def forward(self, x):
        return torch.clamp(self.gelu(x), max=self.cap)


class HEALPixConv2d(nn.Module):
    """Conv2d over the 12 faces with cross-face halo padding."""

    def __init__(self, in_ch: int, out_ch: int, face_size: int,
                 kernel_size: int = 3, dilation: int = 1, bias: bool = True):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        pad = dilation * (kernel_size // 2)
        self.pad = HEALPixPadding(face_size, pad) if pad > 0 else None
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              dilation=dilation, bias=bias)

    def forward(self, x):
        if self.pad is not None:
            x = self.pad(x)
        return self.conv(x)


class ConvNeXtBlock(nn.Module):
    """Residual block: 3x3 dilated HPX conv + pointwise expansion MLP.

    out = skip(x) + pw2( act( pw1( act( conv3x3(x) ) + t_emb ) ) )

    The timestep term is added after the spatial convolution (DDPM ResBlock
    convention); with time_emb_dim=0 the block is the plain DLWP-HPX one.
    """

    def __init__(self, face_size: int, in_ch: int, out_ch: int,
                 dilation: int = 1, expansion: int = 4, gelu_cap: float = 10.0,
                 time_emb_dim: int = 0):
        super().__init__()
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, 1))
        self.conv = HEALPixConv2d(in_ch, out_ch, face_size, 3, dilation)
        self.act1 = CappedGELU(gelu_cap)
        self.time_proj = nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        self.pw1 = nn.Conv2d(out_ch, expansion * out_ch, 1)
        self.act2 = CappedGELU(gelu_cap)
        self.pw2 = nn.Conv2d(expansion * out_ch, out_ch, 1)

    def forward(self, x, temb=None):
        h = self.act1(self.conv(x))
        if self.time_proj is not None:
            if temb is None:
                raise ValueError("time-conditioned block called without temb")
            h = h + self.time_proj(temb)[:, :, None, None]
        h = self.pw2(self.act2(self.pw1(h)))
        return self.skip(x) + h


class HEALPixUNetSR(nn.Module):
    """U-Net over HEALPix resolutions.

    Input/output: (B, 12, C, F, F) faces at the target (high-res) nside.
    forward(x) for the deterministic SR model, forward(x, t) when built with
    use_time=True (diffusion noise predictor; t is (B,) in timestep units).
    """

    def __init__(self, nside: int, in_channels: int = 1, out_channels: int = 1,
                 channels=(64, 128, 256), dilations=(1, 2, 4),
                 blocks_per_level: int = 2, expansion: int = 4,
                 gelu_cap: float = 10.0, global_residual: bool = True,
                 grad_checkpoint: bool = False, use_time: bool = False):
        super().__init__()
        # Recompute block activations in backward instead of storing them —
        # needed to fit fine meshes (HPX256) in small GPU memory.
        self.grad_checkpoint = bool(grad_checkpoint)
        depth = len(channels)
        if len(dilations) != depth:
            raise ValueError("dilations must match channels in length")
        sizes = [nside // (2 ** i) for i in range(depth)]
        if sizes[-1] < 2 * dilations[-1]:
            raise ValueError(f"nside {nside} too small for depth {depth} "
                             f"with dilation {dilations[-1]}")
        self.global_residual = bool(global_residual) and in_channels == out_channels

        self.time_base = channels[0]
        time_dim = 4 * channels[0] if use_time else 0
        self.time_mlp = (nn.Sequential(
            nn.Linear(self.time_base, time_dim), CappedGELU(gelu_cap),
            nn.Linear(time_dim, time_dim)) if use_time else None)

        def level(face_size, in_ch, ch, dilation):
            blocks = [ConvNeXtBlock(face_size, in_ch, ch, dilation,
                                    expansion, gelu_cap, time_dim)]
            blocks += [ConvNeXtBlock(face_size, ch, ch, dilation,
                                     expansion, gelu_cap, time_dim)
                       for _ in range(blocks_per_level - 1)]
            return nn.ModuleList(blocks)

        self.encoders = nn.ModuleList(
            level(sizes[i], in_channels if i == 0 else channels[i - 1],
                  channels[i], dilations[i])
            for i in range(depth))
        self.downs = nn.ModuleList(nn.AvgPool2d(2) for _ in range(depth - 1))
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2)
            for i in reversed(range(depth - 1)))
        self.decoders = nn.ModuleList(
            level(sizes[i], 2 * channels[i], channels[i], dilations[i])
            for i in reversed(range(depth - 1)))
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def _blocks(self, blocks, h, temb):
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in blocks:
            if ckpt:
                h = torch.utils.checkpoint.checkpoint(blk, h, temb,
                                                      use_reentrant=False)
            else:
                h = blk(h, temb)
        return h

    def forward(self, x, t=None):
        b, nf, c, fs, _ = x.shape
        h = x.reshape(b * nf, c, fs, fs)
        inp = h
        temb = None
        if self.time_mlp is not None:
            if t is None:
                raise ValueError("time-conditioned model called without t")
            # one embedding per sample, shared by that sample's 12 faces
            temb = self.time_mlp(timestep_embedding(t, self.time_base))
            temb = temb.repeat_interleave(nf, dim=0)
        skips = []
        for i, enc in enumerate(self.encoders):
            h = self._blocks(enc, h, temb)
            if i < len(self.encoders) - 1:
                skips.append(h)
                h = self.downs[i](h)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            h = self._blocks(dec, torch.cat([h, skip], dim=1), temb)
        h = self.head(h)
        if self.global_residual:
            h = h + inp
        return h.reshape(b, nf, -1, fs, fs)


def build_model(cfg: dict, use_time: bool = False,
                extra_in_channels: int = 0) -> HEALPixUNetSR:
    """Build from config.

    `extra_in_channels` widens the input for conditioning channels stacked on
    the noisy field (diffusion); `use_time` adds the timestep embedding. The
    global residual is meaningless for a noise predictor and switches itself
    off whenever the widened input no longer matches the output channels.
    """
    m = cfg["model"]
    return HEALPixUNetSR(
        nside=cfg["hpx"]["nside"],
        in_channels=m["in_channels"] + int(extra_in_channels),
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        dilations=tuple(m["dilations"]),
        blocks_per_level=m["blocks_per_level"],
        expansion=m["expansion"],
        gelu_cap=m["gelu_cap"],
        global_residual=m["global_residual"],
        grad_checkpoint=m.get("grad_checkpoint", False),
        use_time=use_time,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
