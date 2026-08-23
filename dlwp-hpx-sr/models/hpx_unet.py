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

All internal tensors are (B*12, C, F, F); the public forward accepts
(B, 12, C, F, F).
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpx.padding import HEALPixPadding  # noqa: E402


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

    out = skip(x) + pw2( act( pw1( act( conv3x3(x) ) ) ) )
    """

    def __init__(self, face_size: int, in_ch: int, out_ch: int,
                 dilation: int = 1, expansion: int = 4, gelu_cap: float = 10.0):
        super().__init__()
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, 1))
        self.conv = HEALPixConv2d(in_ch, out_ch, face_size, 3, dilation)
        self.act1 = CappedGELU(gelu_cap)
        self.pw1 = nn.Conv2d(out_ch, expansion * out_ch, 1)
        self.act2 = CappedGELU(gelu_cap)
        self.pw2 = nn.Conv2d(expansion * out_ch, out_ch, 1)

    def forward(self, x):
        h = self.act1(self.conv(x))
        h = self.pw2(self.act2(self.pw1(h)))
        return self.skip(x) + h


class HEALPixUNetSR(nn.Module):
    """U-Net over HEALPix resolutions for super-resolution.

    Input/output: (B, 12, C, F, F) faces at the target (high-res) nside.
    """

    def __init__(self, nside: int, in_channels: int = 1, out_channels: int = 1,
                 channels=(64, 128, 256), dilations=(1, 2, 4),
                 blocks_per_level: int = 2, expansion: int = 4,
                 gelu_cap: float = 10.0, global_residual: bool = True):
        super().__init__()
        depth = len(channels)
        if len(dilations) != depth:
            raise ValueError("dilations must match channels in length")
        sizes = [nside // (2 ** i) for i in range(depth)]
        if sizes[-1] < 2 * dilations[-1]:
            raise ValueError(f"nside {nside} too small for depth {depth} "
                             f"with dilation {dilations[-1]}")
        self.global_residual = bool(global_residual) and in_channels == out_channels

        def level(face_size, in_ch, ch, dilation):
            blocks = [ConvNeXtBlock(face_size, in_ch, ch, dilation,
                                    expansion, gelu_cap)]
            blocks += [ConvNeXtBlock(face_size, ch, ch, dilation,
                                     expansion, gelu_cap)
                       for _ in range(blocks_per_level - 1)]
            return nn.Sequential(*blocks)

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

    def forward(self, x):
        b, nf, c, fs, _ = x.shape
        h = x.reshape(b * nf, c, fs, fs)
        inp = h
        skips = []
        for i, enc in enumerate(self.encoders):
            h = enc(h)
            if i < len(self.encoders) - 1:
                skips.append(h)
                h = self.downs[i](h)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            h = dec(torch.cat([h, skip], dim=1))
        h = self.head(h)
        if self.global_residual:
            h = h + inp
        return h.reshape(b, nf, -1, fs, fs)


def build_model(cfg: dict) -> HEALPixUNetSR:
    m = cfg["model"]
    return HEALPixUNetSR(
        nside=cfg["hpx"]["nside"],
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        dilations=tuple(m["dilations"]),
        blocks_per_level=m["blocks_per_level"],
        expansion=m["expansion"],
        gelu_cap=m["gelu_cap"],
        global_residual=m["global_residual"],
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
