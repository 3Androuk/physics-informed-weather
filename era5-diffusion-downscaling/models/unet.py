"""UNet noise estimator for the diffusion model (Ronneberger et al. 2015).

Hierarchical conv blocks with multi-level skip connections and self-attention at
the bottleneck (coarsest) resolution, as described in the paper. The same
architecture serves the direct-mapping baseline with ``use_time=False`` (no
timestep conditioning, plain f: X -> Y regression).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (Transformer-style)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = (
            nn.Linear(time_emb_dim, out_ch) if time_emb_dim else None
        )
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.time_proj is not None and t_emb is not None:
            h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention over (H*W) positions."""

    def __init__(self, ch, groups):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x):
        n, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(n, c, h * w).permute(0, 2, 1)   # (N, HW, C)
        k = k.reshape(n, c, h * w)                     # (N, C, HW)
        v = v.reshape(n, c, h * w).permute(0, 2, 1)    # (N, HW, C)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)  # (N, HW, HW)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(n, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        attn_resolutions=(16,),
        dropout: float = 0.1,
        groupnorm_groups: int = 32,
        use_time: bool = True,
    ):
        super().__init__()
        self.use_time = use_time
        temb_dim = time_emb_dim if use_time else 0
        self.time_emb_dim = time_emb_dim

        if use_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.SiLU(),
                nn.Linear(time_emb_dim * 4, time_emb_dim),
            )

        attn_resolutions = set(attn_resolutions)
        g, drop = groupnorm_groups, dropout

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── Encoder ──
        self.down_blocks = nn.ModuleList()
        skip_chs = [base_channels]
        ch = base_channels
        res = image_size
        n_levels = len(channel_mults)
        for lvl, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(ch, out_ch, temb_dim, g, drop)])
                ch = out_ch
                if res in attn_resolutions:
                    block.append(AttnBlock(ch, g))
                self.down_blocks.append(block)
                skip_chs.append(ch)
            if lvl != n_levels - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)
                res //= 2

        # ── Bottleneck (self-attention here, coarsest resolution) ──
        self.mid = nn.ModuleList([
            ResBlock(ch, ch, temb_dim, g, drop),
            AttnBlock(ch, g),
            ResBlock(ch, ch, temb_dim, g, drop),
        ])

        # ── Decoder ──
        self.up_blocks = nn.ModuleList()
        for lvl, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                block = nn.ModuleList([
                    ResBlock(ch + skip_chs.pop(), out_ch, temb_dim, g, drop)
                ])
                ch = out_ch
                if res in attn_resolutions:
                    block.append(AttnBlock(ch, g))
                self.up_blocks.append(block)
            if lvl != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                res *= 2

        self.norm_out = nn.GroupNorm(min(g, ch), ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_time:
            assert t is not None, "timestep t required when use_time=True"
            t_emb = self.time_mlp(timestep_embedding(t, self.time_emb_dim))
        else:
            t_emb = None

        h = self.conv_in(x)
        skips = [h]
        for block in self.down_blocks:
            if isinstance(block[0], Downsample):
                h = block[0](h)
            else:
                h = block[0](h, t_emb)
                for layer in block[1:]:
                    h = layer(h)
            skips.append(h)

        h = self.mid[0](h, t_emb)
        h = self.mid[1](h)
        h = self.mid[2](h, t_emb)

        for block in self.up_blocks:
            if isinstance(block[0], Upsample):
                h = block[0](h)
            else:
                h = torch.cat([h, skips.pop()], dim=1)
                h = block[0](h, t_emb)
                for layer in block[1:]:
                    h = layer(h)

        return self.conv_out(F.silu(self.norm_out(h)))


def build_unet(cfg: dict, use_time: bool = True, extra_in_channels: int = 0) -> UNet:
    u = cfg["unet"]
    return UNet(
        image_size=cfg["patches"]["size"],
        in_channels=u["in_channels"] + extra_in_channels,
        out_channels=u["out_channels"],
        base_channels=u["base_channels"],
        channel_mults=tuple(u["channel_mults"]),
        num_res_blocks=u["num_res_blocks"],
        time_emb_dim=u["time_emb_dim"],
        attn_resolutions=tuple(u["attn_resolutions"]),
        dropout=u["dropout"],
        groupnorm_groups=u["groupnorm_groups"],
        use_time=use_time,
    )
