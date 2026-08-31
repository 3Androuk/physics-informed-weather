"""Hydrostatic (hypsometric) constraint applied during sampling.

Inference-only. See eval/hydrostatic.py for the diagnostic that motivated it,
and note the measured caveat: ERA5's OWN residual is ~41 m2/s2 (850-700) and
~63 (700-500), because Tv_bar is approximated by a two-level mean. Driving the
residual to zero therefore moves AWAY from the data's own balance, so coef is
exposed as a sweepable knob rather than fixed at 1.
"""

import torch

R_D = 287.05
EPS_Q = 0.608
LAYERS = ((850, 700), (700, 500))


def channel_index(cfg):
    idx = {}
    for c, v in enumerate(cfg["data"]["variables"]):
        key = v["name"] if v.get("level") is None else f"{v['name']}@{v['level']}"
        idx[key] = c
    return idx


class HydrostaticProjection:
    """Least-norm correction toward hypsometric balance, in normalized units."""

    def __init__(self, cfg, normalizer, coef=1.0, sweeps=2, device="cpu"):
        idx = channel_index(cfg)
        self.coef = float(coef)
        self.sweeps = int(sweeps)
        self.ch = {}
        for lvl in (850, 700, 500):
            self.ch[lvl] = (idx[f"geopotential@{lvl}"],
                            idx[f"temperature@{lvl}"],
                            idx[f"specific_humidity@{lvl}"])
        m, s = normalizer.mean, normalizer.std          # (C,1,1)
        self.mean = m.to(device).reshape(-1)
        self.std = s.to(device).reshape(-1)

    def _decode(self, x0, c):
        return x0[:, c] * self.std[c] + self.mean[c]

    def _layer(self, x0, p_lo, p_up):
        zl, tl, ql = self.ch[p_lo]
        zu, tu, qu = self.ch[p_up]
        L = torch.log(torch.tensor(p_lo / p_up, device=x0.device, dtype=x0.dtype))
        Tl, Ql = self._decode(x0, tl), self._decode(x0, ql)
        Tu, Qu = self._decode(x0, tu), self._decode(x0, qu)
        R = ((self._decode(x0, zu) - self._decode(x0, zl))
             - R_D * L * 0.5 * (Tl * (1 + EPS_Q * Ql) + Tu * (1 + EPS_Q * Qu)))

        k = R_D * L * 0.5
        # dR/dx in PHYSICAL units, then * std -> dR/dx in normalized units
        b = {
            zu: torch.ones_like(R) * self.std[zu],
            zl: -torch.ones_like(R) * self.std[zl],
            tl: -k * (1 + EPS_Q * Ql) * self.std[tl],
            tu: -k * (1 + EPS_Q * Qu) * self.std[tu],
            ql: -k * EPS_Q * Tl * self.std[ql],
            qu: -k * EPS_Q * Tu * self.std[qu],
        }
        denom = sum(v ** 2 for v in b.values()).clamp(min=1e-12)
        step = self.coef * R / denom
        out = x0.clone()
        for c, bc in b.items():
            out[:, c] = x0[:, c] - step * bc
        return out

    @torch.no_grad()
    def __call__(self, x0):
        if self.coef == 0.0:
            return x0
        for _ in range(self.sweeps):       # the layers share 700 hPa
            for p_lo, p_up in LAYERS:
                x0 = self._layer(x0, p_lo, p_up)
        return x0
