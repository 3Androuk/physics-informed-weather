"""Hydrostatic (hypsometric) balance as a physics-consistency diagnostic.

Shu et al.'s physics residual is a time-evolution PDE; these patches are single
snapshots with no time pairing, so any residual containing d/dt is unavailable.
What remains are DIAGNOSTIC balances that must hold within one instantaneous
state, and the strongest of those is hydrostatic balance.

In pressure coordinates dPhi/dln p = -R_d T_v, which integrates over a layer to
the hypsometric relation

    Phi_upper - Phi_lower = R_d * Tv_bar * ln(p_lower / p_upper)

The 20-variable set carries z, t and q at 850/700/500, so two independent
residuals per pixel are available, and because q is present the proper virtual
temperature T_v = T (1 + 0.608 q) can be used rather than approximated away.

Unlike the alternatives this is non-singular everywhere and purely local:
geostrophic and thermal-wind balances both divide by f = 2 Omega sin(phi),
which vanishes inside the +-60 deg crop; continuity needs vertical velocity,
which is not among the 20 channels.

Reported per layer:
  rmse        RMS residual, m2/s2 (same units as geopotential)
  rel         RMS residual as a fraction of the layer thickness
  bias        mean signed residual

ERA5's own residual is the floor: it is nonzero because Tv_bar is approximated
by the two-level mean rather than the true integral over ln p. A reconstruction
that sits at ERA5's floor is as balanced as the data allows, and hydrostatic
guidance would have nothing to fix.

Run:
    python -m eval.hydrostatic --config config/wb2_20var.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from utils import ensure_dir, get_device, load_config  # noqa: E402

R_D = 287.05          # J kg-1 K-1, dry air
EPS_Q = 0.608         # virtual temperature coefficient
LAYERS = ((850, 700), (700, 500))


def channel_index(cfg):
    """name@level -> channel, from data.variables (the canonical channel order)."""
    idx = {}
    for c, v in enumerate(cfg["data"]["variables"]):
        key = v["name"] if v.get("level") is None else f"{v['name']}@{v['level']}"
        idx[key] = c
    return idx


def virtual_temperature(t, q):
    return t * (1.0 + EPS_Q * q)


def hydrostatic_residual(fields, idx, p_lower, p_upper):
    """Residual of the hypsometric relation for one layer, in m2/s2.

    `fields` is (N, C, H, W) in PHYSICAL units: geopotential m2/s2, T in K,
    q in kg/kg. Positive residual = the geopotential thickness exceeds what the
    layer's virtual temperature supports.
    """
    def ch(name, lvl):
        return fields[:, idx[f"{name}@{lvl}"]]

    phi_l, phi_u = ch("geopotential", p_lower), ch("geopotential", p_upper)
    tv_l = virtual_temperature(ch("temperature", p_lower), ch("specific_humidity", p_lower))
    tv_u = virtual_temperature(ch("temperature", p_upper), ch("specific_humidity", p_upper))
    tv_bar = 0.5 * (tv_l + tv_u)
    thickness = phi_u - phi_l
    return thickness - R_D * tv_bar * np.log(p_lower / p_upper), thickness


def residual_fields(fields, idx):
    """Per-layer residual field (N, H, W), in m2/s2."""
    return {f"{lo}-{up}": hydrostatic_residual(fields, idx, lo, up)[0]
            for lo, up in LAYERS}


def compare_to_truth(r_pred, r_truth):
    """Is the balance REPRODUCED, or merely of similar magnitude?"""
    out = {}
    for k in r_pred:
        d = r_pred[k] - r_truth[k]
        a, b = r_pred[k].ravel(), r_truth[k].ravel()
        out[k] = {
            "discrepancy_rmse": float(np.sqrt((d ** 2).mean())),
            "corr_with_truth": float(np.corrcoef(a, b)[0, 1]),
            "truth_rmse": float(np.sqrt((b ** 2).mean())),
        }
    return out


def score(fields, idx):
    out = {}
    for p_lo, p_up in LAYERS:
        r, thick = hydrostatic_residual(fields, idx, p_lo, p_up)
        out[f"{p_lo}-{p_up}"] = {
            "rmse": float(np.sqrt((r ** 2).mean())),
            "bias": float(r.mean()),
            "rel": float(np.sqrt((r ** 2).mean()) / np.abs(thick).mean()),
            "thickness_mean": float(thick.mean()),
        }
    return out


def bicubic(x, ratio):
    lo = F.avg_pool2d(x, ratio)
    return F.interpolate(lo, size=x.shape[-2:], mode="bicubic", align_corners=False)


def nearest(x, ratio):
    lo = F.avg_pool2d(x, ratio)
    return F.interpolate(lo, size=x.shape[-2:], mode="nearest")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/wb2_20var.yaml")
    ap.add_argument("--n-patches", type=int, default=256)
    ap.add_argument("--ratios", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--ckpts", nargs="+", default=None,
                    help="also reconstruct with these diffusion checkpoints "
                         "(names in paths.ckpt_dir) and score their residual. "
                         "Forces first-N patch indexing, which is what the geo "
                         "payload builder assumes.")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no-project", action="store_true",
                    help="reconstruct WITHOUT the DDNM projection. Tests whether the\n                         projection is what delivers hydrostatic balance: it pins the\n                         coarse scales of all six channels, and the balance is a\n                         large-scale relation.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    idx = channel_index(cfg)
    need = [f"{n}@{l}" for l in (850, 700, 500)
            for n in ("geopotential", "temperature", "specific_humidity")]
    missing = [k for k in need if k not in idx]
    if missing:
        raise SystemExit(f"config lacks required channels: {missing}")

    patch_dir = Path(cfg["paths"]["patch_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])
    normalizer = load_norm_stats(patch_dir)
    ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    n = min(args.n_patches, len(ds))
    # The geo payload builder stacks the FIRST n patches, so a model run must
    # index the same way; without checkpoints we spread over the test period.
    sel = (np.arange(n) if args.ckpts
           else np.linspace(0, len(ds) - 1, n).astype(int))
    x = torch.stack([ds[int(i)] for i in sel])
    truth = normalizer.decode(x)
    print(f"{n} test patches | channels {truth.shape[1]} | "
          f"layers {['%d-%d' % l for l in LAYERS]}")

    truth_np = truth.numpy()
    r_truth_fields = residual_fields(truth_np, idx)
    out = {"n_patches": n, "era5_truth": score(truth_np, idx), "degraded": {}}
    print("\n=== ERA5 truth (the floor: two-level Tv_bar approximation) ===")
    for k, v in out["era5_truth"].items():
        print(f"  {k} hPa: rmse {v['rmse']:9.2f} m2/s2 | {100*v['rel']:5.2f}% of "
              f"thickness {v['thickness_mean']:.0f} | bias {v['bias']:+.2f}")

    for r in args.ratios:
        out["degraded"][f"{r}x"] = {}
        print(f"\n=== degraded {r}x, reconstructed without a model ===")
        for name, fn in (("bicubic", bicubic), ("nearest", nearest)):
            rec = normalizer.decode(fn(x, r)).numpy()
            s = score(rec, idx)
            cmp = compare_to_truth(residual_fields(rec, idx), r_truth_fields)
            for k in s:
                s[k].update(cmp[k])
            out["degraded"][f"{r}x"][name] = s
            for k, v in s.items():
                print(f"  {name:8s} {k} hPa: rmse {v['rmse']:9.2f} | "
                      f"discrepancy {v['discrepancy_rmse']:8.2f} | "
                      f"corr {v['corr_with_truth']:+.4f}")

    if args.ckpts:
        import torch as _t
        from eval.compare_geo import _payload, _recon
        from sample.reconstruct import load_diffusion
        device = get_device()
        ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
        eta = cfg["sample"]["ddim_eta"]
        recons = {int(rc["ratio"]): rc for rc in cfg["sample"]["reconstructions"]}
        hf = x.to(device)
        out["models"] = {}
        out["projection"] = not args.no_project
        print(f"\nprojection: {'ON' if not args.no_project else 'OFF (ablation)'}")
        for name in args.ckpts:
            mdl, dif, cfg_ck = load_diffusion(ckpt_dir / name, device)
            gcfg = cfg_ck.get("geo", {})
            coords = (_payload(patch_dir, normalizer, n, gcfg, device)
                      if gcfg.get("enabled", False) else None)
            out["models"][name] = {}
            print(f"\n=== {name} (geo={gcfg.get('enabled', False)}, "
                  f"encoder={gcfg.get('encoder', '-')}) ===")
            for r in args.ratios:
                if r not in recons:
                    continue
                rec = _recon(dif, mdl, hf, r, recons[r], eta, coords, args.batch,
                             label=f"{name} {r}x",
                             project=not args.no_project)
                rec_np = normalizer.decode(rec).numpy()
                sc = score(rec_np, idx)
                cmp = compare_to_truth(residual_fields(rec_np, idx), r_truth_fields)
                for k in sc:
                    sc[k].update(cmp[k])
                out["models"][name][f"{r}x"] = sc
                for k, v in sc.items():
                    print(f"  {r}x {k} hPa: rmse {v['rmse']:9.2f} | "
                          f"discrepancy {v['discrepancy_rmse']:8.2f} | "
                          f"corr {v['corr_with_truth']:+.4f}")
            del mdl, dif
            _t.cuda.empty_cache()

    path = Path(results_dir) / "hydrostatic.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
