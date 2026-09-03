"""Variance inflation as pure post-processing over saved ensemble members.

    x_i <- mu + alpha * (x_i - mu)

Leaves the ensemble mean (hence RMSE and its spectrum) EXACTLY unchanged and
preserves data consistency for free: every member satisfies coarsen(x_i) = y,
so the deviations (x_i - mu) already lie in ker A and scaling them stays there.
This is what the Langevin corrector paid +8-12% CRPS to approximate.

Sweeps alpha, reports the oracle optimum (in-sample) and the alpha at which
spread meets the reliable line; --alpha applies a fixed value (use the other
lead's optimum for an honest cross-lead number).

Run: python -m eval.inflate_members --members results_t2m/members/combo_eta15 \
        --data-dir datasets/forecast_hres_t2m --lead 24 [--alpha 1.4]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.metrics import crps_ensemble  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--lead", type=int, required=True)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0])
    ap.add_argument("--alpha", type=float, default=None,
                    help="also evaluate this fixed alpha (e.g. the other lead's optimum)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    truth_all = np.load(Path(args.data_dir) / f"truth_{args.lead}h.npy", mmap_mode="r")
    files = sorted(glob.glob(f"{args.members}/members_{args.lead}h_init*.npy"))
    if not files:
        raise SystemExit("no member files found")
    alphas = sorted(set(args.alphas + ([args.alpha] if args.alpha else [])))
    M = None
    acc = {a: {"crps": [], "spread": []} for a in alphas}
    se_mean = []
    for f in files:
        i = int(Path(f).stem.split("init")[-1])
        m = np.load(f).astype(np.float32)            # (M, C, H, W)
        M = m.shape[0]
        t = np.asarray(truth_all[i], dtype=np.float32)   # (C, H, W)
        mu = m.mean(0, keepdims=True)
        se_mean.append(((mu[0] - t) ** 2).mean())
        for a in alphas:
            x = mu + a * (m - mu)
            acc[a]["crps"].append(crps_ensemble([x[k, 0] for k in range(M)], t[0]))
            acc[a]["spread"].append(x.std(0).mean())
    ens_rmse = float(np.sqrt(np.mean(se_mean)))
    reliable = ens_rmse / np.sqrt(1.0 + 1.0 / M)
    rows = []
    print(f"{args.lead}h | {len(files)} inits x {M} members | ens-mean RMSE {ens_rmse:.4f} "
          f"(unchanged by inflation) | reliable spread {reliable:.4f}")
    print(f"{'alpha':>6} {'CRPS':>8} {'spread':>8} {'spread/reliable':>16}")
    for a in alphas:
        c, s = float(np.mean(acc[a]["crps"])), float(np.mean(acc[a]["spread"]))
        rows.append({"alpha": a, "crps": c, "spread": s, "spread_ratio": s / reliable})
        print(f"{a:6.2f} {c:8.4f} {s:8.4f} {s / reliable:16.3f}")
    best = min(rows, key=lambda r: r["crps"])
    cal = min(rows, key=lambda r: abs(r["spread_ratio"] - 1.0))
    print(f"oracle alpha* = {best['alpha']:.2f} (CRPS {best['crps']:.4f}); "
          f"calibrated alpha ~ {cal['alpha']:.2f} (spread/reliable {cal['spread_ratio']:.2f})")
    out = {"lead_h": args.lead, "n_inits": len(files), "members": M,
           "ens_mean_rmse": ens_rmse, "reliable_spread": reliable, "rows": rows,
           "oracle_alpha": best["alpha"], "oracle_crps": best["crps"],
           "calibrated_alpha": cal["alpha"]}
    p = Path(args.out or f"{args.members}/inflation_{args.lead}h.json")
    p.write_text(json.dumps(out, indent=2)); print(f"saved -> {p}")


if __name__ == "__main__":
    main()
