"""Collate the hydrostatic 2x3 matrix into the tables worth reporting.

Reads the six compare_geo result JSONs produced by run_hyd_matrix.sh:

                 coef 0        coef 0.5      coef 1.0
    projection   _proj         _proj_hyd0.5  _proj_hyd1
    no proj      (bare)        _hyd0.5       _hyd1

and prints, per ratio:

  1. headline  -- L2(normalized), L2(physical), spectrum for all six arms,
     each with its delta against the coef-0 arm of the SAME projection row
     (that is the arm the physics term has to beat);
  2. the decisive split -- the hydrostatic term only touches z/t/q at
     850/700/500, so it separates the 6 CONSTRAINED channels from the 14
     UNCONSTRAINED ones. In the wrong-order run the correction paid for
     balance almost entirely out of geopotential (z500 +1052%) while all 14
     unconstrained channels moved <1%. Reordered, the projection restores
     exact data consistency afterwards, so this is where we find out whether
     the damage was purely the ordering bug.

Usage:  python -m eval.collate_hydrostatic [--dir results_wb220]
"""

import argparse
import json
from pathlib import Path

LEVELS = (850, 700, 500)
CONSTRAINED = tuple(f"{v}@{lv}" for lv in LEVELS
                    for v in ("geopotential", "temperature", "specific_humidity"))
COEFS = (0.0, 0.5, 1.0)


def stem(project, coef):
    return ("compare_diffusion_geo"
            + ("_proj" if project else "")
            + (f"_hyd{coef:g}" if coef else ""))


def load(dirp, project, coef, want_n=None, want_sampling="spread"):
    """Load one arm, REJECTING results left over from an earlier sweep.

    compare_geo writes to a fixed stem per (project, coef), so a rerun
    overwrites in place -- and a file for an arm that has not finished yet is
    still the PREVIOUS run's. Without this check the collation silently mixes
    old and new numbers, which is exactly the mistake that makes a sweep look
    finished when it is not. A file only counts if it carries the `config`
    block (written only after the sampling fix) and matches this sweep.
    """
    p = dirp / f"{stem(project, coef)}.json"
    if not p.exists():
        return None, "not run yet"
    d = json.loads(p.read_text())
    cfg = d.get("config")
    if cfg is None:
        return None, "STALE (pre-fix file, no config block)"
    if want_n is not None and cfg.get("n_patches") != want_n:
        return None, f"STALE (n={cfg.get('n_patches')}, want {want_n})"
    if cfg.get("sampling") != want_sampling:
        return None, f"STALE (sampling={cfg.get('sampling')})"
    return d, None


def pct(new, old):
    return float("nan") if not old else 100.0 * (new - old) / old


def fmt_delta(d):
    return "     -" if d != d else f"{d:+6.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_wb220")
    ap.add_argument("--model", default="diffusion_geo")
    ap.add_argument("--n-patches", type=int, default=128,
                    help="the sweep's patch count; arms not matching it are "
                         "treated as leftovers from an earlier run")
    args = ap.parse_args()
    dirp = Path(args.dir)

    loaded = {(pr, c): load(dirp, pr, c, want_n=args.n_patches)
              for pr in (True, False) for c in COEFS}
    grid = {k: v for k, (v, _) in loaded.items()}
    have = {k: v for k, v in grid.items() if v}
    if not have:
        raise SystemExit(f"no current result files in {dirp} "
                         f"(n_patches={args.n_patches}, sampling=spread)")

    any_run = next(iter(have.values()))
    cfg = any_run.get("config", {})
    print(f"{cfg.get('n_patches')} of {cfg.get('n_total')} test patches, "
          f"sampling={cfg.get('sampling')}")
    excluded = [(k, why) for k, (v, why) in loaded.items() if v is None]
    if excluded:
        print("EXCLUDED:")
        for (p, c), why in excluded:
            print(f"  proj={'on ' if p else 'off'} coef {c:<4g} -> {why}")
    print()

    ratios = [r for r in any_run if r.endswith("x")]
    for ratio in ratios:
        print(f"{'=' * 72}\n ratio {ratio}\n{'=' * 72}")

        # bicubic reference, for scale
        bic = any_run[ratio].get("Bicubic")
        if bic:
            print(f" bicubic reference: L2(norm) {bic['l2_normalized']:.4f} | "
                  f"L2(phys) {bic['l2']:8.2f} | spec {bic['spectrum_log_l1']:.4f}\n")

        print(f" {'arm':<22} {'L2(norm)':>9} {'d':>7} {'L2(phys)':>9} {'d':>7} "
              f"{'spectrum':>9} {'d':>7}")
        print(" " + "-" * 70)
        for pr in (True, False):
            base = grid[(pr, 0.0)]
            b = base[ratio][args.model] if base and ratio in base else None
            for c in COEFS:
                run = grid[(pr, c)]
                if not run or ratio not in run:
                    continue
                m = run[ratio][args.model]
                label = f"{'proj' if pr else 'no-proj':<8} coef {c:<4g}"
                if b is None or c == 0.0:
                    print(f" {label:<22} {m['l2_normalized']:9.4f} {'':>7} "
                          f"{m['l2']:9.2f} {'':>7} {m['spectrum_log_l1']:9.4f} {'':>7}"
                          + ("   <- reference" if c == 0.0 else ""))
                else:
                    print(f" {label:<22} {m['l2_normalized']:9.4f} "
                          f"{fmt_delta(pct(m['l2_normalized'], b['l2_normalized']))} "
                          f"{m['l2']:9.2f} {fmt_delta(pct(m['l2'], b['l2']))} "
                          f"{m['spectrum_log_l1']:9.4f} "
                          f"{fmt_delta(pct(m['spectrum_log_l1'], b['spectrum_log_l1']))}")
            print()

        # ── constrained vs unconstrained ────────────────────────────────
        print(f" per-channel effect of the correction (vs coef 0, same row)")
        print(f" {'arm':<22} {'6 constrained z/t/q':>22} {'14 untouched':>16}")
        print(" " + "-" * 62)
        for pr in (True, False):
            base = grid[(pr, 0.0)]
            if not base or ratio not in base:
                continue
            bpv = base[ratio][args.model]["per_variable"]
            for c in (0.5, 1.0):
                run = grid[(pr, c)]
                if not run or ratio not in run:
                    continue
                pv = run[ratio][args.model]["per_variable"]
                con = [pct(pv[k]["l2"], bpv[k]["l2"]) for k in pv if k in CONSTRAINED]
                unc = [pct(pv[k]["l2"], bpv[k]["l2"]) for k in pv if k not in CONSTRAINED]
                mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
                print(f" {'proj' if pr else 'no-proj':<8} coef {c:<4g}      "
                      f"{fmt_delta(mean(con))}  (worst {fmt_delta(max(con))})"
                      f"  {fmt_delta(mean(unc))}")
        print()

        # ── the geopotential columns, named ─────────────────────────────
        print(f" geopotential L2 by level (the channel that broke before)")
        hdr = "  ".join(f"{'z' + str(lv):>10}" for lv in LEVELS)
        print(f" {'arm':<22} {hdr}")
        print(" " + "-" * 58)
        for pr in (True, False):
            for c in COEFS:
                run = grid[(pr, c)]
                if not run or ratio not in run:
                    continue
                pv = run[ratio][args.model]["per_variable"]
                vals = "  ".join(f"{pv[f'geopotential@{lv}']['l2']:10.2f}" for lv in LEVELS)
                print(f" {('proj' if pr else 'no-proj') + f'  coef {c:g}':<22} {vals}")
        print()


if __name__ == "__main__":
    main()
