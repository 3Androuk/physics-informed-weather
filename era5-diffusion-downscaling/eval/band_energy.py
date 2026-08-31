"""Gate 1 for cascade diffusion (docs/h2cd_cascade.md): band-energy analysis.

Decomposes test patches on the dyadic block-average pyramid — the SAME
operator family as the degradation A — into orthogonal detail bands:

    P_r x   = upsample_nearest(coarsen(x, r))        (parent averages)
    band_j  = P_{2^j} x - P_{2^{j+1}} x              (P_1 = identity)
    coarse  = P_{r_max} x

Every band has exactly zero mean inside its parent cells, bands are mutually
orthogonal, and for any dyadic ratio r the free (ker A) content is exactly
the bands finer than r. Reports per-band energy, per-level decay, and the
per-ratio free-band breakdown.

Reading the gate: a staggered per-band noise schedule (H2CD) has something
to exploit only if band energies are strongly non-uniform — a single global
schedule must span the full max/min energy ratio across bands, and the
cascade's advantage grows with that span (WSGM's argument). If adjacent
bands sit within a small factor (~2-4x) of each other, the cascade has
little leverage at this resolution: stop before building the trainer.

Run:
    python -m eval.band_energy --config config/t2m.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import PatchDataset, load_norm_stats  # noqa: E402
from data.degrade import coarsen, upsample_nearest  # noqa: E402
from utils import ensure_dir, load_config  # noqa: E402


def block_bands(x: torch.Tensor, coarsest: int):
    """Orthogonal dyadic block-pyramid decomposition of (N, C, H, W).

    Returns (labels, list of band tensors) ordered finest -> coarsest; the
    last entry is the P_coarsest remainder. Bands sum exactly to x.
    """
    hw = x.shape[-2:]
    bands, labels = [], []
    prev = x
    r = 2
    while r <= coarsest:
        p = upsample_nearest(coarsen(x, r), hw)
        bands.append(prev - p)
        labels.append(f"detail_{r // 2}to{r}")
        prev = p
        r *= 2
    bands.append(prev)
    labels.append(f"coarse_{coarsest}")
    return labels, bands


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--patches", type=int, default=256,
                    help="number of test patches to analyze")
    ap.add_argument("--coarsest", type=int, default=32,
                    help="coarsest pyramid level (power of two); bands finer "
                         "than each eval ratio must be resolved, so this "
                         "should be >= the largest ratio of interest")
    args = ap.parse_args()
    cfg = load_config(args.config)
    patch_dir = Path(cfg["paths"]["patch_dir"])
    results_dir = ensure_dir(cfg["paths"]["results_dir"])

    normalizer = load_norm_stats(patch_dir)
    ds = PatchDataset(patch_dir / "test_patches.npy", normalizer)
    n = min(args.patches, len(ds))
    x = torch.stack([ds[i] for i in range(n)])
    if x.dim() == 3:
        x = x.unsqueeze(1)
    print(f"Analyzing {n} patches of shape {tuple(x.shape[1:])} "
          f"(normalized units), pyramid to {args.coarsest}x")

    labels, bands = block_bands(x, args.coarsest)
    total = float(x.square().mean())
    energies = [float(b.square().mean()) for b in bands]
    additivity = abs(sum(energies) - total) / total
    print(f"orthogonality check: |sum(bands) - total| / total = {additivity:.2e}")

    print(f"\n{'band':>16s} {'energy':>10s} {'fraction':>9s} {'decay vs next-coarser':>22s}")
    for i, (lab, e) in enumerate(zip(labels, energies)):
        decay = (energies[i + 1] / e) if (i + 1 < len(energies) and e > 0) else float("nan")
        print(f"{lab:>16s} {e:10.4f} {e / total:9.3f} {decay:22.2f}")

    detail = energies[:-1]
    span = max(detail) / max(min(detail), 1e-12)
    print(f"\nmax/min DETAIL-band energy span: {span:.1f}x "
          f"(what one global schedule must cover; the cascade's leverage)")

    ratios = sorted({rc["ratio"] for rc in cfg["sample"]["reconstructions"]} | {16})
    per_ratio = {}
    for r in ratios:
        if r > args.coarsest or (r & (r - 1)) != 0:
            continue
        free_idx = [i for i, lab in enumerate(labels)
                    if lab.startswith("detail_") and
                    int(lab.split("to")[1]) <= r]
        free = sum(energies[i] for i in free_idx)
        per_ratio[f"{r}x"] = {
            "free_energy_fraction": free / total,
            "free_bands": {labels[i]: energies[i] / total for i in free_idx},
        }
        print(f"\n{r}x: ker A holds {free / total:.1%} of energy, split "
              + ", ".join(f"{labels[i]} {energies[i] / total:.1%}" for i in free_idx))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(energies)), energies, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("mean energy (normalized units, log)")
    ax.set_title(f"Block-pyramid band energies ({n} patches) — "
                 f"detail span {span:.0f}x")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig_path = results_dir / "band_energy.png"
    fig.savefig(fig_path, dpi=130, bbox_inches="tight")

    out = {
        "n_patches": n, "coarsest": args.coarsest, "total_energy": total,
        "bands": dict(zip(labels, energies)),
        "fractions": {lab: e / total for lab, e in zip(labels, energies)},
        "detail_span": span, "orthogonality_residual": additivity,
        "per_ratio": per_ratio,
    }
    json_path = results_dir / "band_energy.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {json_path}, {fig_path}")
    print("Gate reading: span >~ 10x across the free bands of your target "
          "ratio favors a per-band cascade; span <~ 4x means a global "
          "schedule already covers it — park H2CD.")


if __name__ == "__main__":
    main()
