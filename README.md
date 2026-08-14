# Physics-Informed Weather

Two related experiments in data-driven / physics-informed weather modeling on
ERA5 reanalysis data. Each lives in its own self-contained subfolder with its own
README, requirements, and entry points.

## Projects

### [`fno-afno-baselines/`](fno-afno-baselines/) — Neural-operator forecasting baselines
FNO vs AFNO baseline comparison for ERA5 forecasting (Z500, T850, U850, V850)
using WeatherBench 2 data, with latitude-weighted RMSE / ACC evaluation and
zero-shot high-resolution inference. Also includes a FourCastNet inference demo.

→ See [`fno-afno-baselines/README.md`](fno-afno-baselines/README.md).

### [`era5-diffusion-downscaling/`](era5-diffusion-downscaling/) — Generative super-resolution of ERA5
Started as a reimplementation of the physics-agnostic core of Shu, Li & Barati
Farimani (2023) — a single DDPM trained on high-fidelity patches only, guided at
inference to reconstruct 4×/8× degradations without retraining — and has grown
into a comparison suite of generative downscaling methods:

- **Guided unconditional DDPM** (the Shu et al. reimplementation, Phase 1)
- **Conditional transport models** — flow matching and stochastic interpolants,
  trained on randomized degradation ratios, evaluated at a held-out 16×
- **Residual (split-model) diffusion** — CorrDiff-style learned mean + sampled
  residual
- **Geographic conditioning** — learned sphere-native location embeddings
  (Instant-NGP hash grid, HEALPix pyramid) vs zero-parameter baselines
  (raw coordinates, fixed sinusoidal basis, real orography/land-sea static
  fields), with a permutation control
- **Full-field reconstruction** — patch-trained models applied to whole
  lat-band fields: direct inference, overlap-blend tiling with shared noise and
  exact data-consistency projection, and MultiDiffusion-style per-step fusion

→ See [`era5-diffusion-downscaling/README.md`](era5-diffusion-downscaling/README.md)
for methods, commands, and references.

## Branch guide

| branch | contents |
|---|---|
| `main` | original Phase-1 state (guided DDPM + baselines, single variable) |
| `codex/flow-stochastic-superres` | **active development**: everything listed above, single-variable configs (t2m, z500). Supersedes `worktree-geo-hash-encoding` (its full history is contained here). |
| `claude/wb2-20var-downscaling` | everything above **plus multivariable downscaling**: 20 WB2 variables as 20 input/output channels (`config/wb2_20var.yaml`), per-channel normalization, display-channel evaluation |

## Layout

```
physics-informed-weather/
├── README.md                     # this file
├── .gitignore                    # shared ignores (caches, checkpoints, venvs)
├── fno-afno-baselines/           # FNO/AFNO forecasting baselines
└── era5-diffusion-downscaling/   # generative downscaling suite
```

Each subproject is run from inside its own folder (paths like `cache/`,
`checkpoints/`, `results/` are relative to the subproject directory). Both share
the repo-root `.gitignore`.
