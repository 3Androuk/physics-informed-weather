# Physics-Informed Weather

Related experiments in data-driven / physics-informed weather modeling on
ERA5 reanalysis data. Each lives in its own self-contained subfolder with its own
README, requirements, and entry points.

## Projects

### [`fno-afno-baselines/`](fno-afno-baselines/) — Neural-operator forecasting baselines
FNO vs AFNO baseline comparison for ERA5 forecasting (Z500, T850, U850, V850)
using WeatherBench 2 data, with latitude-weighted RMSE / ACC evaluation and
zero-shot high-resolution inference. Also includes a FourCastNet inference demo.

→ See [`fno-afno-baselines/README.md`](fno-afno-baselines/README.md).

### [`era5-diffusion-downscaling/`](era5-diffusion-downscaling/) — Diffusion super-resolution (Phase 1)
A reimplementation of the physics-agnostic core of Shu, Li & Barati Farimani
(2023), *"A physics-informed diffusion model for high-fidelity flow field
reconstruction"* (J. Comput. Phys. 478, 111972), applied to ERA5 Z500
downscaling. A single DDPM, trained on high-fidelity patches only, reconstructs
across 4× and 8× degradations without retraining.

→ See [`era5-diffusion-downscaling/README.md`](era5-diffusion-downscaling/README.md).

### [`dlwp-hpx-sr/`](dlwp-hpx-sr/) — DLWP-HPX backbone for super-resolution
The backbone of DLWP-HPX (Karlbauer et al. 2024, *"Advancing Parsimonious Deep
Learning Weather Prediction Using the HEALPix Mesh"*) applied to
super-resolution of ERA5 t2m on the HEALPix mesh: conv2d over the 12 mesh
faces with topology-derived cross-face halo padding, capped GELU, ConvNeXt
blocks, and a U-Net over mesh resolutions.

→ See [`dlwp-hpx-sr/README.md`](dlwp-hpx-sr/README.md).

## Layout

```
physics-informed-weather/
├── README.md                     # this file
├── .gitignore                    # shared ignores (caches, checkpoints, venvs)
├── fno-afno-baselines/           # FNO/AFNO forecasting baselines
├── era5-diffusion-downscaling/   # diffusion downscaling reimplementation
└── dlwp-hpx-sr/                  # DLWP-HPX backbone for t2m super-resolution
```

Each subproject is run from inside its own folder (paths like `cache/`,
`checkpoints/`, `results/` are relative to the subproject directory). Both share
the repo-root `.gitignore`.
