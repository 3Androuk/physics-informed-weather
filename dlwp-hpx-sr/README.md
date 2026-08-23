# DLWP-HPX backbone for weather super-resolution (t2m)

The backbone of **DLWP-HPX** — Karlbauer et al. (2024), *"Advancing
Parsimonious Deep Learning Weather Prediction Using the HEALPix Mesh"*, JAMES
16, e2023MS004021 — applied to a new task: **super-resolution of ERA5 2-meter
temperature (t2m) on the HEALPix mesh**.

The original DLWP-HPX is a forecasting model; here its *spatial* backbone is
used as a direct mapping `f: degraded t2m -> high-res t2m` on the sphere.

## What is taken from the paper

| Element | Here |
|---|---|
| HEALPix mesh: 12 square faces, conv2d per face | yes (nested ordering, faces as `(12, F, F)` arrays) |
| Cross-face halo padding before every conv | yes — derived from mesh topology, see below |
| Capped GELU activation (cap 10) | yes |
| ConvNeXt-style residual blocks | yes — 3×3 dilated HPX conv + pointwise expansion MLP + 1×1 skip |
| U-Net over mesh resolutions, dilation growing with depth | yes — avg-pool down (= exact HEALPix coarsening), transposed-conv up, dilations `[1, 2, 4]` |
| Temporal recurrence (GRU), multi-step forecasting | **no** — the SR task is single-time-step, so the recurrent parts are dropped |

A practical bonus of the mesh, emphasized by the paper: HEALPix pixels are
**equal-area**, so the plain pixel MSE loss and all mesh-space metrics are
already area-fair — no latitude weighting anywhere.

### The padding, derived instead of hard-coded

The reference implementation hand-codes the 12-face adjacency with per-edge
rotations. Here the halo **index maps are derived from the mesh topology**
(astropy-healpix `neighbours`) by common-neighbour completion: a halo cell
must be a sphere-neighbour of every already-assigned cell that touches it in
the face grid, and may not duplicate a pixel already placed on that face.
Edge strips are filled ring by ring first, then corner blocks. At the eight
3-valent mesh vertices no real pixels exist beyond the corner; those cells
are filled at apply time by averaging their valid grid neighbours (the same
treatment DLWP-HPX gives its missing corners). Arbitrary halo widths are
supported, so dilated convolutions pad correctly.

`tests/test_hpx.py` validates every grid adjacency of the padded maps against
the true sphere topology, *including* the orthogonal/diagonal distinction —
which would catch any sheared, rotated or misplaced halo assignment.

## Task setup

- **Grid:** HPX64 (nside 64, 12×64×64 = 49,152 pixels, ~0.92°), remapped
  bilinearly from the WeatherBench 2 ERA5 0.7° (512×256) conservative regrid.
- **Degradation:** average-pool each face 4× — because of the nested
  ordering this is *exactly* the HEALPix coarsening to HPX16 (~3.7°) — then
  nearest-upsample back to the HPX64 grid as model input.
- **Model:** HEALPix U-Net (`models/hpx_unet.py`), MSE in z-score space,
  global residual (the net predicts a correction to the upsampled input).
- **Split:** train 2007–2015 (validation = time-ordered tail), test 2016–2017,
  daily samples (every 4th 6-hourly step).
- **Baselines:** seam-aware bilinear upsampling (coarse faces padded with a
  1-pixel HEALPix halo before interpolation) and nearest (the input itself).

## Usage

```bash
pip install -r requirements.txt   # install torch with the right CUDA build first

# 1. Data: stream ERA5 t2m from WB2 GCS, remap to HEALPix, cache faces (~300MB)
python -m data.download_era5   --config config/default.yaml

# 2. Train
python -m train.train_sr       --config config/default.yaml

# 3. Evaluate: metrics.json + histogram/map figures under results/
python -m eval.evaluate_sr     --config config/default.yaml

# Tests (mesh/padding/remap/model correctness; no data or GPU needed)
python -m tests.test_hpx
```

Weights & Biases is opt-in, as in the sibling projects: `wandb login` once,
then set `wandb.enabled: true` in the config or pass `--wandb`. Training logs
loss/validation curves; evaluation logs the metrics and figures.

## Evaluation

`eval/evaluate_sr.py` compares model / bilinear / nearest in physical units
(K) on the test split: global RMSE / MAE / bias, RMSE per latitude band, and
value histograms — all computed on the mesh (equal-area, no weighting) — plus
global maps (truth / input / model / error) remapped back to lat-lon.

## Layout

```
dlwp-hpx-sr/
├── config/default.yaml     # single config for data/model/train/eval
├── hpx/
│   ├── mesh.py             # nested index <-> (face, y, x), pixel centers
│   ├── padding.py          # derived cross-face halo padding (the core op)
│   └── remap.py            # lat-lon <-> HEALPix remapping
├── data/
│   ├── download_era5.py    # WB2 GCS -> HEALPix faces (resumable, per-year)
│   ├── dataset.py          # normalized face dataset
│   └── degrade.py          # exact HEALPix coarsen / upsample operators
├── models/hpx_unet.py      # CappedGELU, HPX conv, ConvNeXt block, U-Net
├── train/train_sr.py
├── eval/evaluate_sr.py
└── tests/test_hpx.py       # topology-validated padding + pipeline tests
```

## Extensions

- More variables (winds, precipitation — the paper's variable set), multi-channel SR.
- Larger ratios / OOD robustness (compare with the sibling diffusion project).
- The temporal GRU blocks of DLWP-HPX for video-style SR of t2m sequences.

## Reference

Karlbauer, M., Cresswell-Clay, N., Durran, D. R., Moreno, R. A., Kurth, T.,
Bonev, B., Brenowitz, N., & Butz, M. V. (2024). *Advancing parsimonious deep
learning weather prediction using the HEALPix mesh.* JAMES 16, e2023MS004021.
https://doi.org/10.1029/2023MS004021 — original code:
https://github.com/CognitiveModeling/dlwp-hpx
