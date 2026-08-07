# ERA5 Diffusion Downscaling (Phase 1, physics-agnostic)

A reimplementation of the **physics-agnostic core** of Shu, Li & Barati Farimani
(2023), *"A physics-informed diffusion model for high-fidelity flow field
reconstruction"* (J. Comput. Phys. 478, 111972), applied to **ERA5 Z500**
(500 hPa geopotential) downscaling.

The original method reframes super-resolution as **guided denoising**: a DDPM is
trained on high-fidelity fields only, and at inference a low-fidelity input is
mixed with Gaussian noise and used to guide a partial (DDIM) backward diffusion
chain. The headline claim — reproduced here — is **distribution robustness**: a
*single* trained model reconstructs across multiple input degradations (4×, 8×)
**without retraining**, where a direct-mapping baseline degrades badly
out-of-distribution.

> **Phase 1 excludes the physics-informed conditioning.** ERA5 has no exact
> closed-form governing PDE, so the residual-gradient term (`c`, `w`) from the
> paper's Algorithm 2 is dropped. See *Deferred work* below. This is a faithful
> reimplementation of contribution (a), not a new method — cite the original.

## Method (what is built)

1. **Training data:** high-fidelity Z500 patches only (the model never sees
   low-fidelity data during training — this is what gives robustness).
2. **Objective:** standard DDPM noise-prediction loss (Ho et al. 2020), Eq. (2).
3. **Noise-mixing `g`:** at inference, mix the (nearest-upsampled) low-fidelity
   input with Gaussian noise so different degradations are pulled toward a common
   Gaussian shape.
4. **Guided sampling:** start the backward chain from an intermediate timestep
   `t ∈ [0, T/2]`, injecting guidance via
   `x_t = √ᾱ_t · x^(g) + √(1−ᾱ_t) · ε`, then DDIM (σ=0) down to `x_0`
   (Algorithm 2, physics term dropped). Outer loop `K` for recursive refinement.

### Baselines
- **Bicubic** interpolation (classical).
- **Direct-mapping UNet** `f: X→Y`, trained on **4× pairs only** — the brittle
  benchmark that should degrade on 8× (out-of-distribution).

## Conditional transport models

Two alternatives model the conditional high-resolution distribution directly.
Unlike the original unconditional DDPM, both receive the upsampled coarse field
as a conditioning channel during training and randomize the degradation ratio
per batch. Optional hash-grid or HEALPix geographic conditioning is supported
through the same `geo` config used elsewhere in the project.

- **Flow matching:** learns the conditional velocity on the linear path
  `x_t = (1-t)z + t x_HF`, with target velocity `x_HF-z`. Sampling integrates
  the learned probability-flow ODE from Gaussian noise to a high-resolution
  field using Euler or Heun steps.
- **Stochastic interpolants:** uses
  `x_t = (1-t)z + t x_HF + γ sin(πt)ε`. The UNet jointly learns the velocity
  and a variance-scaled score. It supports deterministic probability-flow ODE
  sampling and a marginal-preserving SDE sampler for ensembles.

Both samplers optionally apply an exact final block-average projection so the
generated field coarsens back to the observed input. The default training ratios
are `{2, 4, 8}` and the default evaluation also includes held-out `16×`.

## Multivariable (20-channel) downscaling

`config/wb2_20var.yaml` runs the same pipeline on **20 WB2 variables jointly —
20 input and 20 output channels**: 5 surface fields (t2m, u10, v10, mslp,
total-column water vapour) plus geopotential, temperature, u, v and specific
humidity at 500/700/850 hPa. t2m is channel 0 and the *display channel* used
for figures and the headline physical-unit metrics
(`eval.display_channel`).

- Channels are listed under `data.variables` (name + optional pressure level);
  the list order fixes the channel order everywhere, and
  `unet.in_channels`/`out_channels` must match its length.
- Downloaded fields, patches and models are all `(…, C, H, W)`; normalization
  is per-channel z-score.
- Evaluation reports the display channel in physical units (as before), plus
  an all-channel L2 in normalized units and a per-channel breakdown in the
  JSON outputs.
- Single-variable configs (`t2m.yaml`, `z500.yaml`, `default.yaml`) are
  unchanged and keep working; they are just the C=1 case.

```bash
python -m data.download_era5              --config config/wb2_20var.yaml
python -m data.make_patches               --config config/wb2_20var.yaml
python -m train.train_flow_matching       --config config/wb2_20var.yaml
python -m eval.compare_transports         --config config/wb2_20var.yaml
```

Note the 20-channel dataset is ~20x the single-variable footprint (the default
patch settings produce ~35 GB of training patches; patches are streamed to an
on-disk memmap, never held in RAM — reduce `patches.per_field` to shrink it).

## Data

ERA5 fields are streamed from the **WeatherBench 2 public GCS** Zarr store
(`gs://weatherbench2/...`, no credentials), then cropped into 128×128 patches
with a time-based train/test split. Low-fidelity inputs are produced by
average-pool coarsening (4× → 32×32, 8× → 16×16) and nearest-upsampling back to
128×128.

## Usage

```bash
pip install -r requirements.txt   # install torch with the right CUDA build first

# 1. Data
python -m data.download_era5      --config config/default.yaml
python -m data.make_patches       --config config/default.yaml

# 2. Train (diffusion on HF patches only; direct-map on 4x pairs only)
python -m train.train_diffusion   --config config/default.yaml
python -m train.train_directmap   --config config/default.yaml

# Conditional transport alternatives (one model across configured ratios)
python -m train.train_flow_matching          --config config/default.yaml
python -m train.train_stochastic_interpolant --config config/default.yaml

# 3. Robustness experiment + figures (same diffusion model on 4x AND 8x)
python -m eval.make_tables_figures --config config/default.yaml

# Flow matching vs stochastic interpolants vs bicubic
python -m eval.compare_transports --config config/default.yaml
```

Transport settings live under `transport:` in each config. Useful overrides for
evaluation include `--steps`, `--solver {euler,heun}`,
`--si-sampler {ode,sde}`, `--stochasticity`, and
`--projection {none,final,each}`. Add `--geo` or
`--geo --encoder healpix` to either training command to enable geographic
conditioning; HEALPix requires running `data.make_healpix_index` first.

### Experiment tracking (Weights & Biases)

wandb is **opt-in** and independent of the always-on TensorBoard logs
(`results/tb`). Log in once, then flip `wandb.enabled: true` in
`config/default.yaml` (or a copy):

```bash
wandb login                       # paste your API key (once per machine)
# set wandb.enabled: true in config/default.yaml
python -m train.train_diffusion --config config/default.yaml
python -m train.train_directmap --config config/default.yaml
```

The diffusion run logs training loss and the periodic unconditional sample
grids; the direct-map run logs its MSE curve. Both go to the
`era5-diffusion-downscaling` project. Set `wandb.project/entity/name` in the
config to customize. To log without an account, run `wandb offline` first.

## Evaluation

- **L2 norm** (RMSE) — pointwise error vs ground truth.
- **Energy / power spectrum** `E(k)` — high-wavenumber structure recovery.
- **Value distribution** — histogram of Z500 vs ground truth.

Headline table: rows `{4× in-dist, 8× out-of-dist}` × columns
`{Diffusion, Direct-map, Bicubic}`. The story is the OOD row.

## Deferred work
- **Physics-informed conditioning** via an *approximate* equation
  (quasi-geostrophic / barotropic vorticity on 500 hPa) — a future ablation.
- Multiple variables (Z500 → T850, wind); sparse-reconstruction task.

## Reference
Shu, D., Li, Z., Barati Farimani, A. (2023). *A physics-informed diffusion model
for high-fidelity flow field reconstruction.* J. Comput. Phys. 478, 111972.
https://doi.org/10.1016/j.jcp.2023.111972 — original code:
https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution
