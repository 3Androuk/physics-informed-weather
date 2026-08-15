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

### Geographic conditioning: learned embeddings vs baselines

The learned location tables (`--encoder hash|healpix`) are compared against a
ladder of baselines that isolate what, if anything, the *learning* contributes
(all selectable via `--geo --encoder <name>` on every trainer; checkpoint names
gain matching suffixes `_geo`, `_geo_hpx`, `_geo_xyz`, `_geo_sin`,
`_geo_static`, `_geo_combo`):

| encoder      | payload                              | learned geo params | isolates |
|--------------|--------------------------------------|--------------------|----------|
| `xyz`        | raw unit-sphere coords as channels   | 0                  | is *any* encoder needed beyond coordinates? |
| `sinusoidal` | fixed multiscale Fourier basis       | 0                  | learned tables vs engineered multiscale basis |
| `static`     | real orography / land-sea mask / slope (WB2) | 0          | learned location identity vs physiography (the literature default) |
| `hash_static` | static fields + learned hash table | ~0.5M             | does learning add anything ON TOP of physiography? (the tie-breaker) |
| `hash`, `healpix` | learned multiresolution tables  | ~0.5M              | — |

`static` needs a one-time precompute:
`python -m data.make_static_fields --config config/t2m.yaml` (a few MB from
the same WB2 zarr). The `--shuffle-geo` permutation control in
`eval.make_tables_figures` applies to every arm.

Leveled encoders (`hash`, `healpix`, `hash_static`) additionally accept
`--gated` (config: `geo.level_gating`): noise-dependent gating that hides the
fine embedding levels at high noise, so fine-table gradient comes only from
denoising steps where location detail is actually resolvable — a
zero-parameter regularizer against fine-scale location memorization
(checkpoint suffix `_gated`).

#### Findings (guided DDPM, t2m, projected reconstruction)

Measured on the deterministic test-patch set, 4× ratio (L2 in K after
denormalization; spectrum = mean |log10| radial-spectrum error):

| arm | L2 | spectrum |
|---|---|---|
| bicubic | 0.4559 | 0.0691 |
| no geo | 0.4052 | 0.0101 |
| `xyz` | 0.4061 | 0.0123 |
| `sinusoidal` | 0.4112 | 0.0233 |
| `hash` (learned) | 0.3526 | 0.0086 |
| `static` (physiography) | 0.3478 | 0.0100 |

- **The geographic gain is real (~13 % L2) but encoder-invariant at the top.**
  `hash`, `static`, and the ring-matched `healpix` ladder (Nside 8–64, i.e.
  scales matched to the hash band) all land within noise of each other, at 8×
  as well as 4×. The learned tables converge to a *physiography-equivalent*
  signal — they earn their keep only where a static descriptor of the surface
  would too.
- **Raw coordinates are a null** (`xyz` ≈ no-geo): the UNet cannot exploit
  location identity without a multiscale representation.
- **A fixed Fourier basis actively hurts the spectrum** (`sinusoidal`,
  2.3× worse than no-geo): globally-supported oscillatory channels leak into
  the generated high frequencies.
- **Scale misallocation, not spherical geometry, explained the original
  HEALPix gap.** The original power-of-two ladder (Nside 1–128) spent levels
  outside the useful band and underperformed; matching the ladder to the hash
  band recovers hash parity with zero other changes. This kills the case for
  more elaborate spherical parameterizations (icosphere / cubed-sphere hash)
  as a route to accuracy on this task.

Pending before these become thesis-final: the `hash_static` combo arm (does
learning add anything *on top of* physiography?), seed replicates for the
~0.005-L2 gaps, and the `--gated` ablation.

## Full-field reconstruction

The models train on 128×128 patches but reconstruct **whole test fields**
(the full lat-band grid) two ways, compared by `eval/full_field.py`:

- **Direct** — the fully-convolutional UNet consumes the entire field at once.
  The bottleneck self-attention now runs memory-efficiently
  (`F.scaled_dot_product_attention`, mathematically identical to the explicit
  softmax, no retraining needed), but it sees ~40× more tokens than at
  training time — this mode measures whether that mismatch hurts.
- **Tiled** — overlapping training-sized tiles blended with a smooth cosine
  window (`sample/full_field.py`). Three details make the stitching seam-free
  and physically consistent: tile origins are snapped to multiples of the
  ratio so each tile's coarse observation is an exact crop of the global one;
  every stochastic sampler starts from crops of ONE global noise field, so
  overlapping tiles agree where they overlap; and a final exact block-average
  projection re-pins the stitched field to the observed coarse input globally.
  At weakly-constrained ratios (8×) independent tile chains can still drift
  apart at low frequencies (visible tile-scale squares); `--project-steps`
  anchors every tile to the shared observation at every step.
- **Fused** (`--modes fused`) — MultiDiffusion-style synchronized sampling
  (Bar-Tal et al., ICML 2023): ONE global chain whose per-step prediction is
  fused from overlapping tile evaluations, so tiles share a single trajectory
  and cannot drift — seam-free by construction, at the same compute cost as
  tiled. The same per-step tile-merging scheme is used for km-scale weather
  diffusion (NVIDIA CorrDiff-family tiled sampling).

```bash
python -m eval.full_field --config config/t2m.yaml --n-fields 4
# fewer/more tiles overlap, skip direct mode on small GPUs:
python -m eval.full_field --config config/t2m.yaml --overlap 48 --modes tiled
```

Reports L2 + spectrum error and runtime per method (diffusion, flow matching,
stochastic interpolant, direct map, bicubic) and mode, with full-field
comparison figures under `results/full_field/`.

## Data

ERA5 Z500 is streamed from the **WeatherBench 2 public GCS** Zarr store
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

## References

**Core method (reimplemented here):**
- Shu, D., Li, Z., Barati Farimani, A. (2023). *A physics-informed diffusion
  model for high-fidelity flow field reconstruction.* J. Comput. Phys. 478,
  111972. https://doi.org/10.1016/j.jcp.2023.111972 — original code:
  https://github.com/BaratiLab/Diffusion-based-Fluid-Super-resolution
- Ho, J., Jain, A., Abbeel, P. (2020). *Denoising Diffusion Probabilistic
  Models.* NeurIPS. https://arxiv.org/abs/2006.11239
- Song, J., Meng, C., Ermon, S. (2021). *Denoising Diffusion Implicit Models.*
  ICLR. https://arxiv.org/abs/2010.02502

**Conditional transport models:**
- Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M. (2023). *Flow
  Matching for Generative Modeling.* ICLR. https://arxiv.org/abs/2210.02747
- Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E. (2023). *Stochastic
  Interpolants: A Unifying Framework for Flows and Diffusions.*
  https://arxiv.org/abs/2303.08797

**Residual (split-model) diffusion:**
- Mardani, M., et al. (2025). *Residual corrective diffusion modeling for
  km-scale atmospheric downscaling* (CorrDiff). Communications Earth &
  Environment. https://doi.org/10.1038/s43247-025-02042-5

**Full-field reconstruction (fused tiled sampling):**
- Bar-Tal, O., Yariv, L., Lipman, Y., Dekel, T. (2023). *MultiDiffusion:
  Fusing Diffusion Paths for Controlled Image Generation.* ICML, PMLR 202.
  https://proceedings.mlr.press/v202/bar-tal23a.html

**Geographic conditioning:**
- Müller, T., Evans, A., Schied, C., Keller, A. (2022). *Instant Neural
  Graphics Primitives with a Multiresolution Hash Encoding* (Instant-NGP).
  ACM TOG 41(4). https://arxiv.org/abs/2201.05989
- Górski, K.M., et al. (2005). *HEALPix: A Framework for High-Resolution
  Discretization and Fast Analysis of Data Distributed on the Sphere.*
  ApJ 622, 759. https://arxiv.org/abs/astro-ph/0409513

**Data:**
- Rasp, S., et al. (2024). *WeatherBench 2: A Benchmark for the Next
  Generation of Data-Driven Global Weather Models.* JAMES.
  https://doi.org/10.1029/2023MS004019
