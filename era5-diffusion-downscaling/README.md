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
| `hash_static` (combo) | 0.3410 | 0.0096 |

Same three top arms at 8× (weaker guidance, larger errors overall):

| arm (8×) | L2 | spectrum |
|---|---|---|
| bicubic | 0.7654 | 0.1017 |
| `hash` | 0.6353 | 0.0257 |
| `static` | 0.6322 | 0.0265 |
| `hash_static` (combo) | 0.6217 | 0.0258 |

Re-evaluating the same checkpoints moves L2 by ~±0.002–0.005 (fresh sampling
noise; e.g. `static` 0.3478 vs 0.3458, `hash` 0.3526 vs 0.3530 across runs) —
the run-to-run noise floor for reading the tables above.

- **The geographic gain is real (~13 % L2) but nearly encoder-invariant at
  the top.** `hash`, `static`, and the ring-matched `healpix` ladder
  (Nside 8–64, i.e. scales matched to the hash band) all land within noise of
  each other, at 8× as well as 4×. The learned tables converge to a
  *physiography-equivalent* signal — they earn their keep only where a static
  descriptor of the surface would too.
- **The combo (tie-breaker) arm comes out best at both ratios**: −0.005 L2 vs
  `static` at 4× and −0.011 at 8×, with a spectrum between `hash` and
  `static`. Direction is consistent, but the 4× margin sits at the noise
  floor, so the small-increment reading (learning adds a little *on top of*
  physiography) needs seed replicates to be conclusive; a `--shuffle-geo`
  control on the combo arm would also rule out the capacity confound (the
  combo conditions on 19 channels vs static's 3).
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

4-member ensembles (64 patches, projected, `--ensemble 4`):

| arm | single L2 | ens-mean L2 | CRPS | spread |
|---|---|---|---|---|
| 4× `hash` | 0.3715 | 0.2847 | 0.1150 | 0.2001 |
| 4× `hash_static` | 0.3590 | 0.2752 | 0.1109 | 0.1933 |
| 8× `hash` | 0.6518 | 0.5102 | 0.2267 | 0.3708 |
| 8× `hash_static` | 0.6419 | 0.5029 | 0.2237 | 0.3658 |

- The combo leads every probabilistic metric while being *sharper* (lower
  spread at lower CRPS: more information, narrower posterior). Its one loss
  is the 4× spectrum (0.0093 vs hash 0.0083) — the static channels trade a
  little spectral fidelity for pointwise/probabilistic accuracy in the
  mild-degradation regime. Note hash ⊂ combo, so this ordering is the
  expected direction; the *discriminating* probabilistic pair is combo vs
  `static`, whose ensemble has not been run yet.
- Both arms are ~20 % underdispersive (a reliable 4-member ensemble would
  have spread ≈ ens-mean L2 / sqrt(1 + 1/M) ≈ 0.246 at 4×; measured 0.193) —
  expected with `ddim_eta: 0`, where members differ only through the
  noise-mixing initialization. The licensed upgrade is sampling-time
  stochasticity: `--eta` on `eval.compare_geo` overrides the config value
  without retraining anything.

Pending before these become thesis-final: the `static` ensemble (the
probabilistic tie-breaker), seed replicates (the combo-vs-static single-draw
margin sits near the sampling-noise floor), the `--shuffle-geo` control on
the combo arm, an `--eta` calibration sweep, and the `--gated` ablation.

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
comparison figures under `results/full_field/`. In multi-channel runs the
figures and headline metrics use the display channel (`eval.display_channel`),
with the all-channel normalized L2 alongside in the JSON.

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

### Multi-node / multi-GPU training (optional)

Any trainer can split the **same run** across several GPUs or nodes
(data-parallel DDP). Nothing changes by default — the distributed path only
activates when a trainer is launched under `torchrun`. For the standard
4-node × 1-GPU setup, run this on every node (from this directory, with all
nodes seeing the same `datasets/`):

```bash
# node 0 (the rendezvous host); nodes 1..3 run the same with NODE_RANK=1/2/3
MASTER_ADDR=node0.example NODE_RANK=0 \
  scripts/train_multinode.sh train.train_diffusion --config config/t2m.yaml
```

Single node with 4 GPUs instead:

```bash
NNODES=1 NPROC_PER_NODE=4 NODE_RANK=0 MASTER_ADDR=localhost \
  scripts/train_multinode.sh train.train_diffusion --config config/t2m.yaml
```

Works with every trainer (`train.train_diffusion`, `train.train_directmap`,
`train.train_residual`, `train.train_flow_matching`,
`train.train_stochastic_interpolant`) and composes with the usual flags
(`--geo`, `--encoder`, `--seed`, `--resume`, `--wandb`).

Semantics (details in `train/distributed.py`): each process draws a disjoint
shard of every epoch and gradients are averaged every step, so
`train.batch_size` is **per process** — the effective global batch is
`batch_size × world size` (scale the LR yourself if you want large-batch
rules). Only rank 0 validates, logs (stdout / TensorBoard / wandb), and writes
checkpoints; the checkpoint format is unchanged, so runs move freely between
single- and multi-GPU setups and `--resume` works across both.

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

## Scale, applied motivation, and scope

**Physical scale.** The target grid is 0.25° ≈ 28 km (zonal spacing 14–28 km
across the ±60° band): mesoscale downscaling, an order of magnitude coarser
than km-scale systems like CorrDiff (25 km → 2 km). Inputs per ratio:
2× ≈ 56 km, 4× ≈ 111 km, 8× ≈ 223 km, held-out 16× ≈ 445 km. A 128-px patch
spans ~3,600 km. (Cross-check: the hash grid's finest level ≈ 100 km cells and
the matched HEALPix Nside 64 ≈ 102 km — the ladder matching is within 2 % in
physical units.)

**Applied motivation: CMIP6 → CORDEX.** Impact assessment needs ~25 km fields;
the CMIP6 archive behind IPCC AR6 provides ~50–250 km, and the conventional
bridge (dynamical downscaling) costs CPU-months per scenario. The ratio ladder
brackets the archive: 4× ≈ a typical 1° CMIP6 model, 8× ≈ its coarse end,
16× the stress test beyond it. Because CMIP6 models do NOT share a resolution,
a multi-model ensemble needs a ratio-agnostic downscaler — which turns the
method comparison into a deployment rule: fixed known ratio → conditional
models; heterogeneous or unknown ratio → the guided unconditional model (the
only family that survives held-out 16×).

**Perfect prognosis, by necessity.** Training pairs GCM→truth do not exist:
free-running climate models are not time-synchronized with reality (their
15 March 2007 is a draw from a similar climate, not the same day). MOS-style
joint training (as in CorrDiff) requires a fine model DRIVEN by the coarse one
— the expensive runs this line of work exists to avoid. So training is
coarsened-truth → truth, and deployment composes with a separate
bias-correction stage (e.g. quantile mapping) that makes GCM output a
plausible coarsening before downscaling.

**The input is treated as truth — deliberately.** With projection on,
coarsen(output) == input exactly; measured worth ~2× in L2 here, where the
input genuinely is a coarsening. For biased real-GCM input the same property
preserves the bias, hence the two-stage pipeline above. `--project` is
effectively a dial between "the input is truth" and "the input is a
suggestion" (both ends measured: 0.404 vs 0.787 L2); soft consistency —
projecting onto a ball around the observation sized by input uncertainty — is
the principled middle ground, left to future work.

**How underdetermined is the task?** The observation fixes only block means:
1 − 1/r² of the degrees of freedom are free (94 % at 4×, 98 % at 8×, 99.6 % at
16× — the ker-A dimension). This is why the framing is generative (sample a
posterior) rather than regressive (invent a unique answer), and why ensemble
metrics (CRPS, spread–skill) are first-class. The encoder ladder measures the
stationary-geography share of the recoverable signal (~13 % of L2); the rest
is weather.

**What transfers to km-scale, what doesn't.** Transfer directly: the
data-consistency result, the η-underdispersion finding, the scale-matching
principle, the conditional model families, geo conditioning, and the tiling
machinery. Contingent: everything built on an exact linear degradation
(guided unconditional sampling, DDNM projection, null-space transport) —
real km-scale pairs two different models with no exact A. Likely to change:
the geo plateau — at 2 km, terrain drives nonlinear local circulations, so
the learned-vs-static gap is predicted to widen. Also noted: ERA5's effective
resolution is coarser than its grid (spectral truncation), so the finest
generated scales sit near what the reanalysis itself resolves — an argument
for scoring coherence, not just spectral power.

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

**Data-consistency projection (`--project` / per-step anchoring):**
- Wang, Y., Yu, J., Zhang, J. (2023). *Zero-Shot Image Restoration Using
  Denoising Diffusion Null-Space Model* (DDNM). ICLR.
  https://arxiv.org/abs/2212.00490 — the exact range/null-space projection
  x + A†(y - Ax) applied to the x0 estimate at each step; exact here because
  A is a block average and A† its nearest-upsampling pseudo-inverse.
- Choi, J., Kim, S., Jeong, Y., Gwon, Y., Yoon, S. (2021). *ILVR: Conditioning
  Method for Denoising Diffusion Probabilistic Models.* ICCV.
  https://arxiv.org/abs/2108.02938 — the ancestor: per-step low-frequency
  replacement on the noisy iterate.

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
