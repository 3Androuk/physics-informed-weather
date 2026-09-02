# Project context: ERA5 diffusion super-resolution + extensions

Working document capturing what has been built, what has been measured, why each
decision was made, and what is still open. Written 2026-08-02;
**status update in §10 (2026-09-02)** — read that first for the current picture.

**Convention:** every headline result in this document carries its figure.
Figures live in `context_figures/` (gitignored, local only, copied from the
cluster). When a new good result lands, copy its `spectrum.png` /
`qualitative_*.png` across and embed it next to the numbers.

**Redaction:** cluster identifiers are replaced with placeholders
(`<user>`, `<login-host>`, `<project>`) because this file is public; the
technical content is unchanged.

Metrics throughout are **`L2 / spectrum_log_l1`**, both lower-is-better.
`L2` is the mean over patches of per-patch RMSE in physical units (K for t2m).
`spectrum_log_l1` is the mean absolute error of the log10 radially-averaged
power spectrum over wavenumbers k >= 1 — it measures whether fine-scale
structure has the right *statistics*, which pointwise L2 is nearly blind to.

---

## 1. What the project is

A reimplementation of the physics-agnostic core of **Shu, Li & Barati Farimani
(2023)**, *A physics-informed diffusion model for high-fidelity flow field
reconstruction* (J. Comput. Phys. 478, 111972), applied to ERA5 downscaling,
plus original extensions.

### 1.1 The base method (Shu et al.)

A DDPM is trained on **high-fidelity patches only** — it never sees a degraded
field. At inference, super-resolution is reframed as *guided denoising*:

1. Coarsen the HF field by `ratio` (average pool), upsample back (nearest),
   optionally Gaussian-smooth. This is the low-fidelity guidance `x^(g)`.
2. Mix it with Gaussian noise at an intermediate timestep:
   `x_t = sqrt(abar_t) x^(g) + sqrt(1-abar_t) eps`.
3. Run DDIM (sigma = 0) from `t` down to `x_0`.
4. Repeat `K` times, each pass re-noising the previous output to a *lower*
   starting timestep — recursive refinement.

**Why this is robust:** because no degradation ever enters training, the model
has nothing to be out-of-distribution about. The ratio enters only through the
sampling schedule. This is the paper's central claim and our 16× result
confirms it (§4.3).

The physics-informed term (`c`, `w` in the paper's Algorithm 2) is dropped —
ERA5 has no exact closed-form governing PDE. Spectrum and value-distribution
metrics replace the PDE residual as physical-consistency checks.

### 1.2 The extensions (the MSc contribution)

1. **Learned geographic conditioning** — a per-pixel static location embedding
   concatenated as extra UNet input channels. Two interchangeable encoders.
2. **Per-step DDNM data-consistency projection** — enforce `coarsen(x0) == LF`
   at *every* DDIM step rather than trusting the initial noise-mixing.
3. **CorrDiff-style split model** — deterministic mean + generative residual.
4. **(new, in progress)** Transport methods — conditional flow matching and
   stochastic interpolants, as an alternative generative paradigm to DDPM.

---

## 2. Theory of each component

### 2.1 Why projection matters

Guided diffusion consults the low-fidelity input **once**, at initialization.
Over a long backward chain the sample drifts and consistency with the
observation is lost. DDNM-style projection re-imposes it at every step:

```python
x0_pred = x0_pred + upsample_nearest(lf - coarsen(x0_pred, ratio), hw)
```

This keeps the model's invented fine scales but pins the block averages to the
observation. **It is the single largest effect in the whole study** (§4.5).

### 2.2 Why the split model needs to be generative

Super-resolution is ill-posed: many HF fields coarsen to the same LF field. So
the residual `HF - mean` is not a deterministic function of the input.

If you trained the second stage to predict the residual under L2, the optimal
predictor is the *conditional mean of the residual* — which is ~zero, because
the first stage already predicts the mean. You would recover the deterministic
model and gain nothing, and what you did get would be blurry.

This is visible in our numbers: the direct map is a pure L2 regressor and its
4× spectrum is **0.0120 against the residual model's 0.0030** — 4× worse —
despite winning on L2.

**Distortion–perception tradeoff** (Blau & Michaeli 2018): you cannot have
optimal pointwise error and correct spectral statistics in a single output. A
draw from the correct distribution sits *further* from the truth in L2 than a
blurry conditional mean. This is a theorem, not a tuning failure, and it is why
we report single-draw *and* ensemble numbers (§4.6).

### 2.3 Geographic encoders — both are learned lookup tables

Both encoders are multiresolution pyramids of **learnable** feature tables
(`nn.Parameter`, same initialization, trained end-to-end). They differ only in
how a coordinate maps to table entries:

| | hash grid | HEALPix |
|---|---|---|
| indexing | unit-sphere xyz -> cubic lattice -> spatial hash | (lat,lon) -> HEALPix cell -> dense index |
| interpolation | trilinear, 8 corners, regular lattice | 4 neighbours, irregular tessellation |
| finest resolution (t2m cfg) | **0.895 deg** | **0.458 deg** |
| encoder params | 3,318,628 | 524,280 |
| finest-level load factor | 0.098 (~90% empty) | dense, exact |

Neither is "the geometric one" — the hash grid maps through unit-sphere xyz
first, so it already has no dateline seam and no pole over-sampling. The real
contrast is with **static fields** (orography, land–sea mask), which key on
*terrain properties* rather than *location identity* — see §7.3.

### 2.4 Transport methods (flow matching / stochastic interpolants)

Both transport **noise -> full HF field**, with the upsampled LF field as a
conditioning channel (same conditioning structure as residual diffusion, but
the target is the whole field).

**Flow matching:** straight-line path `x_t = (1-t) z + t x_data`, regress the
velocity `x_data - z`. Deterministic ODE sampling (Heun).

**Stochastic interpolant:** `I_t = (1-t) z + t x_data + gamma sin(pi t) eps`,
with two heads — velocity and `sigma_t * score_t`. The extra noise permits SDE
sampling that preserves the same marginals; ODE sampling uses only the velocity
head.

---

## 3. Infrastructure

### 3.1 Cluster

- Login node `<login-host>` (user `<user>`). **No GPU**, and its
  `/lib64/libstdc++.so.6` is too old for the venv's numpy
  (`CXXABI_1.3.9 not found`) — **torch will not import there**. Anything needing
  torch (e.g. `status.py`) must run on a GPU node.
- Login shell is **tcsh**: `cmd 2>/dev/null` fails with "cd: Too many
  arguments". Pipe a script to `bash -s` instead.
- GPU nodes: `ssh -J <login-host> <user>@<node>` — the `<user>@` is
  required, the jump host's `User` does not carry over.
- Login startup spews harmless `VBoxManage` errors on every command.
- Cluster git is **1.8.3.1** — no `git worktree` subcommand.
- venv: `physics-informed-weather/.venv`, install with `python -m pip`.

### 3.2 Directory layout (three independent checkouts)

```
physics-informed-weather/            # repo root, on main
├── era5-diffusion-downscaling/      #   holds the real datasets/
├── geo-hash-encoding/               #   branch worktree-geo-hash-encoding @ b81ad41
│   └── era5-diffusion-downscaling/  #   ALL diffusion results + 19 checkpoints
└── flow-stochastic-superres/        #   branch codex/flow-stochastic-superres @ 867e3d1
    └── era5-diffusion-downscaling/  #   transport runs; datasets symlinked
```

`load_config` resolves `paths.*` relative to each checkout's own root, so
`checkpoints_t2m/` and `results_t2m/` are automatically separate per checkout.
`datasets/` is shared by symlink (read-only use). The `meanmap*.pt` checkpoints
are symlinked into the flow checkout so residual transport can find them.

### 3.3 Results layout

`results_t2m/` is organised **one folder per run** under `evals/`:

```
results_t2m/
├── evals/<tag>/            # headline_table.{json,txt}, spectrum.png, qualitative_*.png
├── training_panels/{diffusion,directmap,residual}/
└── tb/
```

**Why:** `make_tables_figures` and (pre-`b81ad41`) `compare_geo` write
fixed filenames. Two runs in the same directory silently overwrite each other —
this already destroyed the primary-seed geo ablation JSON, which was
overwritten by the seed-43 replicate. Per-run folders make that impossible.

---

## 4. Experiments and results (t2m, projected, 256 patches)

### 4.1 Master table

| method | 2× | 4× | 8× | 16× (held out) |
|---|---|---|---|---|
| Bicubic | 0.2097/0.0354 | 0.4559/0.0691 | 0.7654/0.1017 | 1.1604/0.1449 |
| Direct map | 0.2938/0.0237 | **0.2662**/0.0159 | 1.0078/0.1350 | 1.4829/0.2027 |
| Direct map + geo | 0.2277/0.0185 | **0.2396**/0.0120 | 0.9732/0.1185 | 1.4559/0.1839 |
| Guided diffusion | 0.1883/0.0041 | 0.4060/0.0093 | 0.7567/0.0289 | 1.3443/0.0245 |
| Guided diffusion + geo | 0.1666/0.0037 | 0.3520/0.0089 | 0.6332/0.0256 | **0.8947/0.0092** |
| Residual (bicubic mean) | – | 0.4105/0.0045 | 0.8591/0.0064 | – |
| Residual (learned mean) | – | 0.3893/0.0046 | 0.7133/0.0117 | 1.6482/0.1098 |
| Residual + geo (bicubic) | – | 0.3373/**0.0030** | 0.5929/0.0108 | – |
| Residual + geo (learned) | **0.1496/0.0027** | 0.3031/0.0057 | **0.5041/0.0061** | 1.4428/0.1014 |

### 4.2 Headline findings

- **No single best model.** Direct map wins 4× L2; residual+geo+learned-mean
  wins 2× and 8×; guided diffusion+geo wins the held-out 16×.
- **Robustness tracks how much degradation information entered training.**
  Direct map (1 ratio) collapses immediately; residual family (3 ratios) holds
  inside {2,4,8} and fails at 16×; guided diffusion (0 ratios) is the only
  method still beating bicubic at an unseen ratio.
- **Geo conditioning helps every architecture**, most of all the split model
  (−31% L2 at 8×) and at extreme degradation.

![Geo vs no-geo spectra, guided diffusion](context_figures/geo_ablation_spectrum.png)

*The geo main effect on guided diffusion: geo and no-geo both track the
reference far better than bicubic, with the geo curves sitting closest across
the mid wavenumbers — the spectral counterpart of the 13–17% L2 gain.*

![Split model + geo at 8x](context_figures/phaseB_geo_qual_8x.png)

*The best L2 model at 8x (residual + geo + learned mean, 0.5041). Input, the
three method columns, and reference on a shared colour scale.*

![Split model + geo spectra](context_figures/phaseB_geo_spectrum.png)

*Same run's radial spectra. The direct-map curve falling away at high k is
regression-to-the-mean blurring; the residual curve tracks the reference.*

### 4.3 Held-out ratio (16×)

`degrade.py` asserts `128 % ratio == 0`, so `train_residual`'s docstring
suggestion of 6× is **unreachable** at 128px — valid divisors are
1/2/4/8/16/32/64, making 16 the smallest held-out option. A `t2m_ood16.yaml`
was generated from `t2m.yaml` with only the reconstructions block replaced
(verified byte-identical elsewhere).

Guided diffusion + geo is the only method beating bicubic (0.8947 vs 1.1604).
**Without geo it loses** (1.3443). Ratio-randomized training interpolates
within its set but does not extrapolate.

![Guided diffusion + geo at the held-out 16x](context_figures/ood16_geo_qual_16x.png)

*16x reconstruction, shared colour scale. The input is an 8x8 block field; only
guided-diffusion+geo recovers coherent structure, and it is the only column that
beats bicubic on both metrics at this ratio.*

*Caveat:* 16×'s `t_steps=[640]` / `smooth_sigma=10` were linearly extrapolated
from 4×(160, 0) and 8×(320, 5), untuned. This affects only the guided-diffusion
column; residual, direct-map and bicubic take no schedule parameters.

### 4.4 Permutation control — the capacity confound, resolved

The obvious objection to the geo result: geo models have ~3.3M extra encoder
parameters, so is the gain geography or capacity? Test: feed each patch
**another patch's** coordinates (`--shuffle-geo`, seeded permutation, 1 of 256
kept its own by chance).

| | correct | shuffled | no-geo | bicubic |
|---|---|---|---|---|
| 4× diffusion | 0.3522 | **0.5863** | 0.4037 | 0.4559 |
| 4× residual | 0.3031 | **0.6089** | 0.3893 | 0.4559 |
| 8× diffusion | 0.6340 | **0.9674** | 0.7639 | 0.7654 |
| 8× residual | 0.5041 | **0.9433** | 0.7133 | 0.7654 |
| 16× diffusion | 0.8947 | **1.5799** | 1.3443 | 1.1604 |

Performance falls **below** the no-geo baseline and below bicubic — wrong
geography is worse than no geography, which is only possible if the models
genuinely condition on location. Damage scales with degradation (+66% at 4×,
+53% at 8×, +77% at 16×): geography matters most where the input carries least.

Spectrum stays good under shuffling (4× diffusion 0.0141 vs 0.0085 correct) —
the model still synthesises realistic structure, just for the wrong place.

![Permutation control at 4x](context_figures/shufgeo_qual_4x.png)

*Every geo model reconstructing with ANOTHER patch's coordinates. Output is
still spectrally plausible weather — it is simply the wrong geography, which is
why L2 collapses while spectrum holds.*

**Limit:** this proves the models *use* location. It does not prove a
parameter-matched non-geo model couldn't do as well; that needs a trained
control.

### 4.5 Projection is load-bearing — **for guided diffusion specifically**

Same checkpoints, unprojected: 4× L2 **0.734** (hash) / 0.750 (hpx) — both
*worse than bicubic's 0.456* — while still beating it on spectrum. Projection
roughly halves L2 and is what makes guided diffusion competitive on pointwise
error at all.

**But it does essentially nothing for flow matching** (0.03% change, §4.8),
because flow matching conditions on the LF field at every step and never drifts.
State this contribution as fixing a formulation-specific weakness, not as a
general-purpose improvement.

![Unprojected guided diffusion at 4x](context_figures/unprojected_qual_4x.png)

*The failure this contribution fixes: UNPROJECTED guided diffusion at 4×. Both
geo models produce plausible texture but have drifted from the observation —
L2 0.734/0.750 against bicubic's 0.456. With per-step projection the same
checkpoints score 0.353/0.354.*

### 4.6 Ensembles (4 members, 64 patches — do not compare against 256-patch tables)

| | single L2 | ens-mean L2 | CRPS | spread |
|---|---|---|---|---|
| 4× residual geo | 0.3205 | **0.2706** | **0.1102** | 0.1411 |
| 4× diffusion geo | 0.3705 | 0.2979 | 0.1149 | 0.1893 |
| 8× residual geo | 0.5293 | **0.4523** | **0.2053** | 0.2527 |
| 8× diffusion geo | 0.6536 | 0.5349 | 0.2276 | 0.3522 |
| 16× diffusion geo | 0.9144 | **0.7093** | 0.3147 | 0.5598 |
| 16× diffusion no-geo | 1.3929 | 1.0962 | 0.4509 | 0.7929 |

The residual model beats guided diffusion on every probabilistic metric while
being *sharper* (lower spread). At the held-out 16×, the geo ensemble mean
reaches **0.7093 vs bicubic 1.1604 — 39% better at a ratio nothing trained on**.

![Geo vs no-geo guided diffusion at 8x](context_figures/ensemble_geo_vs_base_qual_8x.png)

*Single draws from the ensemble run's compare at 8× (projected), geo vs no-geo
side by side on a shared colour scale — the qualitative face of the geo rows in
the table above.*

### 4.7 Hash vs HEALPix — architecture-dependent, do NOT state unqualified

Source: `evals/phaseB_geo/` (hash) vs `evals/phaseB_hpx/` (HEALPix), plus
`evals/ood16_geo/` vs `evals/ood16_hpx/` for 16×, and
`evals/20260726_hpx_vs_hash/` for the dedicated guided-diffusion head-to-head.
Same test patches, same projection setting throughout.

| | 4× | 8× | 16× (held out) |
|---|---|---|---|
| guided diffusion, hash | **0.3530/0.00878** | **0.6346/0.02487** | **0.8947/0.0092** |
| guided diffusion, HEALPix | 0.3545/0.01293 | 0.6462/0.03303 | 0.9741/0.0260 |
| residual, hash | **0.3031**/0.0057 | **0.5041**/0.0061 | **1.4428/0.1014** |
| residual, HEALPix | 0.3117/**0.0027** | 0.5187/**0.0031** | 1.4497/0.1066 |

(The 16× residual pair is effectively tied — 1.4428 vs 1.4497 L2, 0.1014 vs
0.1066 spectrum — and both are far worse than bicubic's 1.1604, so neither
encoder rescues the split model out of distribution. HEALPix's spectral
advantage exists only *inside* the training ratios.)

**Guided diffusion: hash wins everywhere** — HEALPix spectrum is 47% worse at
4×, 33% worse at 8×, and 2.8× worse at 16×.

**Split model: HEALPix wins spectrum decisively and reverses the conclusion** —
0.0057 -> **0.0027** at 4× (2.1× better) and 0.0061 -> **0.0031** at 8×
(2.0× better), for a ~3% L2 cost.

![Split model with HEALPix: spectra](context_figures/phaseB_hpx_spectrum.png)

*The reversal, spectrally: the HEALPix-residual curve is the closest to the
reference in the whole study (0.0027 at 4×). Per the §4.7 diagnosis, its
0.458° finest level is the only level in either encoder at the scale of the
residual corrections — which is exactly why the same level hurts guided
diffusion.*

![Hash vs HEALPix, guided diffusion](context_figures/hpx_vs_hash_spectrum.png)

*Radial power spectra for the guided-diffusion head-to-head — the case where the
hash grid wins. Both encoders track the reference far better than bicubic; the
HEALPix deficit is in the mid-to-high-k band.*

Working hypotheses for the guided-diffusion loss: (a) HEALPix's finest level is
2× finer angularly (0.458 vs 0.895 deg), ~3.8× more free parameters per unit
area, which is the memorization failure mode already seen on Z500; (b) its
irregular tessellation has derivative discontinuities at the 12 base-face
boundaries, injecting broadband high-k error — and it uses the `ring` scheme
where `nested` would give hierarchically-aligned pyramid levels.

**This rests on n=1 training run per arm.** A 2× spectrum difference on a single
seed cannot be separated from training noise.

#### Diagnosis (2026-08-14): scale misallocation, not the tessellation

Encoder forensics on the two trained checkpoints
(`evals/20260726_hpx_vs_hash/encoder_diagnostics.png`; script re-runnable):

![Encoder forensics: hash vs HEALPix](context_figures/hpx_encoder_diagnostics.png)

*Top row: both encoders learned the same geography (Tibetan plateau visible in
both), but HEALPix's embedding spectrum tail runs ~10× above the hash beyond
k≈10. Bottom row: the hash's finest level is smooth ~2° blobs; HEALPix's shows
tessellation-aligned diagonal striping — structure from the pixelisation, not
the planet. Bottom-right: table norms are FLAT for both (fine/coarse ≈ 1.0), so
this is NOT Z500-style weight memorization.*

Three facts, all configuration rather than tessellation:

1. **Ladder waste:** HEALPix levels 0–2 have 58.6° / 29.3° / 14.7° cells — at or
   above the 32° patch span, so ~3 of 8 levels are near-constant per patch. In
   the 1–7° band guided diffusion needs, hash has 8 levels, HEALPix ~4.
2. **Cap violation:** `t2m.yaml`'s own comment caps cells at ~150 km; the hash
   finest is 0.895° ≈ 100 km, but `healpix_nside_max: 128` is 0.458° ≈ **51 km**
   (~2 ERA5 px). The config contradicts itself.
3. **Rough interpolation:** at k≥32 the finest HEALPix level carries **14×** the
   relative energy of the hash's (0.043 vs 0.003) with visible tessellation
   striping — 4-neighbour weights on an irregular tessellation vs trilinear on
   a lattice.

**This dissolves the architecture-dependence.** Guided diffusion needs the
mid-scale levels HEALPix lacks and is polluted by the rough sub-degree channel
→ hash wins. The residual model's learned mean carries the coarse scales; its
job is placing ~50 km corrections, and HEALPix's 0.458° level is the only level
in either encoder at that scale → HEALPix wins spectrum. One misallocated
ladder explains both directions.

**Predictions (cheap to test):**
- *Ladder-matched HEALPix* — `ring` scheme accepts any integer Nside, so
  Nside = 8,11,15,20,26,36,48,64 gives b≈1.346, identical scale band to the
  hash, ~220k dense params, honors the cap. Needs: relax the pow-2 rounding in
  `healpix_nside_ladder`, separate `healpix_index_matched.npz`, plumb
  `healpix_index_path` through the trainers. If it matches hash on guided
  diffusion, tessellation was never the issue; any residue = interpolation
  seams, cleanly isolated.
- *Hash with `finest_resolution: 256`* (0.45°) should close the residual
  model's spectrum gap toward HEALPix.
- If seams survive the matched test: **spherical Fibonacci pyramid** — N_l
  Fibonacci-lattice points per level (any geometric ladder, like NGP), dense
  tables (no collisions, like HEALPix), Gaussian K=8-NN interpolation (smooth,
  no seams). Slots into the existing pipeline: `make_fib_index.py` mirroring
  the HEALPix precompute, same packed payload with K=8, `HealpixGrid.forward`
  generalized to K, one new `_ENCODER_TAG` entry.

*Caveat: spectra alone cannot fully separate "seams" from "legitimately finer
content" — a 0.458° grid is entitled to some sub-degree energy. The striping
and the 14× ratio implicate roughness beyond resolution; the matched-ladder
run is what makes the separation rigorous.*

### 4.8 Transport methods vs diffusion (2026-08-03)

Evaluated with `eval.compare_transports` at ratios 2/4/8/16. **The bicubic column
matches the existing diffusion tables exactly at every ratio** (0.2097 / 0.4559 /
0.7654 / 1.1604), which proves the test patches, ordering and metric code are
identical across the two checkouts — the tables are directly comparable.

| method (geo) | 2× | 4× | 8× | 16× (held out) |
|---|---|---|---|---|
| Guided diffusion + geo | 0.1666/0.0037 | 0.3520/0.0089 | 0.6332/0.0256 | **0.8947/0.0092** |
| Residual + geo (learned mean) | **0.1496**/0.0027 | **0.3031**/0.0057 | **0.5041**/0.0061 | 1.4428/0.1014 |
| Flow matching + geo | 0.1719/**0.0011** | 0.3441/**0.0017** | 0.5726/**0.0041** | 1.4674/0.0867 |

| method (no geo) | 2× | 4× | 8× | 16× |
|---|---|---|---|---|
| Guided diffusion | 0.1883/0.0041 | 0.4060/0.0093 | 0.7567/0.0289 | 1.3443/0.0245 |
| Residual (learned mean) | 0.2386/0.0125 | 0.3893/0.0046 | 0.7133/0.0117 | 1.6482/0.1098 |
| Flow matching | 0.1888/0.0019 | 0.3918/0.0050 | 0.7140/0.0058 | 1.6457/0.1055 |
| Stochastic interpolant | 0.2061/0.0016 | 0.4164/**0.0027** | 0.7498/**0.0045** | 1.6527/0.0983 |

**Flow matching + geo sets a new best spectrum at every in-distribution ratio**
(0.0011 / 0.0017 / 0.0041) — the previous best anywhere was 0.0027. Its L2 sits
between guided diffusion and the split model.

![Flow matching + geo radial spectra](context_figures/flow_geo_spectra.png)

*The clearest single figure in the study. Flow matching at 2x/4x/8x lies on top
of the reference across the whole wavenumber range, while every bicubic curve
falls below it from k~4 onward — that energy deficit IS blurring. The 16x curve
(yellow) oscillates violently around k~7-10: the extrapolation failure is
visible spectrally, not just in the L2 number.*

![Flow matching + geo, 8x](context_figures/flow_geo_qual_8x.png)

*8x reconstruction. Fine structure is synthesised at the right amplitude, which
is what the 0.0041 spectral score measures.*

![Flow matching + geo training panel](context_figures/flow_geo_training_panel.png)

*Training-time panel (epoch 90). Bottom row, 16x column: the input is an 8x8
block field, yet the sample reproduces coastline, ridge and small water bodies
absent from the input. That is the geo encoder supplying terrain knowledge —
the visual counterpart of the permutation control in section 4.4.*

**Projection is irrelevant for flow matching.** Per-step vs final projection,
flow + geo: 4× 0.3441/0.0017 vs 0.3442/0.0016; 8× 0.5726/0.0041 vs 0.5750/0.0048
— a 0.03% difference, against roughly *halving* L2 for guided diffusion (§4.5).
Mechanism: flow matching takes the LF field as a conditioning channel at **every**
step, so it never drifts from the observation; guided diffusion consults the
input only once, at noise-mixing initialization. **The DDNM contribution fixes a
problem specific to the guided-diffusion formulation**, not a general one.

**Flow matching does not extrapolate.** At 16× it lands with the split model,
*worse than bicubic* (1.4674 vs 1.1604) — it also conditions on the degraded
field and was trained on {2,4,8}.

![Flow matching + geo at the held-out 16x](context_figures/flow_geo_qual_16x.png)

*The failure case, worth keeping. Compare against the guided-diffusion 16x panel
in section 4.3: flow matching produces plausible-looking texture but does not
recover the correct large-scale field at a ratio it never trained on.*
 Guided diffusion + geo remains the only method
beating bicubic at an unseen ratio, now demonstrated against four competing
paradigms rather than two. This strengthens §4.2: robustness tracks how much
degradation information entered training, and conditioning on the degraded field
is itself a form of that information.

**SI vs flow is a clean distortion–perception trade:** SI wins spectrum at every
ratio (0.0027 vs 0.0050 at 4×) and loses L2 at every ratio (0.4164 vs 0.3918),
matching the visible graininess in its sample panels (§7.1).

### 4.9 Ratio 2× — brittleness is distribution shift, not difficulty

The direct map **loses to bicubic at 2×** (0.2277 geo / 0.2938 no-geo vs
0.2097) — and 2× is *easier* than its 4× training ratio. Its 8× collapse can no
longer be dismissed as "harder problem"; it fails on the easy side too.

![Ratio 2x qualitative](context_figures/r2_geo_qual_2x.png)

*2× reconstruction (geo run, shared colour scale). The input is barely
degraded, yet the direct map — trained at 4× — visibly underperforms plain
bicubic on it: brittleness in the EASY direction, i.e. distribution shift.*

### 4.10 Seed variance

Two seeds of guided-diffusion-geo at 4×: 0.3512 (primary) vs 0.3501 (s43) —
**~0.3%**, far below every effect above. Caveat: this is measured on guided
diffusion only; assuming the residual family is equally stable is an assumption,
and it is exactly the assumption §4.7 leans on.

---

## 5. Implementation decisions and why

| decision | rationale |
|---|---|
| Held-out ratio = **16**, not 6 | `degrade.py` asserts `128 % ratio == 0`; 6 is unreachable at 128px |
| Per-run result folders | fixed filenames silently clobbered the primary geo ablation |
| `--shuffle-geo` seeded off `cfg['seed']` | reproducible, and every model in a run sees the same mismatch |
| Ensembles only for stochastic methods | bicubic/direct map are deterministic; repeated draws are identical and CRPS/spread meaningless |
| Ensemble metrics in JSON only, not the text table | different shape; would not fit ratio×method columns |
| Residual transport conditions on the **mean field**, not LF | strictly more informative, and matches `ResidualConditionalUNet` so the comparison to `residual_geo_lm` is like-for-like |
| `res_std` estimated once over 64 patches × all ratios | matches `train_residual`; the mean-field channel tells the model which regime it is in, so one shared scale suffices |
| `res_std` restored from checkpoint on `--resume` | re-estimating from a different 64-patch draw would silently change the training target |
| Residual transport disables **in-sampler** projection | the constraint `coarsen(x)==coarse` holds for the composed field, not the residual; projection is applied once to `mean + res_std*residual` instead |
| meanmaps symlinked, not copied | ~700 MB each, read-only use |
| Separate clone for the flow branch | cluster git 1.8.3.1 has no `git worktree`; keeps checkpoints/results isolated automatically |

---

## 6. Current state

### 6.1 Trained and evaluated (geo-hash-encoding checkout)

19 checkpoints, all at 200/200 epochs — 17 for t2m (guided diffusion ± geo ±
hpx ± s43, direct map ± geo, 3 meanmaps, 5 residual variants), 2 for z500.
Every one has been evaluated; results in `results_t2m/evals/`.

### 6.2 In progress (flow-stochastic-superres checkout)

Four transport runs launched 2026-08-02, ~200 epochs each:

| run | node | epoch | val 2×/4×/8× |
|---|---|---|---|
| flow, no geo | aylesbury-l | 78 | 0.01127 / 0.02207 / 0.03038 |
| flow, geo | brent-l | 78 | 0.00968 / 0.01913 / 0.02507 |
| SI, no geo | shoveler-l | 78 | 0.98595 / 0.99784 / 1.01951 |
| SI, geo | pochard-l | 52 | 0.98364 / 0.98956 / 1.00817 |

### 6.3 Residual transport mode — implemented, not yet launched

`--residual` / `--mean-ckpt` added to `train_transport.py`, with sampling and
eval support in `sample/transport.py` and `eval/compare_transports.py`.
Checkpoint stems gain `_res` and `_lm`. Smoke-tested on the cluster:

```
Learned mean: meanmap_geo.pt (geo=True, frozen)
Residual std (normalized units): 0.0367
flow | ratios=[2, 4, 8] | patches=26296 | params=65,602,085 | geo=True
```

`res_std = 0.0367` means the learned meanmap explains almost all the variance —
the regime where the split model should help most, and where flow matching's
straight-line path becomes much shorter (the rectified-flow argument).

The existing `tests/test_transport.py` (5 tests) still passes, so the
full-field path is intact.

---

## 7. Open issues

### 7.1 The stochastic interpolant is NOT broken (resolved)

SI's total loss sits near 1.0 while flow matching's is ~0.011. This looks
alarming but the two losses are **not comparable** — they regress different
targets with different irreducible noise floors.

Tensorboard split (`results_t2m/tb/stochastic_interpolant*`):

| | velocity loss | score loss |
|---|---|---|
| stochastic_interpolant | 1.596 -> **0.9921** | 0.2013 -> **0.0125** |
| stochastic_interpolant_geo | 1.754 -> **1.0187** | 0.2003 -> **0.0087** |

The **score head learns very well** — 0.0125 against a unit-variance target is
98.7% of variance explained. The **velocity** head is the one near 1.0, and that
is structural:

```python
velocity = target - z + bridge_dot * eps      # bridge_dot = gamma*pi*cos(pi t)
```

`eps` is a second, independent noise that enters `x_t` only as
`gamma*sin(pi t)*eps` — a *different* linear combination than the velocity
target needs. Given `x_t` you cannot recover both `z` and `eps`, so the model
predicts the conditional expectation and the orthogonal part of
`bridge_dot*eps` is a permanent floor. With `gamma = 0.5` that term carries
variance ~(0.5*pi)^2 * E[cos^2] ~ 1.23 — right where the loss settled.

Sample quality confirms both methods work:

| run | samples/spectrum_log_l1 |
|---|---|
| flow_matching | 0.0499 -> 0.0436 |
| flow_matching_geo | 0.0284 -> **0.0267** |
| stochastic_interpolant | 0.0641 -> 0.0339 |
| stochastic_interpolant_geo | 0.0322 -> **0.0256** |

Visual inspection of the epoch panels: both reconstruct plausible fields at
4x/8x/16x, and at 16x (an 8x8 input) both recover coastline, ridge and small
water-body structure that is absent from the input — direct visual evidence for
the geo conditioning result in §4.4.

**Open caveat:** SI's samples are visibly grainier than flow matching's.
`spectrum_log_l1` rewards the right *amount* of high-k energy, not the right
high-k *structure*, so SI's better spectral score may be partly speckle. Check
against L2 and CRPS in the real eval before claiming SI wins on spectrum.

![SI training panel](context_figures/si_geo_training_panel.png)

*Stochastic interpolant + geo, epoch 70. Reconstructions are structurally right
at 4×/8×/16× but carry a fine speckle absent from the flow-matching panel in
§4.8 — the visible form of the stochastic bridge, and the reason to
cross-check SI's spectral wins against L2/CRPS.*

**Lesson for the write-up:** never compare raw training losses across transport
parameterizations — compare sample metrics.

### 7.2 Other known gaps

- **HEALPix reversal rests on n=1 per arm** (§4.7). Needs a seed replicate of
  `residual_geo_lm` and `residual_geo_hpx_lm` — ~1 day, the only outstanding
  item that fixes a contradiction rather than adding scope.
- **No confidence intervals.** `l2_norm` averages 256 per-patch RMSEs and
  discards the spread. Returning it would put a CI on every number for free.
- **CRPS at 4 members is noisy.** Large gaps (16× geo vs no-geo, −30%) are safe;
  small ones (residual 0.1102 vs diffusion 0.1149) are not.
- **z500 is stale.** `config/z500.yaml` still has the fine encoder
  (`n_levels: 12`, `finest_resolution: 512`) that was diagnosed as the cause of
  its negative geo result. The fix only ever went into `t2m.yaml`.
- **`init_wandb` is called at the END** of `make_tables_figures` / `compare_geo`,
  after all compute. Eval runs therefore record a runtime of seconds and give no
  live progress.
- **Training sample panels still collide.** `results_t2m/residual_epoch*.png`
  have interleaved mtimes from two concurrent runs writing the same filenames.
  Checkpoints are fine (distinct names); the PNGs cannot be attributed.
- **Local branch diverged from remote.** Local tip `7d5a601` is an amend of
  remote `867e3d1` (same tree, corrected message). A future push needs
  `--force-with-lease`.

### 7.3 The experiment that would most strengthen the geo claim

Now that the permutation control has passed, the natural control is
**CorrDiff-style static fields**. The WeatherBench2 store already carries them
on the same grid — verified present:

```
geopotential_at_surface, land_sea_mask, standard_deviation_of_orography,
slope_/angle_/anisotropy_of_sub_gridscale_orography, high_/low_vegetation_cover,
soil_type, lake_cover, lake_depth
```

Four-way: (a) no geo, (b) hash, (c) HEALPix, (d) static fields, (+e) static +
learned. Every outcome is publishable — if (d) matches (b) with ~1000× fewer
parameters, the learned embedding is an expensive way to rediscover orography;
if (b) > (d), the hash grid captures something the covariates miss. For t2m the
signal is mostly lapse rate + land–sea contrast + albedo, which are literally
three of those fields, so there is a strong prior worth testing.

Implementation slots into existing machinery: a `make_static_fields.py`
mirroring `make_healpix_index.py`, `PatchDataset` cropping by `origins` exactly
as it does the HEALPix payload, and `build_unet`'s `extra_in_channels`.

---

## 8. Suggested next steps

1. **Let all four transport runs finish** — both methods are healthy (§7.1);
   then run `eval.compare_transports` to get L2/spectrum on the same test
   patches as everything else, so transport slots into the master table and the
   ratio curve.
2. **Launch residual transport** with `--mean-ckpt meanmap_geo.pt` once GPUs are
   free; code is ready and smoke-tested.
3. **Seed replicates** of the two residual geo models — resolves §4.7.
4. **Per-patch spread in `l2_norm`** — free CIs on everything.
5. **Static-fields comparison** (§7.3) — the highest-value new experiment.
6. **z500 with the coarse encoder**, or report the negative result with the
   fine-encoder caveat stated explicitly.

---

## 10. Status update — 2026-09-02

Audit of all three checkouts and both clusters, written the same day. Everything
here was read off the clusters directly; numbers are quoted from the result
files named. Figures for the BriCS tables live in `results_wb220/` on BriCS and
have NOT been copied into `context_figures/` yet.

### 10.1 Where things stand

| checkout | branch / HEAD | state |
|---|---|---|
| Mac `physics-informed-weather` | `multivar-20ch` @ `867e3d1` (Jul 31) | stale: 51 behind `origin/codex/flow-stochastic-superres` (`85b5e58`); 8 uncommitted files that are an OLDER version of the residual-transport work now upstream — stash and check out origin |
| UCL `flow-stochastic-superres` | `85b5e58`, in sync | active: t2m deployment demo; five forecast runs in flight at time of writing |
| UCL `wb2-20var` | — | files fine, but **`git` is broken there** (`.git/worktrees/wb2-20var` pruned) — re-attach before committing/pulling |
| UCL `geo-hash-encoding` | `b81ad41` | dormant, 2 modified |
| BriCS `$HOME/physics-informed-weather` | `brics-aip2` @ `3149859`, **0 ahead / 3 behind** origin | **13 tracked files modified (+1125/−58) and 31 untracked source files, none committed, none backed up**: `core/`, `data/fetch_nwp.py`, `fetch_poles.py`, hash2d / hash_compact encoders, SpikeGuard + divergence rollback, `tests/test_hash2d.py`, `test_spike_guard.py`, `smoke_ddp.py`, `eval/merge_eval.py`, `run_*.sh`, three configs. The whole Aug 31–Sep 2 BriCS effort exists only in a `$HOME` BriCS does not back up. **Commit it.** |

BriCS access: `ssh <project>.aip2.isambard`, cert via `clifton auth && clifton ssh-config write`,
12 h validity. One partition `workq` (24 h max), per-project cap 32 GPUs,
billing proportional to allocation. **Never poll `squeue`** — the AUP names
disruptive polling as grounds for suspension; check jobs by tailing the `.out`.
Login-node jobs die with the SSH session (no linger): hold a session from the
Mac or use sbatch.

### 10.2 BriCS 20-var ladder — complete, six arms, all epoch 200

Global batch 128 (32 × 4 GH200), bf16, lr 1e-4. Evaluated in one sharded run
on 1024 patches spread over the period, projection ON
(`results_wb220/compare_FINAL_merged.json`, Sep 1). L2(norm) / spectrum:

| arm | 4× L2n | 4× spec | 8× L2n | 8× spec | train val |
|---|---|---|---|---|---|
| **compactcombo** (compact hash + static) | **0.0423** | 0.00063 | **0.0944** | **0.00214** | 0.01079 |
| static | 0.0426 | **0.00062** | 0.0954 | 0.00232 | 0.01086 |
| hashcompact | 0.0431 | 0.00077 | 0.0950 | 0.00244 | 0.01083 |
| hash2d (2-D chart) | 0.0435 | 0.00073 | 0.0954 | 0.00238 | 0.01084 |
| hpx (matched ladder 8→64) | 0.0435 | 0.00087 | 0.0955 | 0.00264 | 0.01087 |
| no-geo | 0.0463 | 0.00112 | 0.1000 | 0.00298 | 0.01128 |
| bicubic | 0.0507 | 0.01713 | 0.1027 | 0.03380 | — |

- Ensemble diagnostics on compactcombo (8 members, 128 patches): 8× single L2
  2.9976 → ens-mean 2.4352, CRPS 1.0496, spread 1.5075. Inflation sweep: best
  λ=1.5 (CRPS −3.3%, spread/err 0.93); λ≥2 hurts.
- Also trained, **not yet scored**: `meanmap`, `residual_lm`,
  `stochastic_interpolant_res_lm` (all with `_best`). The eval job `loadchk`
  (6244305) failed loading them: their state dicts carry a `unet.` prefix
  (wrapper module) that `compare_geo.py`'s plain-UNet loader does not strip.
  One loader fix unblocks the whole split-model branch of this study.

### 10.3 The two 20-var studies must never be pooled — now measured, not asserted

The arms are **split across machines with different ladders**:

| arm | where | ladder |
|---|---|---|
| hash 3-D (`diffusion_geo.pt`) | **UCL only** | 1 GPU, global 20, fp32 |
| no-geo, static | both | — |
| HEALPix, hash2d, hashcompact, compactcombo | **BriCS only** | 4 GH200, global 128, bf16 |

`Bicubic` is byte-identical in the UCL and BriCS comparison files
(4× 0.05157057510178596), so eval protocol, patches and metric are the same and
only training differs. The `static` arm ran on both: UCL 0.044788 vs BriCS
0.043704 at 4× — **the ladder alone is worth 2.4%**, comparable to the whole
hash-over-no-geo effect (3.8%). So: no valid hash-vs-HEALPix comparison exists
anywhere on wb220. Getting one means training the 3-D hash arm ON BriCS.
Valid statements today: within BriCS, static ≥ HEALPix (2.0% L2 at 4×, tie at
8×, 18–26% better spectrum); within UCL, hash < no-geo < static.

### 10.4 Why HEALPix lost, and what the 2-D chart encoder is

The matched ladder (Nside 8,11,14,20,26,35,48,64 — 814→102 km) was already in
the BriCS HEALPix run, so the §4.7 "ladder misallocation" explanation is
**refuted for wb220**; the spectrum penalty survived the fix. What remains is
the interpolation: healpy's 4-neighbour weights on an irregular mesh switch
neighbour sets discontinuously at ring and base-face boundaries, giving kinks
that are stationary in lat/lon — a fixed high-k pattern the UNet latches onto.
The hash's trilinear tensor-product weights are smooth.

**Capacity was never the axis.** Measured over the ±60° grid:

| encoder | allocated | live params | distinct surface locations | utilization |
|---|---|---|---|---|
| hash 3-D (16→128, T=2^19) | 3,318,628 | 535,790 | 137,296 | 16.1% |
| HEALPix matched | 217,968 | 191,008 | 95,504 | 87.6% |
| **2-D chart, 86 km finest** | **267,072** | **267,072** | **133,536** | **~100%** |

The hash touches 288,752 corner entries but resolves only 137,296 distinct
cubes — **2.10× radial redundancy** (a 3-D grid over a 2-D shell: cells at
different radii encode the same surface point and are always queried together).
Its real spatial advantage over HEALPix is 1.44×, not the 2.8× the parameter
count suggests. Fixing the dimension mismatch (`input_dim: 2`, equal-area chart
`u = lon/360` periodic, `v = sin(lat)/sin 60°` fitted to the band, `n_u ≈ 3.63
n_v`) gives the hash's resolution at 12× fewer parameters, all live, every
level dense (no hashing, no collisions), smooth bilinear interpolation kept.
Cost: cos²φ shape distortion (4× at 60°) — the one thing HEALPix does better.
**This is `hash2d`, now trained on BriCS:** ties HEALPix on L2 at both ratios
and beats it on spectrum by 16% / 10% — the interpolation prediction held — but
does not beat static. The framing: once the wasted dimension is removed, hash
and HEALPix converge on the same object (a dense 2-D multiresolution pyramid);
the only remaining choice is the chart, and the rectangle wins on smoothness.

### 10.5 The no-geo baseline collapse — diagnosed

Collapsed at epochs ~92 and 82 (two runs), and again on Aug 31 (saved by the
new rollback); both geo arms ran 200 clean; the UCL no-geo arm (fp32, global
20) trained fine. Signature: no precursor (epochs 76–81 were the calmest of the
run, grad mean 0.102, max 0.128), then within ~100 steps loss 0.0149 → 0.408 →
1.003 and grad norms fall to ~0.03 — the degenerate ε≈0 solution at loss 1.0.
EMA followed it down. `clip_grad_norm_(…, 1.0)` was active (the logged `grad`
is its pre-clip return value) and did not help; static survived a 1.578 spike
at epoch ~10, so spikes per se are survivable — **timing is what matters.**

Mechanism: `AdamW` at stock β₂=0.999, ε=1e-8, constant lr, no warmup. Adam is
scale-invariant, so clipping the raw norm does not bound the step; the step is
set by the outlier's ratio to the recent gradient scale. Late in training v̂
has shrunk to match ~0.10 gradients while β₂=0.999 makes it respond over ~1000
steps (≈5 epochs) and m̂ over ~10 — the ratio can reach ≈1/√(1−β₂) ≈ 32×
nominal with nothing in the denominator to damp it. The no-geo arm has the
fattest gradient tail (max 3.157 vs static 1.578 vs hpx 0.374), so it draws the
outlier. Fixes, cheapest first: ε 1e-8→1e-6; β₂ 0.999→0.99; a skip-step guard
(grad_norm > 5× running median → zero grads, skip) — the least confounding,
since on a healthy run it never fires; brief warmup on resume.
**Resume trap:** the divergence guard exits 0, the sbatch says "finished
cleanly", and `--resume` reloads the *collapsed* rolling `<stem>.pt` (weights
finite, merely degenerate). Restore `<stem>_best.pt` over it first; `_save_ckpt`
writes the best file with the full opt/scaler/epoch payload.

### 10.6 NWP forecast data and the deployment demo

**UCL (t2m, flow checkout):** HRES 1.5° forecasts downloaded for 24/72/120/240 h
(`datasets/forecast_hres_t2m`, 575 MB, `FCSTDL_DONE rc=0`); 24 h and 120 h
scored (§ nwp memory: geo-combo CRPS 0.5417 / 0.8344 vs native 0.25° HRES
0.5966 / 0.9150). **72 h and 240 h are downloaded but unscored.** No IFS ENS
download started. New 24 h DDNM rows today (lat-weighted t2m RMSE / control /
CRPS): geo-combo diffusion **1.1533 / 0.6498 / 0.5417**; flow matching 1.2375 /
0.7850 / 0.6131; stochastic interpolant 1.3730 / 1.0208 / 0.7308; bicubic
1.2630 / — / 0.6995. SI loses to bicubic on both, and its control error of
1.02 K (pure downscaling error) says the SI model is the weak link, not the
input. Five more runs were in flight at time of writing (`run_fcst_{fm,si,
geohash,rescm,resdiff}.sh`, `eval.downscale_forecast`, 8 inits each).

**BriCS (20-var):** `$PROJECTDIR/<user>/datasets/nwp_hres/` is **complete** —
256 inits (00/12Z, linspace over 2016–2017), all four arrays, 28 GB:
`analysis_coarse` (hres_t0, 1.5°), `forecast24_coarse` (hres +24 h, 1.5°),
`truth_fine` (hres_t0 at 0.25°, ±60° band, the verification target — NOT ERA5,
so the pair is internally consistent), `forecast24_fine` (so the forecast can be
coarsened with our operator, separating source-model shift from
coarsening-operator shift). **HRES has 19 of the 20 channels**: TCWV is absent
and `data/fetch_nwp.py::reconstruct_tcwv` rebuilds it from q on 13 levels + sp
(mask below-surface levels, integrate, add the sub-1000 hPa layer, affine
0.883·col − 0.25 fitted on ERA5): 6.7% of std, r=0.9978, vs ~4–5% model error
on that channel; naive trapezoid 16.5% and uncorrectable. The first attempt
died after stage 1 with the SSH session; re-running `run_nwp.sh` resumed
(stages skip existing files) and finished in 58 min.
`raw_wb220_global` (full-globe 20-var for the mesh arm) is **dead** at
`train_2008.npy.tmp` day 208/366 since Sep 1 13:07 — same cause, resumable
(`download_era5` caches per year; single process is 2.4× faster than sharding).

### 10.7 Checkpoints — inventory, deletions, what is unscored

61 real checkpoints, ~97 GB after deletion, all on the clusters, none on the
Mac, **none backed up**. Deleted on BriCS (13 GB): `diffusion_COLLAPSED_6169725`,
`diffusion_COLLAPSED_6189791`, `meanmap_lr1e4_COLLAPSED`, `meanmap_lr1e4_best`
(val 0.00275 vs the kept meanmap's 0.00200; unreferenced).

BriCS `checkpoints_wb220` (16): diffusion, static, hpx, hash2d, hashcompact,
compactcombo (each with `_best` where trained after the rollback landed),
meanmap, residual_lm, stochastic_interpolant_res_lm.
UCL: `wb2-20var` 3 (diffusion, diffusion_geo, diffusion_geo_static);
`geo-hash-encoding/checkpoints_t2m` 14; `checkpoints_z500` 2 —
**`checkpoints_z500_coarse/diffusion.pt` is a symlink to the fine one; the
coarse-encoder z500 model was never trained**; `flow-stochastic-superres/
checkpoints_t2m` 13 real + 8 symlinks into geo-hash; `dlwp-hpx` 3 pairs.

**No evaluation artefact exists for:** BriCS `meanmap`, `residual_lm`,
`stochastic_interpolant_res_lm` (loader bug above); UCL `diffusion_s43`,
`diffusion_geo_s43` (the §4.10 "two seeds agree to ~0.3%" claim has no JSON
behind it — only a TB events file); `stochastic_interpolant_geo` (no
`evals/transport_si_geo`); and ambiguously `flow_matching_res_geo_lm`,
`stochastic_interpolant_res_geo_lm`, because `compare_transports.py` writes
hardcoded row labels rather than checkpoint stems (on Aug 28 its two rows were
`ns_lm` and `cm_lm`). Fix the tool to record the checkpoint path.
`diffusion_geo_hpx_n128` appears in a flow JSON but no longer exists on disk.

### 10.8 Full-field stitching — tiled vs MultiDiffusion

`sample/full_field.py` has three modes. **direct**: whole field through the
UNet at once (attention sees far more tokens than at train time — suspect).
**tiled**: overlapping training-sized tiles (128, overlap 32), each its own
guided chain, blended once at the end with a separable cosine-ramp window
(flat interior, ½−½cos ramp over the overlap, floored at 1e-4); all tiles crop
their starting noise from ONE global noise field so overlaps agree; tile
origins snap to multiples of the ratio so per-tile coarsening commutes with
cropping; a final exact global block-average projection re-pins the field.
**fused** = MultiDiffusion (Bar-Tal et al. 2023): `_FusedTileModel` wraps the
UNet so one global DDIM chain runs on the full field and every noise prediction
is assembled from the tiles with the same window — seam-free by construction,
same compute.

Who uses what: `compare_geo.py` (every ladder table) — no stitching, patches
scored independently; `eval/downscale_forecast.py` (all forecast-demo numbers)
— **tiled only** (it imports nothing else); `eval/full_field.py` — both plus
direct. **Verdict (Aug 31 ladder): tiled ≈ fused within 0.5%** (4× diffusion
tiled 0.4527/0.0298 vs fused 0.4473/0.0296) — fusion buys nothing once shared
noise, ratio-aligned origins and the final projection are in place, so the demo
standardised on tiled. Two things only tiled supports: the Weather-DDNM
covariance projector (needs the patch geometry) and DDNM+ skipping the final
projection under noisy observations.

### 10.9 Open items, in the order I would take them

1. **Commit the BriCS working tree** (§10.1) — it is the only copy.
2. Fix the `unet.` prefix in `compare_geo.py`'s loader; score `residual_lm`,
   `stochastic_interpolant_res_lm`, `meanmap` on BriCS (~35 min each, 1 GPU).
3. Train the 3-D hash arm on BriCS so hash-vs-HEALPix exists on one ladder
   (~2 h 45 on one node); add `xyz` as the zero-capacity positional null.
4. Score the 72 h / 240 h HRES leads on UCL (data already down).
5. Re-attach the `wb2-20var` worktree metadata on UCL.
6. Resume `raw_wb220_global` from a held session or sbatch (~5 h).
7. Make `compare_transports.py` record checkpoint paths; rerun the ambiguous
   transport arms if their numbers are needed.
8. Train the coarse-encoder z500 model (§10.7) before any z500 geo claim.
9. IFS ENS: the definitive CRPS test (§ nwp memory) — nothing started.

---

## 9. References for the write-up (with why each matters here)

The repo README (commit `3ad551e`) already carries the method references —
Shu et al. 2023, CorrDiff, Ho/Song, flow matching, stochastic interpolants,
Instant-NGP, Górski's HEALPix. The list below is the *related-work* layer
around the geographic-conditioning contribution, found and verified 2026-08-16.

**Novelty claim, stated carefully:** components exist separately — sphere-native
location encoders, HEALPix meshes in weather DL, terrain conditioning in
generative downscaling — but no prior work (a) compares *learned*
multiresolution location tables, planar-hash vs sphere-native, as conditioning
for generative super-resolution, (b) controls for scale allocation between them
(the matched-ladder experiment), or (c) validates the conditioning causally via
a permutation control. Guard with "to the best of our knowledge"; searches were
thorough but not systematic, cutoff Aug 2026.

### The closest relative — engage directly in related work

- **Rußwurm, Klemmer, Rolf, Zbinden, Tuia (ICLR 2024)** — *Geographic Location
  Encoding with Spherical Harmonics and Sinusoidal Representation Networks*.
  arXiv:2310.06743. The nearest thing to "someone tried this": a systematic
  sphere-native vs planar encoder comparison motivated by pole/seam artifacts.
  **Differentiate on three axes:** fixed basis + MLP (not learned multires
  tables); classification/regression tasks (not generative SR conditioning);
  no ladder control or causal test. Their sphere-native gains concentrate at
  the poles — consistent with our ±60° crop neutralizing much of the spherical
  advantage (§2.3, §4.7).

### Learned location encodings (lineage of contribution 1)

- **Müller et al. 2022** (Instant-NGP, ACM TOG) — the hash grid. Already cited.
- **Mai et al. 2023** — *Sphere2Vec* (ISPRS J. Photogramm. Remote Sens.;
  arXiv:2306.17624). First spherical-distance-preserving location encoder;
  proves planar encoders distort spherical distance, gains concentrated in
  polar regions. Directly supports the ±60° argument.
- **Mai et al. 2020** — *Space2Vec* (ICLR). The planar multiscale location
  encoder the two above build on; the conceptual ancestor of using multiscale
  position features at all.
- **Mac Aodha, Cole, Perona 2019** (ICCV) — origin of "location embedding as
  model conditioning" (geo priors for classification).
- **Tancik et al. 2020** — Fourier features (NeurIPS). Justifies the
  `sinusoidal` baseline encoder.

### HEALPix as a DL substrate for weather

- **Karlbauer et al. 2024** — *Advancing Parsimonious Deep Learning Weather
  Prediction Using the HEALPix Mesh* (DLWP-HPX), JAMES,
  doi:10.1029/2023MS004021. Weather forecasting *on* the HEALPix mesh. Key
  contrast to state explicitly: they use HEALPix as the computational grid;
  we use it as a learned feature pyramid read from a lat-lon grid — different
  role, so their success and our mixed §4.7 result do not conflict.
- **Perraudin et al. 2019 / Defferrard et al. 2020** — DeepSphere. Spherical
  graph CNNs with HEALPix sampling; the standard "HEALPix in ML" citation.
- **FourCastNet 3 (2025)** — arXiv:2507.12144. Spherical-geometry forecasting
  at scale; supports the "sphere-aware architectures matter" framing.

### Terrain/static conditioning in downscaling (the CorrDiff axis)

- **Mardani et al.** — CorrDiff. Already cited; the static-field conditioning
  and split-model precedent.
- **Sha et al. 2020** (J. Appl. Meteorol. Climatol.) — elevation-conditioned
  deep downscaling of near-surface temperature in complex terrain.
  **NOT re-verified online — check exact title/venue before it enters the bib.**
- **Stengel et al. 2020** (PNAS) — adversarial climate SR; standard field
  citation.

### Scale allocation / coarse-to-fine (the §4.7 diagnosis)

- **Lin et al. 2021** — *BARF* (ICCV). Coarse-to-fine annealing of positional
  encoding: established precedent that frequency/level *allocation* — not the
  representation itself — drives artifacts in multiresolution encodings.
  Exactly the shape of the ladder finding, and the citation for the
  coarse-to-fine training guard if ever needed.
