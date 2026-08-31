# Paper outline — journal style, ~25 pages (~9,000–11,000 words + figures)

**Working title:** Data consistency, not conditioning, drives diffusion-based
climate downscaling: a method anatomy from single-variable to 20-variable ERA5

**Thesis sentence:** For CMIP6-style inputs (a 50–250 km resolution
continuum), a single unconditional diffusion model with inference-time data
consistency is the deployable architecture — the only family that survives
off-ladder ratios — and its performance lives in the consistency mechanism,
not in conditioning, covariance sophistication, or grid geometry.

**Scope decisions (locked):**
- Thesis = one journal-style paper (~25 pages, JAMES/GMD-class length).
- Theory proposals (fibre training, minimum-action, fibre-bridge) are OUT —
  at most one future-work sentence on training-time vs inference-time
  consistency.
- Results A (method comparison) stays but lean: the 16× below-bicubic cliff
  is the *evidence* for the deployment rule; in-distribution conditional wins
  are the rule's other half. One figure + one table.
- HPX backbone pilot: future-work sentence only (follow-up paper).
- Fused and tiled full-field experiments: own results section (§7).
- 20var experiments (done + in progress): own results section (§8), written
  to stand with whichever runs have finished at submission time.

## Section plan with page budgets (~22 pages body + refs/figures ≈ 25)

### 1. Introduction — 2 pp
- Impact assessment needs ~25 km; CMIP6 delivers 50–250 km; dynamical
  downscaling costs CPU-months per scenario.
- The archive is a resolution *continuum* (EC-Earth3 ~80 km, MPI-ESM1-2-HR
  ~100 km, CNRM-CM6-1 ~140 km, IPSL-CM6A-LR ~250 km; effective resolution
  4–8× coarser than grid) → deployment is always off-ladder → zero-shot ratio
  transfer is the deployment condition, not a robustness extra.
- Perfect prognosis by necessity (free-running GCMs are not time-synchronized
  with reality; MOS pairs impossible) → train coarsened-truth→truth, compose
  with bias correction at deployment.
- Contributions: (i) four-family comparison across a ratio ladder incl.
  held-out 16×; (ii) data-consistency anatomy (projection worth ~2×;
  covariance-aware refinements measured null, with mechanisms); (iii)
  conditioning ablation with controls; (iv) full-field reconstruction
  (tiled vs fused); (v) 20-variable scaling.

### 2. Data and problem setup — 2 pp
- ERA5 0.25° (~28 km) via WeatherBench2; t2m single-variable track + the
  20-channel set (5 surface + 5 vars × {500,700,850} hPa).
- **Table: CMIP6 variable mapping** (t2m→tas, u10/v10→uas/vas, msl→psl,
  tcwv→prw, z→zg, t→ta, u→ua, v→va, q→hus; day/plev8 covers all three
  levels; zg unit ×g caveat; daily-mean vs snapshot caveat).
- Degradation A = block average; ratios 4×/8×/16× ↔ 111/223/445 km ↔ typical
  CMIP6 / coarse end / stress test. Underdetermination: 1−1/r² of DOF free
  (94/98/99.6% — the ker A dimension).
- 128-px patches (~3,600 km), normalization, train/test split.

### 3. Methods — 3.5 pp
- Guided unconditional diffusion (Shu et al. 2023): noise-mixing init, K
  refinement loops; + DDNM projection x + A†(y − Ax) per step (Wang et al.
  2023). Derive A, A† (nearest-upsample), P = I − A†A properly here.
- Conditional families: flow matching, stochastic interpolant, residual
  (CorrDiff-style mean + generative residual), direct map; bicubic baseline.
- Geo conditioning: encoder ladder (none/xyz/sinusoidal/static/hash/HEALPix/
  combo), matched capacity, shuffle-geo control.
- Full-field reconstruction: tiled sampling with overlap blending and a
  shared global noise field (overlapping tiles crop init noise from one
  field so they agree where they overlap) vs the fused variant; tile size,
  seam handling.
- Ensembles: members via η and init noise; CRPS, spread–skill,
  reliable-spread target.
- 20-variable configuration: 244.4M-param UNet (width 128), per-channel
  normalization, fp32, DDP multi-GPU.

### 4. Results A — method comparison — 2 pp (deliberately lean)
- **Table 1:** L2 / spectrum / CRPS, all methods × 4×/8×/16×.
- **Figure 1:** L2 vs ratio; every conditional line crosses bicubic at 16×,
  guided+projection does not (1.3443 / 0.0245).
- In-distribution reading: residual+geo ensemble best L2/CRPS
  (0.2706 / 0.4523); SI best spectrum at 2–8×.
- **Deployment rule:** fixed known ratio → conditional; heterogeneous or
  unknown ratio → guided unconditional. State it, then move on.

### 5. Results B — anatomy of data consistency — 3.5 pp (the core)
- Projection is the dominant effect: 0.787 → 0.404 L2 (bicubic 0.456) —
  ~2×, larger than any conditioning or architecture effect measured.
  Figure: with/without projection maps + spectra.
- Covariance-aware projection (weather-DDNM) is null: 0.4025 vs 0.4044,
  identical spectra — despite the mechanism being real in isolation (<1% of
  nearest-upsampling's spurious block-scale power). Mechanism: the denoiser
  erases per-step correction detail; what matters between steps is
  information injection.
- Covariance-optimal initialization is catastrophic (spectrum 28× worse):
  guidance must carry high-frequency ENERGY; statistical optimality is the
  wrong objective for an init.
- Diagnostics figure (promoted from supplement): spectral coherence + qq —
  separates reconstructed fine scales from hallucinated-with-right-power.
- Ensemble calibration: ~20% underdispersion at η=0; η sweep.
- Pending rows slot here when runs land: corrector calibration curve;
  projection vs DPS vs both.

### 6. Results C — conditioning ablation — 2.5 pp
- **Table:** encoder ladder at 4× (combo 0.3410 / static 0.3458 /
  hash 0.3530 / …) + shuffle-geo control + ensemble metrics.
- Findings: stationary geography ≈ 13% of recoverable L2; learned-vs-static
  gap small; hash ≈ HEALPix at matched capacity → grid geometry is not the
  bottleneck at 28 km; combo (hash+static) wins both ratios and CRPS/spread;
  level gating rejected in two sentences (loses spectrum to static; fine
  levels legitimate early in the chain) — details in supplement.
- Untouched-cells mechanism paragraph (sphere shell touches ~5% of cube
  cells → hash ~90% collision-free; bigger table useless).

### 7. Results D — full-field reconstruction (tiled vs fused) — 1.5 pp
- Full-domain figure: stitched field with invisible tile boundaries +
  seam-difference panel; qualitative synoptic structure.
- Small metrics table: tiled vs fused (L2, spectrum, coarse-consistency of
  the stitched field, cost). [numbers from the shark-l runs]
- Shared-noise-field trick as the enabling detail.

### 8. Results E — scaling to 20 variables — 2.5 pp
- Done: full multivariable pipeline (memmapped patching, per-channel
  stats), 244M model, DDP.
- Training-stability finding (main-text paragraph): at width 128, lr 2e-4
  collapses to the trivial loss=1 fixed point (Var(ε); runaway steps →
  GroupNorm-mediated signal death → predict-zero); stable at 1e-4, fp32,
  batch 20.
- Running: baseline vs geo arms; report per-channel L2/CRPS; does the
  multivariate setting change the encoder verdict (hash vs static at 20
  channels)?
- Written to stand regardless of run completion: framed as "does the
  single-variable anatomy transfer to the multivariate regime."

### 9. Discussion and limitations — 2 pp
- Deployment rule vs the CMIP6 continuum; two-stage pipeline
  (bias-correct then downscale); --project as the truth-vs-suggestion dial;
  soft consistency as principled middle ground (future work).
- Limitations: coarsened-truth ≠ real GCM distribution; ERA5 effective
  resolution; exact linear A assumption; what breaks at km-scale (learned
  vs static gap predicted to widen; no exact A across model pairs).
- Future work, one sentence each: HEALPix backbone pilot; training-time
  (fibre) consistency and its zero-shot trade-off; temporal downscaling.

### 10. Conclusion — 0.5 pp

## Supplement
- Hyperparameters and schedules; DDP implementation; capacity matching per
  encoder; level-gating details; diagnostics battery definitions
  (errmap/memgap/coherence/qq); weather-DDNM estimation numerics (planar
  detrend, Hann taper, Gaspari-Cohn localization); corrector and DPS sweep
  results; per-channel 20var tables; lr-collapse training curves.

## Figure budget (~9 main)
1. Ratio-collapse curve (L2 vs ratio, all methods).
2. Projection anatomy (maps + spectra, with/without).
3. Diagnostics (coherence/qq).
4. Full-field seams (tiled vs fused).
5. Encoder ladder.
6. Shuffle-geo control.
7. 20var per-channel summary.
8. Radial spectra panel.
9. Qualitative reconstruction panel.

Tables: comparison (Table 1), CMIP6 mapping, encoder ladder, tiled-vs-fused,
per-channel 20var, config/hyperparameters (supplement).

## Pending runs → slots (no new sections)
- 20var baseline + geo (width 128, batch 20, lr 1e-4, fp32, DDP) → §8.
- Corrector calibration gate → §5 (+ supplement curve).
- Null-space training run → one number in §9's training-vs-inference
  paragraph.
- DPS sweep → §5 row (+ supplement).
- t2m ladder remainder (static ensemble, η sweep, 16× combo, seed
  replicates, memgap read) → §5/§6 tables.
