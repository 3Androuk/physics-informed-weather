# Paper outline v2 — journal style, ~25 pages (~9,000–11,000 words + figures)

**Working title:** What matters in guided diffusion downscaling of
atmospheric fields: data consistency dominates conditioning and architecture

**Thesis sentence:** In diffusion-based downscaling with a frozen
unconditional prior, one analytic component — exact inference-time data
consistency — is worth more than every conditioning, covariance, and
architecture choice combined; we measure each component with controls,
explain the pattern (per-step corrections are erased; injected information
and energy survive), and demonstrate the resulting zero-shot sampler on real
coarse forecast products against paired truth.

**Scope decisions (locked, v2):**
- Framing: ABLATION/ANATOMY paper. Method-family comparison demoted to a
  1-page corroboration section citing the published intercomparisons (GMD
  2026; arXiv 2512.13987; CORDEX-ML-Bench) as agreement, not competition.
- Motivation: coarse-run ML/hybrid emulators (NeuralGCM-ENS 1.4°, ACE2 1°)
  and dissemination/archive-grade coarse products (1.5° IFS grids). CMIP6
  gets one sentence as the long-range cousin; no CMIP6 experiments.
- Deployment demo = forecast products with PAIRED ERA5 truth (built:
  data/download_forecast.py + eval/downscale_forecast.py + channel mask).
- Physics-vs-observation scoped: effect size not substitution;
  inference-only; hydrostatic residual diagnostic carries the argument.
- Every table reports the sampling noise floor (single_l2_std); t2m
  headline numbers re-run with the spread patch selection before writing.
- Out: fibre/minimum-action/H2CD theory (H2CD parked by the band-energy
  gate — one future-work sentence), backbone pilot, RL, learned
  post-processors (ratio leaks through training data; spread inflation is
  standard DA practice — cite, use, don't claim).

## Sections and page budgets (~22 pp body + refs/figs ≈ 25)

### 1. Introduction — 2 pp
- Fine-resolution fields are needed where fine-run compute or storage is
  unavailable: many users hold only coarse-DISSEMINATED or archived
  products (e.g. 1.5° forecast grids), and coarse-RUN systems exist at
  the same scales (one clause; no claims about them until §9's tests).
  One sentence: the same need at climate scale (CMIP6 → CORDEX).
- The deliverable is a calibrated ensemble of plausible fine states, not a
  forecast — exactly as in dynamical downscaling.
- The field has adopted guided diffusion with inference-time consistency
  (ZSSD; scale-adaptive downscaling, Nature MI 2025) and has compared model
  families (GMD 2026; CORDEX-ML-Bench) — but nobody has measured WHICH
  COMPONENTS of the recipe do the work. We do.
- Contributions: (i) component anatomy with controls — projection worth
  ~2×, statistically-optimal initialization catastrophic, per-step
  refinements erased, with an organizing principle; (ii) conditioning
  ablation with a permutation control, replicated from 1 to 20 variables;
  (iii) zero-shot deployment across ratio AND channel availability
  (masking = channel inpainting) demonstrated on real forecast products
  with paired truth; (iv) a practitioner's do/don't list.

### 2. Related work — 1.5 pp (essential now; the field moved)
- Method papers: CorrDiff (residual), CDSI (interpolants), PC-AFM /
  stochastic flow matching (physics-constrained flows).
- Intercomparisons/benchmarks: GMD 2026 inter-comparison (DDPM most
  physically coherent — corroborates §6), arXiv 2512.13987, CORDEX-ML-Bench.
- Zero-shot: ZSSD (DPS-based, heterogeneous GCMs), Nature MI
  scale-adaptive; DDNM/ILVR/DPS lineage from imaging.
- Ensemble calibration: variance inflation (EnKF tradition), post-hoc
  spread control for diffusion (arXiv 2501.14822).
- Positioning sentence: those papers propose or rank methods; this one
  measures the components they share.

### 3. Data and problem setup — 1.5 pp
- ERA5 0.25° (~28 km), ±60° band, 2007–2015 train / 2016–2017 test;
  6-hourly snapshots, daily stride. t2m single-variable track + 20-channel
  set (5 surface + 5 vars × {500,700,850} hPa).
- Degradation A = block average; train ratios {2,4,8}; eval 4×/8×/16×
  (held out); deployment adds ratio 6 (1.5° products). ker A holds
  1−1/r² of DOF: 94/98/99.6%.
- Evaluation protocol: spread patch selection across the test years (the
  contiguous-window pitfall documented), sampling noise floor
  (single_l2_std) reported with every table, latitude-weighted RMSE
  (WB2 convention) alongside plain RMSE.

### 4. Methods — 3 pp
- Guided unconditional sampler (Shu et al. 2023): noise-mixing init, K
  refinement loops. Present via the split-process exposition:
  x_t → D_θ (denoise) → Π_{A,y} (project x̂₀) → R (re-noise); only the
  middle algebraic step sees A — hence zero-shot across operators. State
  why the NOISY state must NOT satisfy Ax_t = y (it would contradict the
  forward marginal the network was trained on) — the justification for
  projecting x̂₀, not x_t.
- Projection (DDNM) as the single data-consistency mechanism.
- Channel-inpainting mask: A = coarsen ∘ select-channels; unobserved
  channels zero-guidance + excluded from projection → generated.
- Conditional families in brief (flow matching, SI, residual, direct map)
  — corroboration section only.
- Geo-conditioning ladder: none/xyz/sinusoidal/static/hash/hash_compact/
  HEALPix/combo, matched capacity, shuffle-geo permutation control,
  parameter-free nulls (xyz_static, sinusoidal_static).
- Full-field: tiled overlap-blend with one global noise field vs fused
  (MultiDiffusion-style per-step fusion).
- Ensembles: members via init noise and η; post-hoc spread inflation about
  the ensemble mean (deviations lie in ker A after projection, so it
  preserves consistency exactly — one-line remark, cited to inflation
  literature, not claimed).
- Metrics: RMSE (+lat-weighted), spectrum log-L1, radial coherence, qq,
  CRPS, spread–skill, noise floor.

### 5. Results A — anatomy of data consistency — 3.5 pp (THE CORE)
- 5.1 Projection dominates: unprojected 0.787 vs projected 0.404 vs
  bicubic 0.456 (t2m 4×; 0.62 K drift unprojected) — larger than every
  other effect measured. [Re-run with spread selection; 8×/16×
  spot-checks or scope claims to 4×.]
- 5.2 Statistical refinements are erased: covariance-weighted projection
  (data-assimilation-style gain) is mechanistically real in isolation yet
  an end-to-end null (0.4025 vs 0.4044, inside the noise floor). One
  compact subsection — the negative result that motivates the principle.
- 5.3 Initialization must carry energy: covariance-optimal (smooth) init
  degrades the spectrum ~28×; the denoiser reads missing high-frequency
  energy as "nothing to reconstruct."
- 5.4 The organizing principle: each step contracts to the manifold and
  erases off-manifold detail; what survives is injected INFORMATION
  (which constraints) and ENERGY (at expected wavenumbers) — explains
  5.1–5.3 at once. Diagnostics figure (coherence + qq).
- 5.5 Stochasticity and calibration: η sweep for the ~20%
  underdispersion [pending]; spread-inflation as the analytic fix (cited,
  consistency-preserving in ker A); no learned post-processors (ratio
  leaks through training data).
- 5.6 Physics vs observation, stated carefully: NOT substitutes (the
  ~linear hydrostatic relation decomposes across the same range/null
  split; projection pins block means only). Effect-size comparison at
  inference. Decisive measurement = hydrostatic residual of (truth,
  bicubic, unconstrained samples, projected samples): if the model's
  residual already sits at the reanalysis's own level, there is no
  headroom for ANY constraint on that relation [pending run — arms
  built]. Scope stated: inference-time only; training-time physics
  untested; the erasure principle predicts per-step corrections are
  partly erased regardless, so the residual diagnostic, not the metric
  delta, carries the argument.

### 6. Results B — method families across ratios — 1 p (corroboration)
- One table + one figure: flow/SI/residual/direct/guided ± projection at
  4×/8×/16×. In-distribution: residual+geo ensemble best L2/CRPS
  (0.2706/0.4523), SI best spectrum 2–8×. Held-out 16×: every conditional
  family below bicubic; guided+projection the only survivor
  (1.3443/0.0245).
- Deployment rule (fixed known ratio → conditional; heterogeneous/unknown
  → guided unconditional), noted as consistent with the published
  intercomparisons; ratio-6 deployment (§9) is interpolation for
  conditional models, plain zero-shot for guided.

### 7. Results C — conditioning ablation — 2.5 pp
- t2m ladder table [re-run]: bicubic/no-geo/xyz/sinusoidal/static/hash/
  hash_compact/combo (+matched HEALPix), with shuffle-geo control and
  noise floor columns.
- Findings: geography worth ~13% of L2; top of the ladder
  encoder-invariant (hash ≈ static ≈ matched HEALPix) → grid geometry is
  not the bottleneck at 28 km; combo's margin sits AT the noise floor —
  stated as unresolved unless the re-run separates it; raw coordinates
  are a null; fixed Fourier features hurt the spectrum; untouched-cells
  mechanism paragraph; level gating rejected (two sentences, supplement).
- 20-variable replication [runs done, tables in]: hash / hash_compact
  (−43% encoder params) / 2D-HEALPix / static / no-geo + parameter-free
  nulls, per-variable metrics; does the multivariate setting change the
  encoder verdict? [shuffle-geo control at 20 var pending.]

### 8. Results D — full-field reconstruction — 1 p
- Tiled (overlap-blend, shared global noise so overlapping tiles agree)
  vs fused (per-step fusion): seams, coarse consistency of the stitched
  field, cost. [Numbers from the fused/tiled runs — slot in.]

### 9. Results E — deployment on real forecast products — 2 pp (the applied anchor)
- IFS HRES 1.5° (2016–2017 inits = the TEST years; 19/20 channels, tcwv
  masked): per-lead lat-weighted RMSE / CRPS / spectrum for downscaled vs
  bicubic vs the coarsened-truth CONTROL (same seeds) — the control gap
  isolates forecast error from downscaling error; where the curves
  converge, "downscale the coarse product" costs nothing at that lead.
- NeuralGCM-ENS 1.4° (true coarse-RUN system; 15 observed + 5 generated
  channels): the compute-savings framing plus the channel-inpainting
  readout — can the 20-var model generate t2m/msl/winds from upper-air
  fields alone, scored against truth? Either outcome is a finding.
- Zero-shot statement: ratio 6 and the channel mask both enter only
  through A at inference; the checkpoints are untouched.
- One paragraph: input-as-truth vs input-with-error — hard projection
  anchors to forecast error at long leads; DDNM+-style damped consistency
  (σ_obs, estimable per lead HERE because paired truth exists) as the
  principled extension [flag or future work].
- [All runs pending; downloader + driver + mask are built and tested.]

### 10. Discussion and limitations — 1.5 pp
- The practitioner's list: (1) enforce exact consistency every step — it
  outweighs architecture/conditioning; (2) never smooth the guidance/init
  — energy, not optimality; (3) don't buy statistically-refined per-step
  corrections — they get erased; (4) location conditioning ≈ physiography
  — static fields capture most of it; (5) calibrate post hoc (η /
  inflation), not with learned ratio-locked post-processors.
- Zero-shot as the deployment property: products form a resolution
  continuum and effective resolution ≠ grid resolution.
- Physics answered honestly: the conservative remap IS physics, enforced
  exactly; the testable cross-channel relation shows [diagnostic outcome];
  training-time physics untested.
- Limitations: one band/region; stationarity; snapshots (no temporal
  consistency); coarsened-truth training vs real-input deployment (partly
  probed by the §9 control); tails beyond p99 unassessed.
- Future work, one sentence each: σ_obs soft consistency; temporal
  downscaling; cascade diffusion (parked by the band-energy gate);
  spherical backbones.

### 11. Conclusion — 0.5 pp

## Supplement
Level-gating details; covariance estimation numerics (detrend/Hann/
Gaspari–Cohn) + isolation tests; corrector calibration; training
stability (width-128 lr collapse to the Var(ε) fixed point; divergence
guard; best-by-val checkpoint); DDP; capacity matching per encoder;
band-energy gate (H2CD); per-variable 20-var tables; hyperparameters.

## Figures (~10)
1. Schema: split-process sampler + where A enters (the only figure with
   boxes). 2. Projection anatomy: maps + spectra, with/without + cov-init.
3. Diagnostics: coherence + qq. 4. Ratio-collapse curve (all families).
5. Encoder ladder with noise-floor bars + shuffle control. 6. 20-var
per-variable encoder summary. 7. Tiled-vs-fused seams. 8. Deployment:
RMSE/CRPS vs lead (fcst / control / bicubic). 9. Channel-inpainting panel
(generated t2m vs truth). 10. Qualitative full-field reconstruction.

## Pending runs → slots (nothing reopens the structure)
- t2m ladder + anatomy re-run, spread selection + `--ensemble` noise floor
  → §5.1–5.3, §7 tables.
- 8×/16× projection spot-check → §5.1 (else scope to 4×).
- η sweep → §5.5; corrector gate → supplement.
- Hydrostatic residual diagnostic + constraint arms (20 var) → §5.6.
- Shuffle-geo at 20 var → §7.
- Fused/tiled numbers → §8.
- Forecast demo: HRES t2m; HRES 20-var (tcwv masked); NeuralGCM-ENS
  (15+5) → §9.
