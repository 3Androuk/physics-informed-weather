# Spectrally preconditioned null-space posterior correction

Working document for the corrector line of work. This branch exists to answer
one falsifiable question before any larger investment:

> **Does a spectrally preconditioned null-space Langevin corrector reach the
> same ensemble calibration as an isotropic one in substantially fewer steps —
> and does either fix the measured ~20 % underdispersion of our ensembles?**

## Thesis question

Can a **frozen unconditional weather diffusion model** be converted into a
**data-consistent posterior sampler** whose residual corrections provably
improve posterior accuracy, **without conditional retraining**?

Every high-resolution state consistent with a coarse observation satisfies
`Ax = y`; the useful output is a calibrated distribution `p(x | Ax = y)`, not
one realistic field. Existing options trade off against each other: bicubic is
consistent but has no variability; CorrDiff learns the posterior but needs
paired conditional training per setup; DDNM reuses an unconditional prior and
guarantees consistency but not posterior samples; particle/SMC methods approach
the posterior at high cost; exact-posterior-score methods need fine-tuning.

## Construction (condensed)

Parameterize every consistent state as `x = A†y + Nz` with `N` a basis of
`ker A` (`N Nᵀ = P = I − A†A`; for block-averaging, `Px = x − blockmeans(x)` —
already implemented as `models.transport.nullspace_project`). The constrained
posterior in residual coordinates is `π_y(z) ∝ p_θ(A†y + Nz)`. The
preconditioned Langevin SDE

    dZ = C Nᵀ ∇log p_θ(A†y + NZ) dτ + √(2C) dB

has `π_y` invariant (Fokker–Planck), dissipates KL monotonically, and keeps
`A X_τ = y` **exactly at every τ, for any score network** (AP = 0). A
Metropolis adjustment would make every discrete step posterior-safe under
every f-divergence (data-processing inequality), but requires model-likelihood
evaluations — see Deferred.

**Why preconditioning is the contribution candidate.** Isotropic Langevin
mixes at rates `1/λ_i` per posterior mode; the step size is capped by the
fastest mode while total mixing waits for the slowest, so steps scale with the
condition number `κ(Σ_y)` — enormous for weather spectra. Choosing
`C = Σ_y` whitens the Gaussian-limit dynamics into an OU process where **every
scale and every variable mixes at rate 1**, giving
`KL(q_τ | π) ≤ e^(−2τ) KL(q_0 | π)` independent of grid resolution,
wavenumber count, spectral dynamic range, and inter-variable scaling. With an
imperfect projected score (error ε_s) and a preconditioned log-Sobolev
constant ρ_C:

    KL(q_τ | π) ≤ e^(−ρ_C τ) KL(q_0 | π) + ε_s²/(2ρ_C) · (1 − e^(−ρ_C τ))

— initialization error decays, score error sets a floor, preconditioning
raises ρ_C, and consistency stays exact throughout. Target discrete theorem:
posterior error ≤ initialization + score + time-discretization +
covariance-estimation errors.

**Weather implementation.** `C_y = P F⁻¹ S_y(k) F P` with `S_y(k)` a
(cross-)spectral residual covariance — the composition of two functions
already in this repo (`nullspace_project`, `weather_ddnm.apply_covariance`),
with `S_y` estimated by a multivariate extension of
`data.estimate_spectral_covariance`. The same `S_y` is the residual source
`L_y` of the null-space transport derivation: one estimation artifact, three
uses.

## Honest caveats (state these in any writeup)

1. **Gaussian-limit circularity.** The resolution-independent rate is proved
   for a Gaussian posterior with `C = Σ_y` exact — a regime where one would
   sample in closed form and need no Langevin. The method matters when the
   posterior is non-Gaussian and Σ is approximate, where the clean theorem is
   only a local-curvature heuristic. Standard status for preconditioning
   results, but it must be framed as such; the synthetic experiment VALIDATES
   THE RATE, it does not demonstrate the use case.
2. **Score quality at small noise.** The corrector needs the score near the
   clean-data level — where learned scores are least reliable. Practical form:
   run at a small fixed t_ε with s = −ε_θ/σ, targeting the σ-smoothed
   posterior; the ε_s² floor in the bound is then the empirical unknown that
   decides everything. Measure it, don't assume it.
3. **Novelty is the combination, not the ingredients.** Preconditioned
   Langevin, entropy dissipation, and MALA are classical; predictor–corrector
   samplers, PnP-ULA, and SMC-based diffusion posterior samplers are adjacent.
   The candidate niche: operator-aware ker-A restriction + multivariate
   spectral preconditioning + weather calibration + finite-compute error
   decomposition. A full literature review is REQUIRED before any novelty
   claim (several closest-looking citations are 2026 preprints, unverified).

## Why our own measurements motivate this (and don't kill it)

- **The gap it targets is measured:** ensembles are ~20 % underdispersive at
  η = 0 (spread 0.193 vs reliable ≈ 0.246 at 4×). The corrector adds
  stochastic exploration of exactly the subspace the observation leaves free,
  with a stationary-distribution justification that η-tuning lacks.
- **The weather-DDNM null does not apply:** there the spectral covariance
  shaped a per-step *correction* — a quality lever the denoiser erases. Here
  it preconditions *mixing* — a compute lever acting after the chain, which
  the denoiser cannot erase. Same artifact, different role.
- **The cov-init failure does not apply:** the theory says initialization
  only sets q_0; the corrector's guarantees are about the dynamics.

## Evidence-gated plan

1. **The decisive cheap experiment** (~1 week, inference-only, existing
   checkpoints): implement the corrector with both `C = I` (isotropic) and
   `C = P F⁻¹ S F P` (spectral); run against the η sweep on the ensemble
   protocol (4-member, 64 patches, projected). Plot CRPS / PIT / spread–skill
   **versus corrector steps**. Headline claim tested directly: spectral
   reaches isotropic's calibration in far fewer steps. Isotropic-vs-spectral
   is the ablation proving the preconditioning does the work.
2. **Synthetic Gaussian notebook** (half a day): exact posterior known;
   verify the e^(−2τ) whitened rate and the κ(Σ) slowdown of the isotropic
   corrector. The theorem's demonstration figure.
3. **Green-light the full paper only if (1) shows the compute gap.** If
   isotropic already calibrates in a handful of steps at this problem size,
   record the negative result (two sentences) and stop.

Full-paper burden (post-gate, realistically post-MSc): baselines DPS,
filtering posterior sampling, CorrDiff, DDNM+, scale-consistent posterior
dynamics; metrics CRPS, energy score, rank histograms/PIT, spread–error,
power and cross-spectra, extremes, coarse consistency, runtime/NFEs.

## Deferred

- Metropolis adjustment (needs probability-flow-ODE likelihoods per proposal:
  hundreds of NFEs per MCMC step — not feasible at thesis compute; unadjusted
  Langevin accepts O(h) bias and loses the exact per-step safety theorem).
- Noisy observations `y = Ax + ν` (soft consistency: ball, not subspace).
- Extension beyond weather (needed for a general-ML venue).

## Scope verdict (agreed)

Strong MSc final chapter with steps 1 + 2 alone. Weather-ML paper if step 1
delivers. The seven-baseline program is PhD-chapter scope and must not
cannibalize the two-act thesis paper (method comparison + encoder anatomy).
