# Section 5 — Anatomy of data consistency (draft)

> Draft for the journal-style thesis paper, §5 (budget ~3.5 pp).
> Numbers in [square brackets] are pending shark-l runs; everything else is
> measured. Figure/table callouts are placeholders to be renumbered.

## 5. Anatomy of data consistency

The comparison in Section 4 establishes *which* method family survives
deployment conditions; this section establishes *why*. We dissect the guided
unconditional sampler into its components — the per-step data-consistency
projection, the statistical sophistication of that projection, the
initialization, and the sampler's stochasticity — and measure the
contribution of each in isolation. The result is a strongly non-uniform
attribution: a single exact linear constraint accounts for more performance
than every other design choice in this study combined, while two natural
statistical refinements of that constraint contribute nothing or worse. We
close with the interpretation these results force, and with a conjecture
about what "physics-informed" should mean at mesoscale resolution.

### 5.1 Hard projection dominates every other design choice

The guided sampler of Section 3 consults the coarse observation in two
places: once at initialization, where the noise-mixed low-frequency guidance
starts each refinement loop, and — optionally — at every denoising step,
where the range–null projection

    x̄₀ = x̂₀ + A†(y − A x̂₀)

pins the block averages of the Tweedie estimate x̂₀ to the observation while
leaving the generated null-space content untouched. Because A is a block
average and A† its nearest-neighbour upsampling pseudo-inverse, the
projection is exact and costs no additional network evaluations.

Disabling it is catastrophic. At 4× on T2M (256 test patches), the
unprojected sampler reaches an L2 of 0.787 — *worse than bicubic
interpolation* (0.456) — and drifts from its own input by 0.62 K in
coarse-consistency error: the chain, consulted only at initialization,
gradually forgets the observation it was asked to condition on. With the
per-step projection enabled, the same model, checkpoint, and noise
realizations reach 0.404 [Table 3; Fig. 2 shows the paired maps and radial
spectra]. The factor of ~1.9 between these two numbers is the largest single
effect measured anywhere in this study: larger than the choice of
conditioning encoder (Section 6, ≈13% of L2), larger than any architecture
variation, and larger than the difference between method families
in-distribution (Section 4). Whatever else a downscaling diffusion model
does, its relationship to the observation is the first-order term.

We emphasize what the projection is *not*: it is not a soft penalty, a
learned correction, or a guidance gradient with a tunable weight. It is the
analytic decomposition of the state into a component the observation
determines (the resolved, conservative part, A†y) and a component it cannot
(the subgrid part, (I − A†A)x̂₀), with the first replaced outright. It has
no hyperparameters and adds no cost.

### 5.2 A statistically better projection changes nothing

The nearest-neighbour pseudo-inverse spreads each coarse-cell residual
uniformly over its block — a piecewise-constant correction that injects
spurious power at and above the block wavenumber. Data-assimilation practice
suggests the remedy: replace A† with the covariance-weighted gain
K_C = C Aᵀ(A C Aᵀ)⁻¹, where C is a stationary spatial covariance estimated
from training fields (implemented spectrally on periodic patches, with
planar detrending, Hann tapering, and Gaspari–Cohn localization to control
estimation artifacts; Supplement S4). The corrected update distributes each
residual smoothly according to the climatological correlation structure. In
isolation the mechanism behaves exactly as designed: on synthetic residuals
the covariance correction carries less than 1% of the spurious
above-block-scale power of the piecewise-constant correction (retained as a
regression test).

End to end, it is a null result. Over 256 patches at 4×, the
covariance-weighted sampler scores 0.4025 against ordinary projection's
0.4044 — inside the ±0.002–0.005 seed-to-seed noise floor — with radial
spectra indistinguishable by eye and by our log-L1 spectral metric
[Table 3]. Ensemble metrics tell the same story. The refinement that
data-assimilation intuition predicts should matter, does not.

### 5.3 Initialization must carry energy, not optimality

The same covariance machinery offers a second, more aggressive use: replace
the nearest-upsampled guidance that initializes the first refinement loop
with the covariance lift K_C y — the minimum-variance-optimal field
consistent with the observation, and therefore the "statistically best"
starting point available without a network. This variant is not merely null
but catastrophic: the spectral metric degrades by a factor of ~28 (0.29
against 0.01), with generated fields visibly starved of fine-scale
structure [Fig. 2, right panel].

The mechanism is instructive. The sampler's noise-mixing initialization
x_t = √ᾱ_t x_g + √(1−ᾱ_t) ε places the chain at a state whose statistics
match what the denoiser saw in training at noise level t — including the
high-wavenumber energy contributed by ε and by the blocky upsampled
guidance. The covariance lift, being optimally smooth, removes precisely
that energy; the denoiser, trained to expect it, interprets its absence as
"nothing to reconstruct here" and the chain never recovers the fine scales.
Guidance must carry high-frequency *energy* for the reverse process to
sculpt; statistical optimality of the initialization is the wrong objective.

### 5.4 Interpretation: what survives the denoiser

Together, 5.1–5.3 support a compact organizing principle. Each denoising
step contracts the state toward the learned data manifold, erasing
off-manifold detail — including the fine structure of whatever correction
was applied in between steps. What survives a step is therefore not the
correction's *shape* but its *information content*: which coarse constraints
it enforced, and how much energy it left at the wavenumbers the denoiser
expects. This explains all three measurements at once. Hard projection
matters enormously because it injects information (the observed block
means) that the chain otherwise forgets. The covariance-weighted projection
changes only the correction's shape — content the next denoising step
overwrites — and so measures null. And the covariance-optimal
initialization removes energy the denoiser requires, which no later step
restores. The live levers of the sampler are the information injected and
the schedule on which it is injected; per-step statistical refinements
between those injections are erased.

This principle motivates a conjecture we state explicitly, since the
project's framing is physics-informed generation. In generative
downscaling at mesoscale resolution, the most valuable physics is not an
approximate dynamical residual imposed as a soft penalty — the w-weighted
PDE guidance of Shu et al. (2023), which we run at w = 0 throughout — but
the exact linear conservation law already embedded in the degradation
operator: block means are the conservative remap, and enforcing them
exactly is worth a factor of two. Soft approximate physics competes for
the same budget and, on the evidence of 5.2, its per-step corrections are
substantially erased. The direct test — approximate-physics guidance
(specific-humidity positivity; the hydrostatic thickness relation between
z500 − z850 and layer-mean temperature) inserted in the same plug-in slot,
compared against projection-only and both — requires the multivariate
checkpoints of Section 8 and is reported there [pending: §8 conjecture
test].

### 5.5 Stochasticity and calibration

The remaining component is the sampler's noise. At the deterministic
setting (DDIM η = 0), ensemble members differ only through their
noise-mixing initializations, and the resulting ensembles are underdispersed
by roughly 20% against the reliable-spread target
(ens-mean L2 / √(1 + 1/M)) [Table 4]. Raising η diversifies members at no
retraining cost [pending: η sweep row]. Two further calibration
instruments are evaluated in the Supplement: a null-space Langevin
corrector that runs fibre-restricted Langevin steps over the frozen prior
after reconstruction (its calibration-versus-steps curve, isotropic and
spectrally preconditioned [pending: corrector gate run]), and DPS-style
likelihood guidance as a soft alternative to hard projection
[pending: DPS sweep]. Neither changes the section's conclusion; they
bound how much calibration can be bought at inference time once the
information-injection structure is fixed.

### Placeholders to resolve before submission
- [ ] Table 3: consolidate 5.1–5.2 numbers (unprojected / bicubic /
      projected / covariance-projected; L2, spec-logL1, coarse-consistency).
- [ ] Fig. 2: paired maps + spectra (with/without projection; cov-init).
- [ ] Table 4 + η row: ensemble metrics at η = 0 vs swept η.
- [ ] Corrector calibration curve (supplement pointer + one sentence here).
- [ ] DPS sweep verdict (one sentence here; curve in supplement).
- [ ] Seed replicates to firm the ±0.002–0.005 noise floor claim.
- [ ] 8×/16× spot-checks of 5.1 (projection gain at higher ratios) if
      available; otherwise scope 5.1 claims explicitly to 4×.
