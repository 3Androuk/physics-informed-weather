# Section 5 — Anatomy of data consistency (draft, v2)

> Journal-style draft: few, large subsections. Numbers in [brackets] are
> pending runs; everything else is measured. Figure/table callouts to be
> renumbered. DPS removed (tested, no improvement — not part of the paper).

## 5. Anatomy of data consistency

The corroboration in Section 6 establishes which method family survives
deployment conditions; this section establishes why. We dissect the guided
unconditional sampler into its constituent choices — the per-step
data-consistency projection, the statistical sophistication of that
projection, the initialization, and the sampler's stochasticity — and
measure each in isolation on identical test patches with shared noise
realizations. The attribution that emerges is strongly non-uniform: a
single exact linear constraint accounts for more performance than every
other design choice in this study combined, while two natural statistical
refinements of the same machinery contribute nothing or actively harm. We
close the section with the interpretation these results force, and with a
carefully scoped answer to where the physics in "physics-informed"
downscaling actually resides.

### 5.1 Hard projection dominates every other design choice

The guided sampler of Section 4 consults the coarse observation in two
places: once at initialization, where the noise-mixed low-frequency
guidance starts each refinement loop, and — optionally — at every
denoising step, where the range–null projection

    x̄₀ = x̂₀ + A†(y − A x̂₀)

pins the block averages of the Tweedie estimate x̂₀ to the observation
while leaving the generated null-space content untouched. Because A is a
block average and A† its nearest-neighbour upsampling pseudo-inverse, the
projection is exact, has no tunable parameters, and costs no additional
network evaluations.

Disabling it is catastrophic. At 4× on T2M, the unprojected sampler
reaches an L2 of 0.787 — worse than bicubic interpolation (0.456) — and
drifts from its own input by 0.62 K in coarse-consistency error: consulted
only at initialization, the chain gradually forgets the observation it was
asked to condition on. With the per-step projection enabled, the same
model, checkpoint, and noise realizations reach 0.404 [Table 3; Fig. 2].
The factor of ~1.9 between these numbers is the largest single effect
measured anywhere in this study: larger than the choice of conditioning
encoder (Section 7, ≈13% of L2), larger than any architecture variation,
and larger than the differences between method families in distribution
(Section 6). Whatever else a downscaling diffusion model does, its
relationship to the observation is the first-order term.
[Pending: spread-selection re-run; 8×/16× spot-checks, else scope to 4×.]

We emphasize what the projection is not: it is not a soft penalty, a
learned correction, or a guidance gradient with a tunable weight. It is
the analytic decomposition of the state into the component the observation
determines (the resolved, conservative part, A†y) and the component it
cannot (the subgrid part, (I − A†A)x̂₀), with the first replaced outright.

### 5.2 What survives a denoising step: two instructive failures

Two refinements of the same machinery, both statistically better-motivated
than what they replace, bracket the projection result from either side and
together force the section's organizing principle.

**A statistically better projection changes nothing.** The
nearest-neighbour pseudo-inverse spreads each coarse-cell residual
uniformly over its block — a piecewise-constant correction that injects
spurious power at and above the block wavenumber. Data-assimilation
practice suggests the remedy: replace A† with the covariance-weighted gain
K_C = C Aᵀ(A C Aᵀ)⁻¹, with C a stationary spatial covariance estimated
from training fields (spectrally, on periodic patches, with detrending,
tapering, and Gaspari–Cohn localization; Supplement S4). In isolation the
mechanism behaves exactly as designed: on synthetic residuals the
covariance correction carries under 1% of the piecewise-constant
correction's spurious above-block-scale power (retained as a regression
test). End to end it is a null: 0.4025 against ordinary projection's
0.4044 — inside the ±0.002–0.005 sampling noise floor — with radial
spectra indistinguishable by our spectral metric [Table 3].

**A statistically better initialization is catastrophic.** The same
covariance machinery offers a more aggressive use: replace the
nearest-upsampled guidance that initializes the first refinement loop with
the covariance lift K_C y — the minimum-variance field consistent with the
observation, and therefore the "statistically best" starting point
available without a network. The spectral metric degrades by a factor of
~28 (0.29 against 0.01), with generated fields visibly starved of
fine-scale structure [Fig. 2, right]. The mechanism is instructive: the
noise-mixing initialization x_t = √ᾱ_t x_g + √(1−ᾱ_t) ε places the chain
at a state whose statistics match what the denoiser saw in training at
noise level t — including high-wavenumber energy from ε and from the
blocky upsampled guidance. The covariance lift, being optimally smooth,
removes precisely that energy; the denoiser, trained to expect it,
interprets its absence as "nothing to reconstruct here," and no later step
recovers the missing scales.

**The organizing principle.** Each denoising step contracts the state
toward the learned data manifold, erasing off-manifold detail — including
the fine structure of whatever correction was applied between steps. What
survives a step is therefore not a correction's shape but its information
content: which coarse constraints it enforced, and how much energy it left
at the wavenumbers the denoiser expects. This explains all three
measurements at once. Hard projection matters enormously because it
injects information — the observed block means — that the chain otherwise
forgets. The covariance-weighted projection changes only the correction's
shape, which the next denoising step overwrites, and so measures null. And
the covariance-optimal initialization removes energy the denoiser
requires, which nothing downstream restores. The live levers of the
sampler are the information injected and the schedule on which it is
injected; per-step statistical refinements between injections are erased.
The spectral-coherence and quantile diagnostics [Fig. 3] confirm that the
projected sampler's fine scales are phase-locked to the reference up to
the wavenumber the observation supports, not merely energetically correct.

### 5.3 Stochasticity and calibration

The remaining component is the sampler's noise. At the deterministic
setting (DDIM η = 0), ensemble members differ only through their
noise-mixing initializations, and the resulting ensembles are
underdispersed by roughly 20% against the reliable-spread target
(ens-mean L2 / √(1 + 1/M)) [Table 4]. Raising η diversifies members at no
retraining cost [pending: η sweep row]. Where residual underdispersion
remains, the analytic fix is post-hoc inflation of member deviations about
the ensemble mean — standard practice in ensemble data assimilation — and
we note a convenient structural fact: after per-member projection all
members share identical block means, so their deviations lie exactly in
ker A and any rescaling of them preserves coarse consistency exactly.
Calibration and consistency do not trade off. We deliberately do not learn
a post-processing network for this: any stage trained on sampler outputs
absorbs the training ratio through the statistics of its inputs and
forfeits the zero-shot property that motivates the sampler (Section 9).

### 5.4 Physics versus observation

Since this line of work descends from physics-informed reconstruction —
the sampler of Section 4 is the w = 0 limit of Shu et al.'s
physics-guided algorithm — we state explicitly where physics enters, and
carefully, because the two available constraints are not substitutes. The
observation constraint fixes the range of A: block means, which are the
conservative remap — coarse-cell budget preservation, itself an exact
physical statement. The hydrostatic relation between thickness
(z500 − z850) and layer-mean temperature is, to good approximation, linear
with constant coefficients, so it decomposes across the same range/null
split: it constrains both observed and unobserved components, and
enforcing it pins no block means. Projection and physics act on largely
orthogonal subspaces; the meaningful comparison is effect size.

Our hypothesis is that at this resolution the observation constraint
dominates for a specific, testable reason: the training data satisfy the
hydrostatic relation to within the reanalysis's own residual, so a
converged model has already internalized it — whereas the observation is
information the model cannot possess. The decisive measurement is
therefore not a metric delta but the hydrostatic residual itself, computed
for (a) ERA5 truth, (b) bicubic upsampling, (c) unconstrained model
samples, and (d) projected samples [pending: 20-variable runs; Section 8
arms built]. If (c) already sits at (a)'s level, there is no headroom for
any hydrostatic constraint on this relation — inference-time or
training-time — and the "physics" of physics-informed downscaling at
mesoscale resolution resides in the conservation law already embedded in
A, enforced exactly and for free. Two scope limits are stated rather than
finessed: we test constraints at inference only, claiming nothing about
physics terms in training objectives; and since §5.2's principle predicts
any per-step correction is partly erased, a null result for an
inference-time physics term is evidence about the inference slot, which is
exactly why the residual diagnostic carries the argument.

### Placeholders to resolve before submission
- [ ] Table 3: unprojected / bicubic / projected / covariance-projected
      (L2, spec-logL1, coarse-consistency) — spread-selection re-run,
      noise-floor column.
- [ ] Fig. 2: paired maps + spectra (with/without projection; cov-init).
- [ ] Fig. 3: coherence + qq diagnostics.
- [ ] Table 4 + η row: ensemble metrics at η = 0 vs swept η.
- [ ] Hydrostatic residual diagnostic (a)–(d) + constraint arms → §5.4.
- [ ] 8×/16× projection spot-checks, else scope 5.1 to 4×.
- [ ] Corrector calibration → supplement pointer only.
