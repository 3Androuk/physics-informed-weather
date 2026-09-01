# Section 1 — Introduction (draft v1)

> Journal prose; [cite] markers to be filled. Emulators kept to one clause
> until §9 tests exist. ~1,000 words ≈ 2 pp.

## 1. Introduction

Most users of atmospheric data never touch the resolution at which the
originating system was run. Operational centres disseminate and archive
forecast fields on reduced grids — commonly 1.5° for products whose native
runs are an order of magnitude finer [cite WB2] — because the storage and
bandwidth cost of full-resolution output, multiplied over variables, lead
times, and ensemble members, is prohibitive. Reanalyses are truncated for
the same reason; long climate archives are coarser still [cite CMIP6, one
sentence]. And a growing class of forecast systems is simply run coarse in
the first place [cite]. Whoever holds such a field and needs local detail —
for verification against station data, impact modelling, or as boundary
forcing — faces the same task: reconstruct a physically plausible
high-resolution state from a coarse one. Downscaling, in this form, is not
an exotic research problem but the everyday condition of working with
atmospheric archives.

The task is fundamentally underdetermined, and honest methods treat it
that way. A coarse field on a grid r times coarser than the target fixes
only the block averages of the fine state: a fraction 1 − 1/r² of the
degrees of freedom — 94% at r = 4, 99.6% at r = 16 — is unconstrained, to
be filled with atmospheric structure that is consistent with the
observation but not determined by it. The appropriate output is therefore
not a single "best" field but a calibrated ensemble of plausible fine
states, exactly as in dynamical downscaling, where a regional model driven
by coarse boundaries produces one realization of many possible; a
generative model makes the ensemble explicit and cheap. This framing puts
probabilistic scores — CRPS, spread–skill — on equal footing with pointwise
error from the outset.

Diffusion models have rapidly become the dominant tool for this task.
Residual formulations pair a regression mean with a generative correction
[cite CorrDiff]; stochastic interpolants and flow matching offer
alternative conditional transports [cite CDSI, PC-AFM, SFM]; systematic
intercomparisons now rank these families across resolutions and report
diffusion among the most physically coherent [cite GMD 2026,
CORDEX-ML-Bench, intercomparison]. A parallel line dispenses with paired
training altogether: a single unconditional diffusion model, trained only
on high-resolution fields, is guided at inference by the coarse
observation, with data consistency imposed by analytic projection or
posterior-sampling corrections [cite Shu, DDNM, ZSSD, scale-adaptive]. In
one form or another, the community has converged on a shared recipe: a
learned prior over fine-scale weather, a sampling loop that consults the
coarse field, a data-consistency mechanism, auxiliary conditioning
(coordinates, orography), and a calibration step.

What the literature has not done is measure the recipe's ingredients.
Method papers necessarily present their configuration as a package;
intercomparisons rank packages against each other. Neither isolates the
components that every package shares — whether and how to enforce
consistency with the input, how to initialize the sampler, what auxiliary
conditioning is worth, how calibration should be applied — and negative
results about individual components are almost never reported. The result
is a field in which design choices propagate by inheritance rather than
evidence, and in which it is genuinely unknown whether the performance of
a published system comes from its novel contribution or from an
unexamined default it shares with every competitor.

This paper measures the components. Working with a single guided
unconditional diffusion sampler on ERA5 fields — one variable first, then
twenty — we vary one ingredient at a time under shared noise realizations,
with permutation controls and an explicit sampling noise floor, and find
an attribution that is strongly non-uniform. A single analytic step, the
exact range–null projection that pins the sample's block averages to the
observation at every denoising step, is worth a factor of ~2 in error —
more than every conditioning, covariance, and architecture choice we
measure combined; without it the sampler is worse than bicubic
interpolation. Two statistically better-motivated refinements of the same
machinery fail instructively: a covariance-weighted projection, exactly
the correction data-assimilation practice recommends, is a measured null;
a covariance-optimal initialization is catastrophic, degrading spectral
fidelity twenty-eight-fold. These results are unified by one principle:
each denoising step erases off-manifold detail, so what survives is the
information a correction injects and the spectral energy it preserves —
not its statistical shape. Auxiliary location conditioning, dissected over
an eight-encoder ladder with a permutation control, contributes a real but
bounded ~13% and is captured almost entirely by static physiography;
learned embeddings add little that survives the controls, at one variable
and at twenty. Calibration, finally, is best applied post hoc: sampler
stochasticity and ensemble inflation preserve the analytic consistency
guarantee exactly, whereas any learned post-processor absorbs the training
resolution and forfeits generality.

The practical payoff of this anatomy is a deployment property. Because
the operator enters only through the inference-time projection, one frozen
checkpoint serves any integer coarsening ratio — including ratios never
seen in training — and, through a channel-selective form of the same
projection, inputs in which entire variables are missing, which the model
then generates rather than receives. We exercise both properties on real
1.5° forecast products with paired reanalysis truth, separating what
downscaling costs from what forecast error costs [§9; runs in progress].

Our contributions are: (i) a component-level anatomy of guided diffusion
downscaling with controls and shared noise, yielding two strong positive
results, two instructive negative ones, and an organizing principle that
explains all four; (ii) a conditioning ablation with a permutation control
and matched capacity, replicated from one to twenty variables; (iii) a
zero-shot deployment demonstration across ratio and channel availability
on operational coarse products, evaluated against paired truth; and (iv) a
compact set of evidence-based recommendations for anyone building such a
system. Section 2 positions this against the method and intercomparison
literature; Sections 3–4 describe data and methods; Sections 5–9 present
the anatomy, corroborating family comparison, conditioning ablation,
full-field reconstruction, and deployment; Section 10 discusses
limitations.
