# Section 1 — Introduction (draft v2, referenced)

> Journal prose with author-year citations; reference stubs at the bottom.
> ~1,250 words ≈ 2–2.5 pp.

## 1. Introduction

Weather is experienced locally, and most of its damage is done at scales
of kilometres to tens of kilometres: convective precipitation, orographic
wind, valley temperature inversions, the sharp gradients that decide which
watershed floods and which power line ices [Bauer et al., 2015; Maraun &
Widmann, 2018]. The systems that predict weather, however, deliver fields
at much coarser scales than the impacts they are used to assess. This
resolution gap — between what is simulated, stored, and distributed on the
one hand and what users need on the other — is the problem this line of
work addresses.

The gap is not closing by brute force. Operational numerical weather
prediction (NWP) is among the most expensive scientific computations
performed daily: the ECMWF Integrated Forecasting System runs its
deterministic forecast at ~9 km on a dedicated supercomputer, and each
factor-of-two refinement multiplies cost roughly tenfold [Bauer et al.,
2015]. In practice, therefore, what most users can actually obtain is
substantially coarser than what was run: dissemination and archive grids
of 0.25°–1.5° are standard [Rasp et al., 2024], because storing and moving
native-resolution output across variables, lead times, and ensemble
members is prohibitive. The recent generation of machine-learning weather
models — FourCastNet [Pathak et al., 2022], Pangu-Weather [Bi et al.,
2023], GraphCast [Lam et al., 2023], the probabilistic GenCast [Price et
al., 2025], and hybrid models such as NeuralGCM [Kochkov et al., 2024] —
has cut the cost of producing a forecast by four to five orders of
magnitude, but not the cost of the resolution itself: these models train
on and emit the same 0.25°–1.4° reanalysis-scale grids, so even the ML era
inherits the gap between archived fields and impact-relevant detail.
Downscaling — reconstructing fine-scale structure from a coarse field — is
the complementary lever: instead of running or storing everything fine,
generate the fine scales on demand from what is already available.

Two families of downscaling exist. Dynamical downscaling nests a
limited-area physical model inside the coarse fields and pays CPU-months
per scenario [Giorgi, 2019; Jacob et al., 2014]; statistical downscaling
learns the coarse-to-fine map from data [Maraun & Widmann, 2018].
Within the statistical family, deterministic regression is known to
produce blurred fields — the conditional mean of an underdetermined
inverse problem, penalized doubly when sharp features are displaced —
which motivated generative approaches that sample sharp, realizable
states: GANs for precipitation super-resolution and nowcasting [Leinonen
et al., 2020; Ravuri et al., 2021; Harris et al., 2022], and more recently
denoising diffusion models [Ho et al., 2020; Song et al., 2021], whose
residual form powers km-scale regional downscaling in CorrDiff [Mardani
et al., 2025], with stochastic interpolants and flow matching as
alternative conditional transports [CDSI, 2026; Fotiadis et al., 2024;
PC-AFM, 2026]. The generative framing is not a stylistic preference but a
consequence of the arithmetic: a field coarsened by a factor r fixes only
block averages, leaving 1 − 1/r² of the degrees of freedom — 94% at r = 4,
99.6% at r = 16 — unconstrained. The honest output is a calibrated
ensemble of plausible fine states, exactly as in dynamical downscaling,
where each regional simulation is one realization among many.

A second line of diffusion work removes the paired training requirement
altogether. A single unconditional diffusion model, trained only on
high-resolution fields, is guided at inference by the coarse observation
[Shu et al., 2023], with data consistency imposed by analytic range–null
projection inherited from image restoration [Choi et al., 2021; Wang et
al., 2023]; recent weather instantiations apply such zero-shot samplers
across heterogeneous inputs and scale factors [ZSSD, 2026; Hess et al.,
2025]. Across both lines, the community has converged on a shared recipe:
a learned prior over fine-scale weather, a sampling loop that consults the
coarse field, a data-consistency mechanism, auxiliary conditioning on
static geography, and a calibration step.

What the literature has not done is measure the recipe's ingredients.
Method papers necessarily present their configuration as a package, and
the systematic intercomparisons that now exist rank packages against one
another [Gutiérrez et al., 2026; CORDEX-ML-Bench, 2026; Rampal et al.,
2025]. Neither isolates the components every package shares — whether and
how to enforce consistency with the input, how to initialize the sampler,
what auxiliary conditioning is worth, where calibration belongs — and
negative results about individual components are almost never reported.
Design choices propagate by inheritance rather than evidence, and it is
genuinely unknown whether a published system's performance comes from its
novel contribution or from an unexamined default it shares with every
competitor.

This paper measures the components, and our experimental setting is
chosen for exactly that purpose. We downscale ERA5 [Hersbach et al.,
2020], the standard global reanalysis at 0.25° (~28 km at the equator):
it is public, spans decades, underlies the WeatherBench 2 evaluation
ecosystem [Rasp et al., 2024], and — decisive for an ablation study —
provides paired ground truth everywhere, so that every component change
can be scored pointwise under identical noise realizations. The coarse
inputs are block averages of the truth with an exactly known operator, at
ratios 4, 8, and 16 (1°, 2°, 4° — spanning the range of disseminated and
archived products, with 16× held out entirely from training); real coarse
products replace the synthetic degradation in the deployment experiments
of Section 9. We work on the ±60° latitude band, where the equal-angle
grid's cell geometry remains near-uniform and the bulk of exposed
population lies, and on two variable sets: 2 m temperature for the main
anatomy, and a 20-channel state (five surface fields plus five upper-air
variables at three pressure levels) for the multivariate replication.
The target scale is deliberately mesoscale rather than km-scale: CorrDiff
and its successors address the regional 25 km → 2 km problem with
bespoke high-resolution training data [Mardani et al., 2025]; we address
the complementary, global problem — the archive gap between reanalysis-
scale products and the grids users hold — in a setting controlled enough
for component attribution.

The attribution that emerges is strongly non-uniform. A single analytic
step — the exact range–null projection pinning each sample's block
averages to the observation at every denoising step — is worth a factor of
~2 in error, more than every conditioning, covariance, and architecture
choice we measure combined; without it, the guided sampler is worse than
bicubic interpolation. Two statistically better-motivated refinements of
the same machinery fail instructively: a covariance-weighted projection,
precisely the correction ensemble data assimilation recommends, is a
measured null, and a covariance-optimal initialization is catastrophic,
degrading spectral fidelity twenty-eight-fold. One principle unifies
these results: each denoising step erases off-manifold detail, so what
survives is the information a correction injects and the spectral energy
it preserves — not its statistical shape. Auxiliary location conditioning,
dissected over an eight-encoder ladder with a permutation control,
contributes a real but bounded ~13% and is captured almost entirely by
static physiography, at one variable and at twenty. Calibration is best
applied post hoc, where it provably cannot violate the consistency
guarantee.

Because the operator enters only through the inference-time projection,
one frozen checkpoint serves any integer coarsening ratio — including
ratios never seen in training — and, through a channel-selective form of
the same projection, inputs missing entire variables, which the model
generates rather than receives. We exercise both properties on
operational 1.5° forecast products with paired reanalysis truth
[Section 9; runs in progress].

Our contributions are: (i) a component-level anatomy of guided diffusion
downscaling with controls and shared noise realizations — two strong
positive results, two instructive negative ones, and the principle that
explains all four; (ii) a conditioning ablation with matched capacity and
a permutation control, replicated from one to twenty variables; (iii) a
zero-shot deployment demonstration across ratio and channel availability
on operational coarse products against paired truth; and (iv) a compact
set of evidence-based recommendations for building such systems.
Section 2 positions this work in the method and intercomparison
literature; Sections 3–4 describe data and methods; Sections 5–9 present
the anatomy, the corroborating family comparison, the conditioning
ablation, full-field reconstruction, and deployment; Section 10 discusses
limitations.

---

## Reference stubs (to be formatted)
- Bauer, Thorpe, Brunet (2015). The quiet revolution of numerical weather
  prediction. Nature 525.
- Maraun & Widmann (2018). Statistical Downscaling and Bias Correction for
  Climate Research. Cambridge UP.
- Hersbach et al. (2020). The ERA5 global reanalysis. QJRMS 146.
- Rasp et al. (2024). WeatherBench 2. JAMES 16 (2023MS004019).
- Pathak et al. (2022). FourCastNet. arXiv:2202.11214.
- Bi et al. (2023). Pangu-Weather. Nature 619.
- Lam et al. (2023). GraphCast. Science 382.
- Price et al. (2025). GenCast. Nature.
- Kochkov et al. (2024). NeuralGCM. Nature 632.
- Giorgi (2019). Thirty years of regional climate modeling. JGR-A.
- Jacob et al. (2014). EURO-CORDEX. Reg. Environ. Change 14.
- Leinonen, Nerini, Berne (2020). Stochastic super-resolution GAN for
  precipitation. IEEE TGRS.
- Ravuri et al. (2021). Skilful precipitation nowcasting (DGMR). Nature 597.
- Harris et al. (2022). Generative precipitation downscaling. JAMES.
- Ho, Jain, Abbeel (2020). DDPM. NeurIPS.
- Song et al. (2021). Score-based SDEs. ICLR.
- Mardani et al. (2025). CorrDiff / residual corrective diffusion.
  arXiv:2309.15214 (Comm. Earth & Env.).
- CDSI (2026). Climate downscaling with stochastic interpolants.
  arXiv:2603.03838.
- Fotiadis et al. (2024). Stochastic flow matching. arXiv:2410.19814.
- PC-AFM (2026). Physics-constrained adaptive flow matching.
  arXiv:2604.03459.
- Shu, Li, Barati Farimani (2023). Physics-informed diffusion for flow
  reconstruction. JCP 478.
- Choi et al. (2021). ILVR. ICCV.
- Wang, Yu, Zhang (2023). DDNM. ICLR (arXiv:2212.00490).
- ZSSD (2026). Zero-shot statistical downscaling via DPS. arXiv:2601.21760.
- Hess et al. (2025). Fast, scale-adaptive, uncertainty-aware downscaling.
  Nature Mach. Intell.
- Gutiérrez et al. (2026). Inter-comparison of generative AI for
  downscaling. GMD 19 (verify author list).
- CORDEX-ML-Bench (2026). arXiv:2606.29172.
- Rampal et al. (2025). Intercomparison of generative ML downscaling.
  arXiv:2512.13987 (verify author list).
