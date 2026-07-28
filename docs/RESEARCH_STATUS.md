# RAE Geometry Research Status

## Current engineering mainline

The active run has moved to the official RAEv2 DINOv3-L-K7 full training state.
It uses a staged continuation gate: first evaluate a 2,000-step Flow-only
control, then run the preregistered 5,000-step Flow/LPL comparison only if the
control remains healthy. Checkpoints are saved every 1,000 steps and all
comparisons reuse labels and initial sampling noise. The protocol is recorded
in `docs/RAEV2_LPL_STRICT_CONTINUATION_ZH.md`.

The remainder of this document records earlier RAE geometry, transport,
spectral, prior-decoder, and LPL evidence. These experiments remain useful
audit history but are not the current training mainline.

## Scope

The repository now contains two distinct research lines.  The earlier geometry
line asks whether frozen RAE tokenizers contain a usable group response; its
adapter attempts did not improve generation.  The current spectral line asks
why a fixed inverse-variance direction loss improves held-out teacher metrics
but worsens rollout.  These lines must not be combined into a claim that
RAE-DINOv2 has a global group representation.

## Latest controlled generative result: prior-decoder mismatch

The Imagenette-64 `16/64/256d x 5 seeds` study confirms a stable trade-off:
empirical latent decoding improves with capacity, while a same-budget latent
prior produces modeling gaps of `2.79/9.92/22.42` FID. Ordinary flow loss,
SWD, effective rank, and condition-space C2ST do not predict the decoded gap.
An equal-angle intervention rejects the stronger claim that prior errors align
with unusually sensitive local decoder directions. Cross-fitted class-mass
reweighting recovers only `15.9%` of the 256d gap; the exact remaining FID
decomposition is dominated by covariance/diversity mismatch.

Post-hoc spectra show strong diminishing intrinsic variance: raw effective
rank changes only `9.94 -> 16.31 -> 20.14`, and decoder-condition effective
rank changes `8.36 -> 16.17 -> 17.91`. This supports low-dimensional added
variance, not yet low-dimensional added information. A within-model PCA
truncation/residual decoding intervention is still required for the stronger
claim. Full evidence and boundaries are in
`docs/IMAGENETTE_DECODER_AMPLIFICATION_RESULTS_ZH.md`.

The next registered direction is decoder-aware prior training. Literature and
the current covariance-dominated decoded gap support exposing the frozen
decoder's response to the prior objective before jointly updating model
parameters. The first comparison is flow-only versus latent, condition,
pairwise decoder-feature, and batch decoder-response distribution losses. Full
rationale and stop/go criteria are in
`docs/PRIOR_DECODER_ALIGNMENT_LITERATURE_AND_PLAN_ZH.md`.

That direction has now failed its mandatory no-training gate. A frozen-decoder
response atlas over `16/64/256d x 5 seeds`, repeated at 256 and 1024 samples,
finds no adjacent layers whose conditional-response Fréchet predicts modeling
gap under both leave-seed-out and leave-dimension-out evaluation. At 1024
samples every primary LODO correlation is negative; up2 is `-0.907`. The
decoder does use latent conditions, as shuffling worsens paired velocity MSE by
`6.87x/3.49x` at high/mid noise in all 15 runs. Formula, trace, hash, power and
projection-seed audits pass. This supports a pathwise joint-mismatch
interpretation, not the proposed fixed-time marginal response loss. Moment
repair, A3-A5 training, decoder adapters and joint training are not authorized.
See `docs/IMAGENETTE_DECODER_RESPONSE_ATLAS_RESULTS_ZH.md`.

## Current active result: latent trust spectrum

The latest five-seed SPC study closes the current subspace-path curriculum as a
method.  SPC worsens paired 5k-sample FID by `+6.30` on average (`0/5` seeds
improve both FID and KID).  Its fitted rank-16 basis captures `94.65%` of the
top-PCA energy optimum, so the intervention suppressed high-variance,
high-SNR directions rather than a cleanly separated detail subspace.

A variance-normalized generalized eigen diagnostic now separates cross-layer
predictability from final-latent variance.  On held-out ImageNet validation,
the whitened rank-16 basis reaches `R2=0.931` while overlapping top PCA by only
`0.198`; its train-half stability is `0.926`.  Across five frozen stage-2
training seeds, 128 held-out latents, six noise times, and 24 rank-16 blocks,
directional response changes smoothly from variance-dominated at low noise to
jointly variance- and predictability-dependent at high noise.  At `t=0.95`,
the two standardized coefficients are `0.542` and `0.558`, and together explain
`94.7%` of block-level log-gain variance.

The inference-only teacher-path rollout gate has now passed on the latent side.
Across 24 blocks, one-step gain predicts endpoint leverage with mean Spearman
`0.961` at `t=0.85` and `0.990` at `t=0.95`, versus `0.665` at `t=0.30`.
Three variance-matched high-/low-predictability pairs have endpoint gain ratios
of `2.83x`, `4.01x`, and `4.87x` at `t=0.95`, with all five seeds agreeing and
the same ordering at two perturbation amplitudes. Removing the high-
predictability clean component also causes more clean-endpoint error at high
noise, supporting a useful-signal-anchor interpretation rather than a harmful
shortcut.

This remains a mechanism result, not a generation improvement.  A small frozen-
decoder spot check is the only remaining gate before one single-seed reverse
trust-curriculum pilot.  Full evidence and stop/go criteria are in
`docs/RAE_LATENT_TRUST_SPECTRUM_RESULTS_ZH.md`.

Keep the following concepts separate:

1. Direct spatial equivariance: `E(gx) ~= P_g E(x)`.
2. Per-transform linear alignability: a separately fitted channel map aligns
   `P_g E(x)` with `E(gx)`.
3. Group consistency: maps fitted only for generators predict composed actions
   on held-out images, for example `C_90^2` predicts `rot180`.

Only the third level is evidence for an approximate group representation.
Decoder output from `D(P_g E(x))` is an OOD-latent robustness test, not an
encoder-side group-structure test.

## What the current evidence says

The largest frozen layerwise study uses 5,196 ImageNet test images and is
stored outside Git at
`$EQVAE_DATA_ROOT/artifacts/layerwise_imagenet/imagenet_test_n5196_3rae_d4`.
For RAE-DINOv2, sample-centered direct `rot90` error falls from `1.366` at the
patch input to `0.852` at `final_raw`; for `flip_h`, it falls from `1.263` to
`0.434`. This is meaningful deep-layer geometric alignment, but the residual
error is far from a direct equivariance identity. RAE-MAE does not share the
same `rot90` trend in this study: `1.186` at the patch input and `1.217` at
`final_raw`.

The appropriate current conclusion is therefore: DINOv2 becomes more
geometrically aligned with depth, especially for flips, but this alone does
not establish a clean global `C4` or `D4` representation. Any claim about
channel maps must still pass held-out generator-power and D4-relation tests.

A stricter 5,196-image correspondence audit now rejects the simple
homogenization explanation but also rejects a monotonic-depth story.  For
`rot90`, exact token correspondence rises from `1.77%` at patch input to
`67.97%` at hidden layer 9, then falls to `43.00%` at final raw while direct
error continues to improve.  For `flip_h`, it peaks at `92.22%` at layer 9 and
ends at `83.77%`.  Spatial effective rank falls from `226.0` to `38.7` tokens.
Thus middle layers form genuine correspondence, while late semantic/low-rank
compression improves average alignment but reduces spatial uniqueness.  See
`docs/RAE_LAYERWISE_CORRESPONDENCE_RESULTS_ZH.md`.

## Adapter attempts and outcome

The encoder-side adapter experiment uses an invertible additive-coupling map
and trains on ImageNet train images with held-out validation/test splits. The
latest full-image variant used `flip_h`, `flip_v`, and `rot180` with the RAE
encoder and decoder frozen. It did not establish a generation improvement.

The decoder-side inverse-adapter pilot is a useful negative control. It
optimized only the inverse adapter through a frozen RAE decoder on 8,000
ImageNet training images. On a fixed-noise 2,048-image validation check,
noisy reconstruction L1 decreased from `0.17418` to `0.16324`, while noisy
latent relative error increased from `0.35395` to `0.36675`. In matched 5,000
sample ADM-FID evaluation, the exact-inverse baseline scored `19.1944` and the
reconstruction-trained inverse adapter scored `19.7137`. Thus improving this
image reconstruction objective did not improve generation.

| Run | Samples | ADM-FID | Interpretation |
|---|---:|---:|---|
| official DINOv2 DiT-S epoch 14 | 50,000 | 12.8641 | official reference checkpoint |
| encoder-adapter DiT-S fine-tune, step 3,750 | 50,000 | 13.7027 | no improvement over that reference |
| encoder-adapter DiT-S exact inverse | 5,000 | 19.1944 | comparison baseline for inverse-adapter pilot |
| frozen-decoder inverse-adapter pilot | 5,000 | 19.7137 | worse generation despite better noisy reconstruction |

The 5,000- and 50,000-sample values must not be compared as a single ranking;
they use different sample counts and serve different controlled comparisons.

### Transport-compatibility audit

The active geometry-related question is no longer whether equivariance is
intrinsic to the RAE latent.  The invertible adapter is now treated as a
controlled coordinate intervention.  A phase-0 audit established that the old
adapted stage-2 runs used a standard Gaussian source and a straight path to
`f(z)`, rather than the transformed source `f(eps)` or the pushforward path
`f((1-t)z+t eps)`.  Those runs therefore confound source-prior mismatch and
conditional-path mismatch.

The real checkpoint passes cycle, decoder-identity, JVP, identity-path and
orthogonal-path numerical checks.  However, the old 20k original/adapted runs
used different world sizes, and the training code derives the pre-init rank
seed from world size.  Their model initialization and random streams were not
paired.  See `docs/LATENT_TRANSPORT_PHASE0_AUDIT_ZH.md` for the evidence and
`docs/LATENT_TRANSPORT_RESEARCH_PROTOCOL_ZH.md` for the staged stop/go rules.

The phase-2 no-training audit is now complete on 2,048 disjoint ImageNet
validation images. It cleanly separates source-prior mismatch from nonlinear
path curvature: anisotropic linear maps increase prior SW1 monotonically while
their chord defect remains numerical zero; the scaled real adapter increases
bridge defect from `0.091` to `0.293` and chord/pushforward velocity gap from
`0.175` to `0.903`. All implementation controls passed. At the same time,
projected VIV is nearly unchanged and the Gaussian-straight local ambiguity
proxy decreases as adapter strength grows. These proxies are therefore not
accepted as transport-quality predictors. See
`docs/LATENT_TRANSPORT_PHASE2_RESULTS_ZH.md`; only a paired four-path toy is
authorized next.

That four-path toy is now complete and stops the transport-recovery method
line. The experiment is valid: all numerical/solver controls pass, Base covers
all eight modes in 5/5 seeds, and Gaussian-straight is at least 10% worse than
Base in 4/5 primary-strength seeds. Strict Pushforward nevertheless recovers
half the gap in 0/5 seeds and averages `2.60x` Base sliced-W1. A post-hoc
no-training conjugacy audit pushes the trained Base field through the exact
Jacobian and recovers its same-noise endpoint to at most `7.56e-6` mean relative
error at 400 Heun steps. Thus the path implementation is correct; the ordinary
finite MLP class and Euclidean velocity loss are not closed under nonlinear
coordinate changes. Per the preregistered gate, no small-image, CIFAR, or RAE
stage-2 experiment is authorized. Full evidence and theory are in
`docs/LATENT_TRANSPORT_PHASE3_RESULTS_ZH.md`.

## Current repository entry points

- `notebooks/latent_playground.ipynb`: reconstruction and latent editing
  interface. Use it for decoder closure and visual inspection.
- `notebooks/dinov2_token_diagnostic.ipynb`: encoder-side token correspondence,
  orbit alignment, Procrustes, and group-consistency diagnostics.
- `notebooks/rae_layerwise_playground.ipynb`: simple per-image and batched
  layerwise visualizations for DINOv2, MAE, and SigLIP2.
- `notebooks/mnist_spectral_rollout_toy.ipynb`: low-cost paired MNIST
  teacher-path/rollout diagnostic with the simple `T(...)` and `V(result)`
  interface.
- `experiments/rae_layerwise_imagenet_study.py`: reproducible larger-scale
  layerwise study.

## Spectral flow-matching direction

The geometry-adapter line above has not produced a generation gain. A separate
mechanism audit found a stronger real-RAE asymmetry in stage-2 flow matching:

- radial DCT residual energy spans about `23.5x` in the executed audit and is
  much more anisotropic than a random orthogonal basis;
- low-frequency bands have higher teacher predictability;
- the frozen decoder is relatively more sensitive to higher-frequency,
  lower-residual bands;
- direction-only `gamma=0.5` flattens output-head gradient-variance slopes, but
  does not improve per-band GSNR and reduces total projected GSNR.

This supports an objective/capacity-allocation hypothesis, not a pure numerical
preconditioning or training-acceleration claim. The mini-DiT toy result is also
slightly negative, so only one strict real-RAE tiny screen is authorized:
`DiTDH-S step-5000 -> step-10000`, three paired seeds, `gamma=0` versus `0.5`,
fixed fp32 numerics, fixed train-only weights, and fixed-noise 5k KID/FID proxy.
The pass rule is preregistered in
`$HOME/data/eqvae/experiments/rae_spectral_tiny/protocol.json`.

The screen is now complete and failed its preregistered practical threshold.
Across three paired seeds, fixed held-out velocity diagnostics at 5k branch
updates improved raw MSE by `1.09%`, the decoder-sensitivity proxy by `1.60%`,
high-frequency MSE by `3.81%`, and the highest-frequency band by `4.17%`, while
the lowest-frequency band worsened by `1.26%`. This confirms that the fixed
weighting changes direction allocation on unseen images. However, exact 5k
generation improved KID by only `0.65%` to `1.89%` (`0/3` seeds reached the
preregistered `5%` threshold), and the 5k FID proxy worsened on `2/3` seeds.
The method line is therefore stopped without tuning gamma or adding seeds.

### Teacher-forced to rollout gap

The failure is now localized more precisely with a no-training audit of all
six endpoint EMA branches.  Each branch uses the same 64 cached ImageNet
validation latents, fixed noise and labels, official shifted 50-step Euler
times, fp32, and the exact frozen ViT-XL decoder.  Decoder/Inception metrics use
12 fixed samples per branch.  The aggregate tables live outside Git at
`$HOME/data/eqvae/experiments/rae_spectral_tiny/teacher_rollout_gap_*.csv`.
The 64 audit indices have zero overlap with the 2,048 ImageNet-train indices
used for the fixed spectral statistics; they are contained in the disjoint
512-image validation cache.

The most important result is counterintuitive.  On known linear interpolation
states, `gamma=0.5` genuinely improves the supervised vector field: for the
middle times it lowers latent error, raises per-band prediction/target cosine,
moves prediction energy toward target energy, and moves the regression slope
toward one on all three seeds.  This is not a train-set-only effect or a trivial
zero-amplitude predictor.  Yet its self-generated ODE states are farther from
the corresponding validation interpolation marginal at every measured time.
The final state has a larger energy deficit in bands 1--7; for band 7 the mean
log-energy ratio changes from about `-0.30` to `-0.49`.

There are two additional sign reversals:

- at `t ~= 0.69`, teacher-forced latent MSE improves by about `3.8%` and
  Inception cosine distance by `8.9%`, while decoded pixel MSE worsens by
  `3.2%`;
- early in rollout, the treatment has a smoother same-noise vector-field
  secant, but after `t ~= 0.69` its secant sensitivity becomes larger and ends
  about `8.8%` above baseline.

The supported mechanism is therefore a teacher-forcing/transport gap:
per-sample velocity accuracy does not control multi-step marginal covariance,
volume change, or perceptual distribution quality.  Pixel reconstruction is
also not a monotone proxy for semantic quality.  The next quality candidate
must directly constrain a short rollout's per-band covariance or log energy,
possibly with a one- or two-step differentiable rollout, rather than tuning the
fixed DCT `gamma`.

A hard vector-field time-switch probe tests the initially plausible causal
story that the high-noise band-0 sacrifice is the main trigger.  It does not
support that simple story.  On the same 64 noises and validation distribution,
using the partial vector field only at high noise (`t >= 0.85`), only in the
middle (`0.30 <= t < 0.85`), or only in the final two low-noise evaluations
worsens endpoint summary SWD by `5.49%`, `11.74%`, and `2.52%`, respectively.
Every result has the same sign on all three training seeds.  The middle window
has the largest cumulative effect, while the final two evaluations have the
largest damage per evaluation.  This SWD is deliberately a low-dimensional
probe over projected token means and log-band energies, not a full latent
Wasserstein distance or a replacement for FID.

The frequency direction is also counterintuitive: every partial-only window
makes endpoint band 0 slightly closer to the validation energy while making
bands 1--7 more under-dispersed.  Thus the deficit is not the direct transport
of one underfit coarse coefficient.  It is consistent with cross-frequency
coupling and excessive contraction in the learned vector-field Jacobian.  A
time-gated splice of the existing checkpoints is not a quality candidate.

There is a useful theoretical boundary.  The experiment's fixed positive
frequency matrix `W(t)` is independent of the sample.  With an unrestricted
function class, weighted and unweighted squared losses have the same
population minimizer, `E[u | z_t,t]`.  The observed changes therefore come
from finite capacity, shared parameters, and optimization-path reallocation;
the weighted loss is not a pure preconditioner that preserves the finite-model
solution.

### Low-cost MNIST mechanism gate

A bounded MNIST experiment now reproduces the qualitative teacher/rollout gap
without a large tokenizer or decoder.  It uses normalized pixels as an exactly
invertible latent, the same radial DCT mean-one weighting, paired small
convolutional velocity fields, 8,192 official training images, 1,024 disjoint
official test images, 1,000 updates, and three seeds.  All weighting statistics
and the independent MNIST feature classifier use training data only.  Decoded
feature and pixel metrics clamp generated pixels to the valid `[0, 1]` range;
the raw latent metric remains unclipped.

For the width-24 model, `gamma=0.5` improves the combined held-out teacher MSE
at `t in {0.1, 0.3, 0.5}` by `3.62%` on average, but worsens it at
`t in {0.7, 0.9}` by `1.50%`; averaged over all five measured times the net
change is only `-0.25%`.  Despite that small aggregate teacher improvement,
all three seeds worsen every rollout distribution metric.  Mean
weighted/baseline ratios are `1.093` for raw latent SWD, `1.109` for decoded
pixel SWD, `1.054` for feature SWD, and `1.163` for the MNIST feature FID.

A preliminary width check adds an important limit to the finite-capacity
story.  Increasing width makes the teacher-path reallocation much smaller,
but does not monotonically remove the raw-latent rollout penalty and can expose
seed-dependent rollout instability.  Finite capacity therefore helps explain
why the weighted objective changes teacher errors, but capacity alone is not a
sufficient explanation of the accumulated transport error.  The later
matched-basis and exact-skip experiments identify data-geometry alignment and
an unprotected high-energy subspace as the missing factors.

This is a cheap mechanism reproduction, not a replacement for ImageNet FID and
not a reason to tune `gamma` further.  Its practical use is to screen rollout-
aware losses, time windows, and Jacobian/covariance diagnostics before any new
RAE run.

### Mechanism resolution

The follow-up causal probes now resolve the teacher/rollout gap more sharply;
the full Chinese evidence report is in
[`TEACHER_ROLLOUT_MECHANISM_ZH.md`](TEACHER_ROLLOUT_MECHANISM_ZH.md).

The final supported mechanism has two phases.  At high noise, partial's
pointwise teacher MSE improvements do not preserve the band marginal drift
`2 E[<z_b,v_b>]`: its teacher-state drift RMSE is `2.77x` baseline at
`t ~= 0.95` and `3.24x` at `t ~= 0.85`.  In the middle interval, partial drift
is better on teacher states (`0.434x` at `t ~= 0.54`, `0.752x` at
`t ~= 0.315`) but worse when both fields are evaluated on the same baseline
rollout states (`1.121x` and `1.197x`).  This is direct evidence for an
on-path objective/transport mismatch followed by off-path generalization
failure.

Time-and-band splicing is causal evidence rather than a correlation.  On RAE,
the nonzero bands explain about `66%` of the high-window summary-SWD damage and
`81%` of the middle-window damage; band 0 explains only about `34%` and `16%`.
Thus the MNIST coarse-band mechanism is real but not a complete RAE account.
Euler discretization, radial-band energy alone, monotone global divergence, and
a single coarse band are all rejected as sufficient explanations.  No more
fixed-gamma tuning is justified.  A future candidate must constrain marginal
drift on one- or two-step self-generated states while preserving high-noise
coarse-direction MSE.

The bounded follow-ups are now:

1. directly match train-only per-band covariance or log energy after one or
   two differentiable rollout steps, concentrating diagnostics in the middle
   window and final two high-leverage steps;
2. test train-only endpoint spectral variance calibration before any retraining;
3. if revisiting the FxLMS analogy, use a parameter-space optimizer
   preconditioner designed to preserve the original MSE stationary points,
   rather than another output-weighted loss.

All three require KID/FID validation because amplifying decoder-sensitive high
bands can also amplify artifacts.

### Predictive small-image mechanism resolution

The preregistered five-seed MNIST/FashionMNIST follow-up is complete.  Its full
Chinese report is
[`TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md`](TRANSPORT_REVERSAL_MECHANISM_STUDY_ZH.md),
with sequential predictions in
[`SMALL_TRANSPORT_GAP_PREREG_ZH.md`](SMALL_TRANSPORT_GAP_PREREG_ZH.md).

DCT, train-only PCA, and random orthogonal losses use the same time-dependent
weight spectrum and bit-identical per-seed baseline checkpoints.  On
FashionMNIST, DCT/PCA improve low/middle teacher MSE but worsen feature FID in
all five seeds, with mean ratios `1.461/1.787`; random orientation is neutral
at `1.003`.  MNIST reproduces the ordering in four of five seeds.  High-window
group-0 field splices explain about `90%-101%` of DCT/PCA FID damage on both
datasets, whereas the same random-basis group is neutral.  Keeping baseline
group 0 while using weighted nonzero groups therefore removes almost all
high-window damage.

The mechanism is time-dependent covariance-aligned risk reallocation.  At high
noise, inverse-residual-variance weights downweight the high-energy DCT/PCA
subspace; a shared raw-velocity network sacrifices that subspace while reducing
many low-energy errors.  Removing the exact linear skip in the analytic mixture
toy makes high-variance direction error and endpoint coordinate W1 worse in all
five seeds at both tested widths, while the skip-protected residual model does
not reverse.  Off-path amplification remains present in RAE, but teacher
restart proves it is not required for the small-image failure.

The subsequent preregistered training-side and gradient interventions sharpen
this into a four-factor mechanism: risk-budget shift, weak or conflicting
coarse/detail gradients, high endpoint leverage in the neglected subspace, and
no protected transport path.  On FashionMNIST, DCT/PCA shared-trunk gradient
cosines are `+0.165/-0.318`, versus `+0.947` for the matched-spectrum random
basis; their weighted coarse-descent ratios are `0.213/0.161`, versus `0.992`.
MNIST reproduces the separation at `0.238/0.224/0.968`.  Thus PCA can exhibit
true gradient conflict, while DCT is better described as weak-coupling neglect.

A direction-specific additive coarse guardrail improves Fashion DCT/PCA
feature-FID ratios from `1.461/1.787` to `1.053/0.923`, while a matched-total-
loss-scale control remains at `1.405/1.709`.  This is predictive causal support
for the risk direction, not a finished method: the guardrail retains only about
`9%/15%` of the original detail-MSE gain.  Parameter-matched split and
asymmetric paths protect coarse outputs but either reduce per-task capacity or
underfit the coarse branch.  The mechanism is now well localized, but no
training-side intervention has passed the preregistered absolute-quality and
detail-retention Pareto criteria.

### Frozen RAE mechanism bridge

The small-image mechanism has now passed two preregistered, no-training tests
on the three existing tiny RAE step-10000 EMA pairs.  P20 computes actual
autograd gradients only for the final DiTDH-S transformer block and output
linear on held-out cached ImageNet latents.  DCT and fixed random orthogonal
bases use the same eight weight eigenvalues.  At `t=0.85/0.95`, baseline DCT
has a band-0/nonzero gradient cosine of `0.098` and a band-0 descent ratio of
`0.244`; the matched-spectrum random basis gives `0.836` and `0.952`.
Partial checkpoints and the output head reproduce the same separation.  All
24 seed/time/checkpoint/parameter-group DCT conditions have descent ratio below
`0.5`.

P21 then tests the prediction at the rollout endpoint.  In high-window
individual-band field splices, band 0 is the most damaging band in all three
training seeds.  For stable bands 0--4, `1 - descent_ratio` and endpoint
summary-SWD damage have Spearman `1.0` in every seed.  The eight individual
effects add to high-all within `-1.03%` to `+2.37%`; band 0 contributes `35.8%`
and seven smaller nonzero effects contribute `64.2%`.  This resolves the old
apparent contradiction: nonzero bands dominate in aggregate because they
accumulate, not because one high-frequency direction has more leverage than
the coarse band.

The supported tiny-RAE mechanism is therefore two-stage.  High-noise
covariance-aligned risk allocation combines with weak cross-band gradient
coupling and endpoint leverage; the resulting state shift is then amplified
by the already measured middle-window off-path sign reversal.  This is a
frozen latent-proxy mechanism result, not a 50k ImageNet FID result and not a
successful new training method.

### Mechanism-to-quality gate

Three further preregistered small-image tests ask whether the resolved
mechanism can improve absolute generation quality.  A zero-initialized
width-12 residual adapter freezes the complete raw baseline field and can only
emit groups 1--7.  Same-state group-0 error is therefore exactly unchanged.
On FashionMNIST DCT, this reduces the raw weighted feature-FID ratio from
`1.461` to `1.009`, removing about `98%` of the added damage; PCA falls from
`1.787` to `1.208`, removing about `74%`.  However, DCT retains only `42%` of
the detail-MSE gain and PCA remains materially worse than baseline.  The
absolute-quality Pareto gate fails.

The initially proposed self-generated-state target is specifically rejected.
Off-path paired MSE alone gives DCT FID ratio `1.131`, almost reproducing the
combined rollout/drift result `1.147`; normalized drift alone gives `1.034`,
close to teacher-only `1.009`.  The original pair velocity is not a valid
conditional target after the state leaves the interpolation marginal.

Finally, a no-training trust-region study excludes all previously viewed test
images, selects residual scale on 1,024 new validation images, and reports on
another 1,024 images with new noise.  Three of five seeds select scale zero,
two select scale one, and none select an intermediate scale.  Final-test mean
FID ratio is only `0.9978`, below the preregistered `2%` improvement threshold.
Thus the mechanism currently supports harm prevention, not a stable positive
quality method.  MNIST replication and tiny-RAE adapter training are not
authorized by the gate.  The full report is
[`MECHANISM_TO_QUALITY_STUDY_ZH.md`](MECHANISM_TO_QUALITY_STUDY_ZH.md).

### Sampling-time probe

A three-seed numerical probe compares reduced Euler grids with the same-model
50-step endpoint.  Contrary to the initial intuition, moving time points toward
the low-noise region is much worse than preserving the official shifted time
warp.  Although state updates are largest late, velocity change per unit time
is largest near `t > 0.97`, so small high-noise updates are not automatically
safe to skip.

Subsampling the official 50-point grid to 25 points gives about `2.02x`
measured speedup.  The directly deployable, officially recomputed
`num_steps=25` grid has mean endpoint latent relative RMS `0.0503` and
Inception cosine distance `0.0296`; the exact 50-grid subsample is only
marginally better at `0.0490` and `0.0285`.  Sixteen points increase these
proxies to about `0.112` and `0.074`.

The fixed 5k check is complete under the same two-process, equal-label,
fp32/no-TF32 sampling contract as the 50-step reference.  For seed 3407
baseline, reducing from 50 to 25 steps changes FID from `132.6809` to
`135.9916` (`+2.50%`, worse), KID from `0.121259` to `0.126041` (`+3.94%`,
worse), and IS from `8.4163` to `8.0732` (`-4.08%`).  An earlier four-process
run with a different noise stream gave the same direction and similar
magnitude.  The 25-step grid is therefore a real twofold quality--speed trade,
not a lossless acceleration.  One checkpoint and one sampling seed are enough
to reject the lossless claim, but not to estimate a precise Pareto curve.

### Transport-risk atlas and path-dependent reversal

A leakage-audited baseline-only atlas was evaluated on the completed
FashionMNIST and MNIST isospectral studies. It perfectly ranks the six
dataset/basis means, but its within-basis seed Spearman is `-0.027`; aggregate
Spearman, ROC-AUC, and fixed-threshold sign accuracy are `0.568`, `0.733`, and
`0.733`, so every prospective gate fails.

The failure is not endpoint sampling noise. A 4,096-image, five-rollout-seed
checkpoint resample confirms a stable reversal: MNIST seed 3 DCT/PCA ratios are
`1.753/2.261`, while seed 4 gives `0.672/0.789`. Exact hash-matched training
replays show that both structured bases look favorable during the first 100
steps; the final sign is set by later basin drift. A step-100 band-0 signal
found exploratorily on MNIST has Spearman `0.855`, but the same frozen test on
FashionMNIST gives only `0.067`. Final train-only rollout moments correlate
with FID damage (`0.802` across 20 structured-basis conditions), but only after
the full training path and with weak sign accuracy.

Therefore static gradient geometry, a final-checkpoint local directional
derivative, and a 10%-budget point probe are all rejected as general endpoint
predictors. The full report is
[`TRANSPORT_RISK_ATLAS_RESULTS_ZH.md`](TRANSPORT_RISK_ATLAS_RESULTS_ZH.md).

### Spectral theory audit and fixed-multiset order gap

A two-round literature audit and a new set of controlled toys now separate the
parts of the spectral result that existing theory explains from the remaining
gap.  The full Chinese report is
[`SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md`](SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md).

Three old ambiguities are resolved.  First, an exact linear least-squares toy
shows that isospectral output metrics can produce effective Hessian condition
numbers `10000` versus `2164.8`; the spectrum of `W` alone is insufficient
because training is controlled by `J^T W J`.  Second, DCT sign scrambling
preserves every coefficient power and eight-band energy to fp32 precision but
reduces MNIST/FashionMNIST classifier accuracy to about `0.16`; marginal band
power cannot certify semantics.  Third, a nested seed factorial localizes the
MNIST sign reversal to stochastic-interpolant time draws first, Gaussian bridge
noise second, and not ordinary minibatch order or initialization.

The most important new result fixes the entire seed-3 time multiset: all
`128000` float32 values, data, initialization, minibatches, Gaussian noise,
optimizer, and evaluation are identical, and only the 1000 training-step order
is permuted.  Across 12 permutations, baseline feature-FID ranges from `66.34`
to `219.25`, while the weighted/baseline ratio ranges from `0.736` to `2.112`;
four permutations help and eight hurt.  Source-stream hashes are identical on
all four GPUs, and tests verify exact sorted-multiset equality.

A 12-time-seed IID sweep also corrects the two-seed narrative.  On MNIST,
baseline and weighted mean FID are `104.57/104.61`; weighting mainly reduces
cross-seed variance in that particular setup.  FashionMNIST does not reproduce
this: means are `96.24/139.62`, with weighting harmful in `11/12` seeds and
increasing variance.  Per-batch stratified time sampling reduces some baseline
variance but does not stabilize the paired effect.  Therefore the current
method has no stable average benefit, and generic Monte Carlo variance
reduction is not a sufficient explanation.

Existing NTK, anisotropic-SGD-noise, random-reshuffling, phase, and flow-error
theories explain why these effects are possible.  They do not predict the sign
or magnitude of endpoint semantic quality for a held-out time ordering.  The
remaining research gap is a pathwise theory linking the time-conditioned
non-commuting gradient/Hessian cocycle under `W(t)` to rollout endpoint
quality.  The next justified low-cost diagnostic is a held-out-prediction test
of Hessian commutator and endpoint-adjoint statistics, not another gamma or RAE
training sweep.

## Research decision

### Latent trust and decoder alignment

The failed SPC line produced a reproducible mechanism result rather than a new
training method.  Across five small stage-2 seeds, matched-variance latent
directions cross over from variance-dominated response at low noise to
cross-layer-predictability-dominated response at high noise.  Fifty-step
teacher-path rollouts preserve this crossover.

Frozen decoding reveals a two-stage leverage handoff.  At low noise the more
predictable directions contract more in the ODE but are amplified more by the
decoder; at high noise the ODE leverage itself dominates.  All three matched
pairs and all five seeds preserve larger decoded L1/LPIPS effects for the more
predictable direction across the tested times.

A direct 24-block decoder atlas strengthens the population-level mechanism:
held-out encoder cross-layer predictability alone explains `0.819--0.860` of
the variance in decoder hidden sensitivity, versus `0.156--0.182` for latent
direction variance.  This relation holds within each absolute, fractional,
and PCA basis family.  It is not a simple decoder-input scaling effect: the
exact linear embed gain spans only `1.47x`, while the first recorded decoder
hidden response spans about `11.4x`.

The method gate nevertheless fails.  A static predictability metric reaches
only `0.065` median per-time Spearman with full decoder LPL and `0.021` median
gradient cosine.  Even an oracle static metric fitted from the decoder atlas
reaches only `0.036/0.029`.  Dynamic decoder prefixes improve as depth grows,
but prefix 3 already uses most of the decoder and reaches only `0.728/0.493`,
with high-noise gradient cosine `0.229`.

Therefore no static predictability-weighted loss, SPC schedule, or truncated
decoder training is authorized.  The supported finding is the gap between a
stable population-level semantic axis hierarchy and state-dependent,
sample/token-specific perceptual correction.  The Chinese report is
[`RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md`](RAE_LATENT_TRUST_DECODER_ALIGNMENT_RESULTS_ZH.md).

The spectral failure mechanism is sufficiently localized for the current
scope; more fixed-`gamma`, basis, time-splice, residual-scale, seed, or 50k-FID
diagnosis is not the next useful spend.  The protected-residual gate prevented
harm but did not beat the absolute baseline, and paired off-path MSE was
actively harmful.  No paired tiny-RAE screen is authorized.

The final low-cost candidate, train-only rollout checkpoint selection, has now
been prospectively rejected on unseen seeds 5--9.  In MNIST, selected DCT/PCA
remain `1.238x/1.346x` worse than the fairly selected baseline, with zero of
five seed wins.  FashionMNIST gives `1.249x/1.422x`, with one and zero seed
wins.  The proxy itself improves by about 46% across all conditions, but its
alignment with FID is weak (`0.265` overall Spearman and `0.029` on
FashionMNIST), and every selected path has a larger class-entropy gap.

This closes the current spectral-preconditioning method line: do not retune
gamma, bases, band moments, or checkpoint schedules, and do not move it to RAE.
The supported contribution is the negative mechanism result that isospectral
basis orientation and path-dependent semantic transport, rather than spectrum
alone, govern endpoint behavior.  The prospective report is
[`ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md`](ROLLOUT_CHECKPOINT_SELECTION_RESULTS_ZH.md).

The group-structure line remains separate.  If it is resumed, only held-out
generator-power and D4-relation tests can justify a group claim; decoder
reconstruction or independently fitted per-transform maps cannot.  Do not
continue decoder inverse-adapter reconstruction training as the primary method.

The speed line is also closed for now: 25 steps is available when a roughly
`2x` speedup is worth a measured 5k quality loss; 16-step and late-dense
schedules are not justified.
