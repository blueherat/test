# RAE Geometry Research Status

## Scope

The working research question is whether frozen RAE tokenizers contain a
usable geometric response that can improve latent-space generation. This is
not yet a claim that RAE-DINOv2 has a global group representation.

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

## Current repository entry points

- `notebooks/latent_playground.ipynb`: reconstruction and latent editing
  interface. Use it for decoder closure and visual inspection.
- `notebooks/dinov2_token_diagnostic.ipynb`: encoder-side token correspondence,
  orbit alignment, Procrustes, and group-consistency diagnostics.
- `notebooks/rae_layerwise_playground.ipynb`: simple per-image and batched
  layerwise visualizations for DINOv2, MAE, and SigLIP2.
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

## Recommended next experiment

Do not continue decoder inverse-adapter reconstruction training as the primary
method. First finish the held-out generator-only group test: fit `C_90` (and
optionally `C_flip_h`) on training images, predict `rot180`, `rot270`, and D4
relations on test images, and compare functional errors with independently
fitted maps. If this fails, the method direction should be a light,
position-aware adapter or an intermediate-layer tokenizer, rather than forcing
global group action in the final RAE latent.

For the spectral line, do not add more toy variants, tune `gamma`, splice the
existing checkpoints by time, add seeds, or run 50k FID. The useful retained
result is the measured predictability versus transport-covariance mismatch;
fixed inverse-standard-deviation DCT weighting is not a sufficiently effective
solution.  The smallest quality follow-up is a paired tiny experiment that adds
train-only, batch-level per-band marginal matching after one or two unrolled
Euler steps.  It must report both teacher velocity metrics and rollout
energy/covariance metrics before generation.  The speed line is closed for now:
25 steps is available when a roughly `2x` speedup is worth a measured 5k
quality loss; 16-step and late-dense schedules are not justified.
