# RAE experiments

This directory contains the research extensions added on top of EQ-VAE. They
are intentionally separated into diagnostics, adapter experiments, and
stage-2 generation orchestration.

## Current mainline: RAEv2 continuation

The active experiment starts from the official RAEv2 DINOv3-L-K7 full training
state and compares three strictly paired branches:

1. the untouched official EMA model;
2. a Flow-only continuation control;
3. an LPL continuation with the same data, optimizer, scheduler, noise, and
   initialization stream as Flow.

The pilot preserves the official ImageNet global batch of 1,024 through
gradient accumulation and first stops at 50 Flow steps. This processes 51,200
images, so step count is not compared to earlier reduced-batch trials.
Checkpoints are retained every 10 steps so a non-monotonic curve cannot be
hidden by endpoint selection. The LPL branch starts only when Flow continuation
itself remains healthy. The implementation and audit protocol are in
`prepare_raev2_imagenet_index.py`, `train_raev2_strict_lpl.py`,
`sample_raev2_threeway.py`,
`raev2_training_core.py`, and
[`RAEV2_LPL_STRICT_CONTINUATION_ZH.md`](../docs/RAEV2_LPL_STRICT_CONTINUATION_ZH.md).

All sections below are a research archive. They preserve successful and failed
attempts, preregistered gates, and mechanism diagnostics; the presence of code
does not imply that its hypothesis passed. Large datasets, checkpoints,
samples, and tables remain outside Git under `/data/shared` or
`$HOME/data/eqvae`.

## Noise-resolved latent responsibility

- `imagenette_noise_responsibility.py` trains the preregistered Imagenette-64
  conditional flow with an always-available latent, 10% in-distribution null
  dropout, three bottleneck capacities, paired shuffle controls, and radial
  frequency profiles. It contains no high/low-noise condition gate.
- `run_imagenette_responsibility_sweep.py` schedules independent capacity/seed
  runs across four GPUs and stops the sweep when any worker fails.
- `summarize_imagenette_noise_responsibility.py` verifies data-stream and shared
  initialization hashes, computes the five preregistered gates, and renders the
  final figures.

The five-seed experiment supports a real capacity-dependent responsibility
curve but fails the preregistered incremental quality-prediction gate, so no
latent prior or end-to-end model is trained. Protocol and full Chinese results:
`docs/IMAGENETTE_NOISE_RESPONSIBILITY_PREREG_ZH.md` and
`docs/IMAGENETTE_NOISE_RESPONSIBILITY_RESULTS_ZH.md`. Large outputs live under
`$HOME/data/eqvae/imagenette_noise_responsibility_formal/`.

```bash
PYTHONPATH=. python experiments/run_imagenette_responsibility_sweep.py \
  --seeds 0,1,2,3,4 --devices 0,1,2,3
PYTHONPATH=. python experiments/summarize_imagenette_noise_responsibility.py
```

## Transport-risk and training-path audits

- `latent_transport_paths.py` implements Base, Gaussian-straight,
  matched-chord, and strict pushforward conditional paths.
- `audit_latent_transport_protocol.py` performs the real checkpoint phase-0
  cycle/decoder/JVP audit.
- `audit_latent_transport_compatibility.py` runs the four-GPU phase-2
  source-prior/path-curvature audit. Its large tables live under
  `$HOME/data/eqvae/artifacts/latent_transport_audit`; projected VIV and local
  velocity ambiguity are explicitly labeled as diagnostics rather than
  generation metrics.
- `latent_transport_four_path_toy.py` is the preregistered five-seed causal
  comparison of Base, Gaussian-straight, matched-chord, and strict pushforward
  paths on an analytic 2D mixture.
- `audit_four_path_toy_conjugacy.py` verifies that an exactly conjugated Base
  vector field recovers the Base endpoint and localizes the negative toy result
  to finite model-class/loss non-invariance rather than a path implementation
  error. The toy failed its recovery gate, so this line does not proceed to
  small images or RAE training.

- `transport_risk_atlas.py` builds the leakage-audited baseline-only risk table
  from completed isospectral studies.
- `small_image_checkpoint_resample.py` re-evaluates frozen checkpoints over
  larger test sets and independent rollout seeds.
- `small_image_signed_leverage.py` compares candidate update directions with a
  train-only differentiable-rollout endpoint gradient.
- `small_image_training_path_probe.py` exactly replays paired training, verifies
  final checkpoint hashes, and records train-only endpoint moments along the
  path.
- `small_image_rollout_checkpoint_selection.py` runs the preregistered
  unseen-seed checkpoint-selection test.
- `summarize_rollout_checkpoint_selection.py` combines both datasets, audits
  leakage, and writes the final H1/H2 and proxy/semantic diagnostics.

Generated tables are stored under `$HOME/data/eqvae/experiments/`; these
scripts do not write repository-local `outputs/` or `artifacts/` directories.

## Spectral theory and time-order audits

- `isospectral_alignment_toy.py` proves on exact least squares that equal
  output-weight spectra need not produce equal `J^T W J` geometry.
- `small_image_dct_sign_scramble.py` preserves DCT power while destroying phase
  and measures the resulting semantic collapse.
- `small_image_seed_factorial.py`, `small_image_stream_factorial.py`, and
  `small_image_bridge_factorial.py` split data, initialization, minibatch,
  Gaussian bridge, and time randomness.
- `small_image_time_sequence_audit.py` replays the exact RNG call order and
  audits time histograms and time-dependent spectral-weight exposure without
  training.
- `small_image_time_order_study.py` compares IID, fixed-multiset permutation,
  and stratified schedules under fixed data/minibatch/noise streams.
- `small_image_time_seed_sweep.py` measures IID or stratified schedule effects
  over 12 time seeds on MNIST or FashionMNIST.
- `small_image_time_permutation_sweep.py` keeps every sampled time value fixed
  and varies only the order of training steps.

Formal commands for the completed low-cost studies:

```bash
PYTHONPATH=. python experiments/small_image_time_sequence_audit.py --device cuda:0
PYTHONPATH=. python experiments/small_image_time_order_study.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
PYTHONPATH=. python experiments/small_image_time_seed_sweep.py \
  --dataset mnist --sampling iid --devices cuda:0,cuda:1,cuda:2,cuda:3
PYTHONPATH=. python experiments/small_image_time_seed_sweep.py \
  --dataset fashion_mnist --sampling iid --devices cuda:0,cuda:1,cuda:2,cuda:3
PYTHONPATH=. python experiments/small_image_time_seed_sweep.py \
  --dataset mnist --sampling stratified --devices cuda:0,cuda:1,cuda:2,cuda:3
PYTHONPATH=. python experiments/small_image_time_permutation_sweep.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
```

Interpretation, primary literature, exact result locations, and the remaining
gap are recorded in
[`SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md`](../docs/SPECTRAL_THEORY_EXPLANATION_AND_GAPS_ZH.md).

## Storage contract

Read datasets and shared evaluation references from `/data/shared`. Store all
user-owned assets under `$HOME/data/eqvae` (override with
`EQVAE_DATA_ROOT`). The repository exposes `artifacts/`,
`pretrained_models/`, and `external/RAE/models/` as compatibility symlinks to
that location, so existing relative paths remain valid.

| Location | Purpose |
|---|---|
| `/data/shared/imagenet-1k` | ImageNet-1K parquet dataset |
| `/data/shared/caltech101` | lightweight diagnostic dataset |
| `/data/shared/adm_refs` | immutable ADM-FID reference files |
| `$EQVAE_DATA_ROOT/artifacts` | checkpoints, diagnostics, and local metrics |
| `$EQVAE_DATA_ROOT/models` | RAE and VAE weights |
| `$EQVAE_DATA_ROOT/stage2_training` | RAE DiT checkpoints and configs |
| `$EQVAE_DATA_ROOT/stage2_samples` | generated images, NPZs, grids, and ADM metrics |

Shell entry points source `common_paths.sh`; use `EQVAE_SHARED_DATA_ROOT` only
when a machine keeps shared datasets elsewhere.

## Diagnostic code

- `rae_layerwise_imagenet_study.py`: batched, layerwise direct spatial
  equivariance error for frozen RAE-DINOv2, RAE-MAE, and RAE-SigLIP2.
- `rae_encoder_decoder_atlas.py` and `run_rae_encoder_decoder_atlas.py`:
  basis-invariant encoder-decoder cross-layer CKA with mismatched-image,
  held-out split, latent-perturbation, and generated-cycle controls.
- `run_rae_encoder_decoder_probe.py`: fits five clean calibration-only reverse
  hierarchy projectors and evaluates them on held-out and generated latents.
  The completed result is documented in
  [`RAE_ENCODER_DECODER_ATLAS_RESULTS_ZH.md`](../docs/RAE_ENCODER_DECODER_ATLAS_RESULTS_ZH.md).
- `latent_smoothness_proxies.py`: no-training proxy measurements for latent
  smoothness and decoder sensitivity.
- `rae_reconstruction_rfid.py`: ImageNet reconstruction rFID for a frozen RAE.
- `evaluate_latent_adapter_transform_rfid.py`: strict transformed-image rFID
  comparison between direct `P(z)` and the adapted latent path.

## Adapter experiments

- `latent_equiv_adapter.py`: trains an invertible encoder-side latent adapter
  `A`. The public latent is `y = A(E(x))`; decoding applies `A^{-1}` before a
  frozen RAE decoder. The intended transform set is `flip_h`, `flip_v`, and
  `rot180`, where the last operation tests a simple composition relation.
- `train_decoder_inverse_adapter.py`: keeps the RAE encoder and base decoder
  frozen by default and optimizes only a separate decoder-side inverse adapter.
  The base decoder is trainable only with the explicit `--train-base-decoder`
  flag.
- `decoder_adapted_rae.py`: stage-1 wrapper combining an encoder adapter and a
  separately checkpointed decoder-side inverse adapter.

## Stage-2 generation

`run_*gfid.sh`, `watch_adapter_finetune_samples.sh`, and
`run_adapter_finetune_final50k.sh` orchestrate RAE DiT training, sampling, and
ADM-FID evaluation. They are experiment runners, not library APIs; each writes
its logs and results below `$EQVAE_DATA_ROOT`.

The current evidence, interpretation boundaries, and recommended next
experiments are tracked in [`docs/RESEARCH_STATUS.md`](../docs/RESEARCH_STATUS.md).

## Spectral tiny screen

The frequency-weighting line is intentionally gated by one paired microtraining
experiment rather than another large run. Its fixed protocol is implemented by:

- `rae_spectral_direction_loss.py`: eight-band DCT direction loss with fixed
  train-only statistics, bounded weights, and coefficient-weighted mean-one
  normalization at every time.
- `train_rae_spectral_tiny.py`: fp32 deterministic branch trainer that restores
  model, EMA, optimizer, and scheduler from the local step-5000 checkpoint.
- `run_rae_spectral_tiny.py`: preregisters and launches three paired seeds for
  `gamma=0` versus `gamma=0.5`.
- `evaluate_rae_spectral_tiny.py`: fixed held-out latent/noise diagnostics at
  branch updates 500, 1k, 2k, and 5k.
- `evaluate_rae_spectral_generation.py`: fixed-noise, equal-label 5k sampling,
  KID, and FID proxy evaluation. Metrics read the exact 5k-sample NPZ rather
  than the sampler's batch-padded PNG directory; the TensorFlow ADM evaluator
  is optional via `--with-adm`.
- [`rae_spectral_tiny_screen.ipynb`](../notebooks/rae_spectral_tiny_screen.ipynb):
  reader-facing plots and the preregistered pass/fail rule.
- `rae_teacher_rollout_gap.py`: separates paired teacher-forced clean estimates
  from distribution-only ODE rollout diagnostics, including decoder hidden
  states, Inception features, per-band calibration, marginal energy drift,
  vector-field secants, and stepwise curvature.
- `run_rae_teacher_rollout_gap.py`: runs the six endpoint EMA branches across
  four GPUs and aggregates tables under `$HOME/data/eqvae`.
- `rae_step_schedule_probe.py`: compares reduced Euler time grids with the
  paired official 50-step numerical endpoint.  It screens speed candidates but
  does not replace 5k FID.
- `rae_vector_field_switch_probe.py`: hard-switches the paired baseline and
  spectral EMA vector fields across high-, middle-, and low-noise intervals to
  localize where teacher improvements turn into rollout contraction.
- `run_rae_vector_field_switch_probe.py`: runs the switch probe for all three
  paired seeds and aggregates the local tables outside Git.
- `mnist_transport_mechanism.py`: low-cost step-convergence, radial-energy
  calibration, divergence, and time/frequency vector-field interventions for
  the paired MNIST velocity fields.
- `rae_frequency_time_switch_probe.py`: splices paired RAE EMA vector fields by
  both time window and radial DCT output band, without training or decoding.
- `rae_band_transport_probe.py`: evaluates both paired RAE vector fields on
  shared teacher, baseline-rollout, and partial-rollout states using the exact
  band second-moment drift `2 E[<z_b,v_b>]`.
- [`rae_teacher_rollout_gap.ipynb`](../notebooks/rae_teacher_rollout_gap.ipynb):
  Chinese reader-facing analysis of the teacher-forcing/transport gap and the
  sampling schedule probe.
- [`TEACHER_ROLLOUT_MECHANISM_ZH.md`](../docs/TEACHER_ROLLOUT_MECHANISM_ZH.md):
  final Chinese evidence report separating on-path marginal-drift mismatch
  from middle-stage off-path generalization failure.

Data and checkpoints stay outside the repository:

```bash
python experiments/run_rae_spectral_tiny.py --mode preflight
python experiments/run_rae_spectral_tiny.py --mode smoke --device 3
python experiments/run_rae_spectral_tiny.py --mode screen --device 3
python experiments/evaluate_rae_spectral_tiny.py --device cuda:3
python experiments/evaluate_rae_spectral_generation.py --mode all --devices 0,1,2,3 --processes 4
python experiments/run_rae_teacher_rollout_gap.py --mode all --devices 0,1,2,3 --count 64 --overwrite
python experiments/rae_step_schedule_probe.py \
  --branch "$HOME/data/eqvae/experiments/rae_spectral_tiny/seed3407_baseline_from_s5000" \
  --device cuda:0 --count 64
python experiments/run_rae_vector_field_switch_probe.py \
  --mode all --devices 0,1,2 --count 64
python experiments/rae_frequency_time_switch_probe.py \
  --baseline "$HOME/data/eqvae/experiments/rae_spectral_tiny/seed3407_baseline_from_s5000" \
  --partial "$HOME/data/eqvae/experiments/rae_spectral_tiny/seed3407_partial_from_s5000" \
  --output "$HOME/data/eqvae/experiments/rae_spectral_tiny/frequency_time_switch/seed3407" \
  --device cuda:0 --count 64 --batch-size 8
python experiments/rae_band_transport_probe.py \
  --baseline "$HOME/data/eqvae/experiments/rae_spectral_tiny/seed3407_baseline_from_s5000" \
  --partial "$HOME/data/eqvae/experiments/rae_spectral_tiny/seed3407_partial_from_s5000" \
  --output "$HOME/data/eqvae/experiments/rae_spectral_tiny/band_transport/seed3407" \
  --device cuda:0 --count 64 --batch-size 8
python experiments/evaluate_rae_spectral_generation.py \
  --mode all --branch-name seed3407_baseline_from_s5000 --steps 25 \
  --sample-count 5000 --devices 0,1 --processes 2 --per-process-batch 8
```

The completed screen failed its preregistered 5% KID threshold, so it should not
be extended by tuning gamma or adding seeds. It remains evidence about
objective/capacity allocation, not a claim of pure
preconditioning or parameter-space acceleration. The source checkpoint lacks
the old RNG and dataloader cursor, so each pair uses a new deterministic stream
from the same full-state checkpoint rather than pretending to continue the old
stream exactly.

## Imagenette latent-prior trade-off

The frozen Imagenette-64 encoder/decoder runs can be closed with one
capacity-independent 256D latent rectified-flow prior:

```bash
python experiments/run_imagenette_latent_prior_tradeoff_sweep.py \
  --output-root ~/data/eqvae/imagenette_latent_prior_tradeoff \
  --capacities 16,64,256 --seeds 0,1,2,3,4 --devices 0,1,2,3

python experiments/summarize_imagenette_latent_prior_tradeoff.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff
```

The preregistration and final Chinese report are
`docs/IMAGENETTE_LATENT_PRIOR_TRADEOFF_PREREG_ZH.md` and
`docs/IMAGENETTE_LATENT_PRIOR_TRADEOFF_RESULTS_ZH.md`. Formal artifacts stay
outside the repository. The NFE200 and semantic-subspace audits are diagnostic
only and do not alter the preregistered gates.

The follow-up decoder mechanism audit can be reproduced without retraining any
model:

```bash
python experiments/run_imagenette_decoder_amplification_sweep.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
python experiments/summarize_imagenette_decoder_amplification.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff
python experiments/run_imagenette_decoder_witness_gap_sweep.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff \
  --devices cuda:0,cuda:1,cuda:2,cuda:3
python experiments/summarize_imagenette_decoder_witness_gap.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff
python experiments/analyze_imagenette_decoder_semantic_reweighting.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff
python experiments/analyze_imagenette_latent_spectrum.py \
  --root ~/data/eqvae/imagenette_latent_prior_tradeoff --device cuda:0
```

The equal-angle intervention rejects special alignment between prior errors and
locally sensitive decoder directions. The post-hoc witness and cross-fitted
semantic reweighting audits instead localize most of the 256d decoded gap to
fine-mode/covariance mismatch. Exact evidence and interpretation boundaries are
recorded in
[`IMAGENETTE_DECODER_AMPLIFICATION_RESULTS_ZH.md`](../docs/IMAGENETTE_DECODER_AMPLIFICATION_RESULTS_ZH.md).
