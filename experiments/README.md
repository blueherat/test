# Research experiments

## Paused mainline snapshot: model-relative bad/good trajectory metrics

> Paused on 2026-08-28 by user request. No command in this section is currently
> authorized for automatic continuation. The source/lock/data retention map is
> in
> [`DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md`](../docs/DIT_BAD_GOOD_PAUSE_ARCHIVE_2026-08-28_ZH.md),
> and corrected final/unfinished statuses are in
> [`DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`](../docs/DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md).

The paused direction was a frozen-model, inference-only search for a quantity
that distinguishes visually obvious, model-relative bad cases from ordinary
outputs before generation finishes. Predictable model outputs, realized
transition surprises, reverse-kernel uncertainty, cross-scale disagreements,
spatial/edge structure, and sparse combinations are evaluated under blind
labels and discovery/confirmation separation. The protocol and current
evidence are in
[`BAD_GOOD_TRAJECTORY_METRICS_ZH.md`](../docs/BAD_GOOD_TRAJECTORY_METRICS_ZH.md).

The older cross-scale path-evidence tools below are retained as reproducible
history and as one metric family. Their fixed higher-noise alternative failed
its prospective cross-prefix test and does not authorize rollback.

The later DiT v2.2 repairability pilot is also complete. It retains every one
of four suffix attempts at two rollback depths for 8 joint `E+B` and 8 exact-
schedule, class, continuous-B-matched `B`-only paths. Its frozen visual-
opportunity guard failed (`3/8` versus `6/8` paths), and the protocol-literal
matched repair differences contain four joint-worse, zero joint-better, and
four tied pairs. Consequently `E` is no longer an active quality or rollback
candidate; B remains a weak blur-phenotype diagnostic only. The first frozen
analyzer's fresh-preference repair definition is preserved for audit but is
superseded for success claims by:

```bash
python experiments/analyze_dit_v22_repairability_protocol_conformance.py --self-test
python experiments/analyze_dit_v22_repairability_protocol_conformance.py
```

The first prefix-conditional branch-consensus form is now also closed. Its
frozen horizon-10 medoid selected 5/18 strict repairs (`27.8%`), below the
same-four-scout uniform-policy value of `37.5%`; higher attempt-0 outlier ratio
had opportunity AUC `0.286`. A reveal-after-freeze reverse audit found the
horizon-10 max-nonconformity rank at 10/18 (`55.6%`), but this was post-hoc,
class-concentrated, horizon-unstable, and slightly worse on the small safety
readouts. It motivates exactly one narrow new-suffix prospective test, not a
claim that outliers are generally better.

The V1.2 lock freezes 128 step149 prefixes, four symmetric baseline-P scouts,
the exact shared-prefix-normalized multi-scale distance, horizon 10, max-
nonconformity direction, anonymous tie breaks, and 512 fresh RNG streams before
any new suffix exists. Its internal extractor reads only `internal_timestep`
and `target_pred_xstart`, seals every selection before endpoint judging, and
never consumes labels, PNG pixels, B/E/O, FID, or embeddings:

```bash
python experiments/freeze_dit_v22_transient_escape_prospective.py
CUDA_VISIBLE_DEVICES=0 python \
  experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2/sources/run_dit_v22_transient_escape_prospective_shard.py \
  --lock experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2 \
  --shard-index 0 --shard-count 2
CUDA_VISIBLE_DEVICES=1 python \
  experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2/sources/run_dit_v22_transient_escape_prospective_shard.py \
  --lock experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2 \
  --shard-index 1 --shard-count 2
python \
  experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2/sources/extract_dit_v22_transient_escape_internal.py \
  --lock experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2
```

The complete method ledger and claim boundaries are documented in
[`DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md`](../docs/DIT_BAD_GOOD_METHOD_LEDGER_2026-08-28_ZH.md).
The detailed E/B repairability record remains in
[`DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md`](../docs/DIT_V22_INTERNAL_SIGNAL_REASSESSMENT_ZH.md).

The first reproducible entry points are:

```bash
bash experiments/download_cross_scale_baselines.sh
torchrun --standalone --nproc_per_node=4 \
  experiments/reproduce_cfg_rejection_edm2.py --protocol smoke
python experiments/summarize_cfg_rejection_edm2.py \
  --run-dir "$EQVAE_DATA_ROOT/cross_scale_evidence/cfg_rejection_edm2/smoke_official_v3"
python experiments/score_cfg_rejection_edm2_imagenet.py \
  --run-dir "$EQVAE_DATA_ROOT/cross_scale_evidence/cfg_rejection_edm2/paper_10k_official_v3"
python experiments/reproduce_adm64_guided.py --self-test
python experiments/reproduce_adm64_guided.py \
  --protocol smoke --batch 2 --device cuda:0
python experiments/adm64_path_evidence.py --self-test
CUDA_VISIBLE_DEVICES=0 python experiments/observe_adm64_cross_scale_evidence.py \
  --protocol smoke \
  --baseline-dir "$EQVAE_DATA_ROOT/cross_scale_evidence/adm64_guided/smoke" \
  --output-dir "$EQVAE_DATA_ROOT/cross_scale_evidence/adm64_cross_scale_evidence/smoke_k020_v2" \
  --max-conditional-kl 0.2 --alpha 0.05 --batch 3 --device cuda:0
python experiments/summarize_adm64_cross_scale_evidence.py \
  --run-dir "$EQVAE_DATA_ROOT/cross_scale_evidence/adm64_cross_scale_evidence/smoke_k020_v2" \
  --output-dir "$EQVAE_DATA_ROOT/cross_scale_evidence/adm64_cross_scale_evidence_summary/smoke_k020_v2" \
  --cap-grid 0.01,0.05,0.2,1.0
python experiments/reproduce_fkc_edm2.py --self-test
CUDA_VISIBLE_DEVICES=0 python experiments/reproduce_fkc_edm2.py \
  --mode cfg --seeds 0-7 --resample-seed 20260826 \
  --outdir "$EQVAE_DATA_ROOT/cross_scale_evidence/fkc_edm2/strict_cfg_seed0_7"
CUDA_VISIBLE_DEVICES=0 python experiments/reproduce_fkc_edm2.py \
  --mode fkc --seeds 0-7 --resample-seed 20260826 \
  --outdir "$EQVAE_DATA_ROOT/cross_scale_evidence/fkc_edm2/strict_fkc_seed0_7"
```

The reproduction runner stores both the paper-equation ASD and the different
statistic actually used by the released ImageNet notebook. Its deterministic
Heun trajectories are not represented as Gaussian likelihood-ratio paths.
Visual review must use full-resolution images and separate semantic class
agreement from structural/image artifacts; the mandatory rubric and the
corrected smoke audit are in
[`CFG_REJECTION_VISUAL_AUDIT_ZH.md`](../docs/CFG_REJECTION_VISUAL_AUDIT_ZH.md).
`build_blind_bad_case_audit.py` creates per-reviewer full-resolution anonymous
packets; `analyze_cfg_rejection_pilot_statistics.py` reports class-stratified,
seed-clustered secondary classifier summaries without confirmatory claims.

The observe-only ADM runner never rejects, rolls back, or changes P. Its
schema-v2 signals retain both raw and tempered sufficient statistics for an
independent LR audit. The primary discovery cap is fixed at 0.2; cap
reconstructions from the raw records are exploratory unless their grid and
weights were frozen before endpoint labels were viewed.

`reproduce_adm64_guided.py` is the stochastic baseline-only runner. It uses
the official ImageNet-64 ADM and noisy classifier, 250-step ancestral DDPM,
classifier scale 1, and does not contain Q or a likelihood ratio. The same
sample seed deliberately reuses its full Gaussian innovation stream across
classes; neural evaluations remain singleton so changing logical `--batch`
does not change a path. `adm64_path_evidence.py` contains the independently
tested normalized-heat coordinate transform, KL tempering, same-covariance
Gaussian likelihood ratio, log-space mixture, and Ville crossing primitives.

`reproduce_fkc_edm2.py` is a fail-closed wrapper around the frozen released
FKC image code. It retains the upstream random-class behavior, extra unused
class draw, 63 systematic-resampling events, and slot-0-only save policy. The
otherwise-uncontrolled CPU resampling RNG must be supplied explicitly through
`--resample-seed`; completed outputs are content-hash validated and immutable.
Its ordinary-CFG mode retains the upstream default batch 32, whereas FKC mode
enforces the released eight-particle batch.

The post-hoc DiT suffix discovery audit is intentionally split into three
tools so visual labels, trajectory dynamics, and endpoint proxies cannot be
silently conflated:

```bash
python experiments/analyze_dit_suffix_trajectory_quality.py --self-test-only
python experiments/visualize_dit_suffix_predxstart.py --self-test
python experiments/score_dit_suffix_endpoint_quality_proxies.py --help
python experiments/replay_dit_suffix_cross_scale_diagnostics.py --self-test
python experiments/summarize_dit_suffix_cross_scale_replay.py --self-test
python experiments/visualize_dit_suffix_cross_scale_replay.py --self-test
python experiments/run_dit_t60_within_prefix_validation_pool.py --self-test
python experiments/build_dit_t60_within_prefix_blind_pack.py --self-test
python experiments/summarize_dit_t60_within_prefix_validation.py --self-test
python experiments/run_dit_t60_cross_prefix_mixture_validation_pool.py --self-test
python experiments/build_dit_t60_cross_prefix_blind_pack.py --self-test
python experiments/lock_dit_t60_cross_prefix_consensus.py self-test
python experiments/summarize_dit_t60_cross_prefix_mixture_validation.py --self-test
```

`analyze_dit_suffix_trajectory_quality.py` validates and reconstructs every
saved Gaussian transition, excludes the shared suffix-entry row from
post-divergence state summaries, and reports tie-aware within-checkpoint ranks.
`visualize_dit_suffix_predxstart.py` only decodes selected frozen
`pred_xstart` frames and performs no scoring. The endpoint proxy script uses
two ImageNet classifiers for class fidelity only and clearly marks its fixed
tail/hind texture boxes as post-hoc and pose-specific. The quality labels are
in `annotations/dit_imagenet256_seed2_suffix_quality_review_v2.json`; they
separate endpoint quality, prefix preservation, and tail naturalness. All
three analyses are discovery-only and require a new output directory rather
than overwriting a completed bundle.

`replay_dit_suffix_cross_scale_diagnostics.py` performs a hash-validated,
observe-only shifted-DiT replay at fixed normalized-heat offsets. It records
the saved-P innovation, predictable whitened mean shift, conditional KL, and
exact same-covariance Gaussian LR for one global and sixteen fixed latent
tiles; it never changes a sample. `summarize_dit_suffix_cross_scale_replay.py`
independently reconstructs the one-sided, path-level fixed-sign, and uniform
change-point mixtures from those raw sufficient statistics. The visualization
script consumes only a fully validated raw bundle and includes an explicit
latent-to-image grid audit. That audit shows the post-hoc t60 `tile_12`
component is nominally lower-left background, not the tail or malformed hind
leg, so it may only be tested as a frozen black-box candidate. The tail rubric
separately evaluates attachment, taper, feathered hair flow, and distal-tip
continuity; a merely identifiable broad, blunt, or filament-tipped structure
is not automatically natural. No discovery ranking is an intervention claim,
and the observed log-e values are far below a useful Ville threshold.

`run_dit_t60_within_prefix_validation_pool.py` is the first prospective
within-prefix falsification run. Its self-hashed protocol fixes the exact
class-207/seed-2 `x60`, 32 branch-local innovation streams in four 8-branch
shards, `delta_nu=0.25`, the post-hoc black-box `+theta/tile_12` component,
total `K=0.5`, and the `log(5)` alarm before GPU execution. It generates only
unchanged baseline-P suffixes; evidence is private and observation-only. The
four completed shards must all be retained, endpoint labels must be locked and
hashed using an evidence-free blind-review pack, and only then may scores be
unsealed once. This pool tests reproducibility conditional on one fixed prefix,
not new-seed, cross-prefix, class-level, or general quality detection. Tail
identity/naturalness, hind-limb topology, and overall structural failure are
three separate review fields.

`build_dit_t60_within_prefix_blind_pack.py` validates all four shards while
suppressing private validator output, strips PNG metadata, and publishes only
32 blind-ID images, a contact sheet, the visual rubric, and an empty template.
`summarize_dit_t60_within_prefix_validation.py` refuses to touch a shard until
the exact 32-row consensus annotation and all evidence-unseen declarations are
validated and copied into a byte-identical staged lock. It then emits only the
preregistered aggregate tables in a closed no-overwrite bundle; no per-branch
alarm, evidence, rank, image, or trace is exported.

The completed one-time readout is a negative result for the frozen primary
candidate: `+theta/tile_12/log(5)` produced 0/32 alarms, including 0/2 blindly
adjudicated clear hind-limb failures and 0/3 overall clear structural bad
images. The primary run is also event-limited because it contains fewer than
three clear failures. The fixed 34-path signed spatial mixture remains a
strictly descriptive lead: running-maximum AUC is 0.944 for the 2 clear versus
18 not-clear primary labels, while terminal AUC is 0.611. This may motivate a
new-prefix preregistration of transient mixed evidence, but it does not permit
threshold or component tuning on these 32 images and does not authorize a
rollback experiment.

`run_dit_t60_cross_prefix_mixture_validation_pool.py` is the prospective
cross-prefix follow-up. Its pre-GPU self-hashed protocol fixes 64 independent
class-207 prefixes, eight shards, the same `delta_nu=0.25`, total
per-component `K<=0.5`, and the uniform complete-path mixture over global plus
16 tiles and both signs. It contains a literal bitwise baseline-P mirror;
evidence is computed before each innovation and never changes the P update.
`build_dit_t60_cross_prefix_blind_pack.py` publishes only randomized images
and frozen external anchors. `lock_dit_t60_cross_prefix_consensus.py` requires
two sealed independent reviews and a later evidence-blind adjudication before
creating an immutable 64-row consensus. The aggregate summarizer claims its
one allowed unseal before opening any private mapping or trajectory and emits
no per-sample join.

The formal cross-prefix result retires the fixed candidate. Consensus contains
4 clear overall bad, 59 not-clear bad, and 1 uncertain image. `E_mix>=5`
crossed on 0/64 paths (TP 0/4, FP 0/59; Fisher one-sided `p=1`), so the frozen
outcome is `frozen_threshold_failed_to_pass`; no rollback is authorized. The
within-prefix running-maximum AUC 0.944 failed to replicate at 0.441, and the
terminal AUC is 0.169. A pre-frozen secondary peak-to-terminal drawdown has AUC
0.805, but with only four positives and eleven related trajectory features it
is a new reversal hypothesis, not a successful detector. Tail identity and
naturalness remain separate: only 7/64 tails were scorable, with 1 natural,
5 odd, and 1 malformed. Any successor must first pass a matched-Q power test;
the current 34-way uniform, `K<=0.5`, threshold-5 combination was not shown to
have useful power even when one operational component is correct.

All earlier RAE geometry, transport, spectral, and continuation experiments
below are retained as an audit archive. They are not active tuning directions.

## Archived RAE experiments

This directory contains the research extensions added on top of EQ-VAE. They
are intentionally separated into diagnostics, adapter experiments, and
stage-2 generation orchestration.

## Archived mainline: RAEv2 continuation

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
