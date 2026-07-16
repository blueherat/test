# RAE experiments

This directory contains the research extensions added on top of EQ-VAE. They
are intentionally separated into diagnostics, adapter experiments, and
stage-2 generation orchestration.

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
