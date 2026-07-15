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

Data and checkpoints stay outside the repository:

```bash
python experiments/run_rae_spectral_tiny.py --mode preflight
python experiments/run_rae_spectral_tiny.py --mode smoke --device 3
python experiments/run_rae_spectral_tiny.py --mode screen --device 3
python experiments/evaluate_rae_spectral_tiny.py --device cuda:3
python experiments/evaluate_rae_spectral_generation.py --mode all --devices 0,1,2,3 --processes 4
```

The completed screen failed its preregistered 5% KID threshold, so it should not
be extended by tuning gamma or adding seeds. It remains evidence about
objective/capacity allocation, not a claim of pure
preconditioning or parameter-space acceleration. The source checkpoint lacks
the old RNG and dataloader cursor, so each pair uses a new deterministic stream
from the same full-state checkpoint rather than pretending to continue the old
stream exactly.
