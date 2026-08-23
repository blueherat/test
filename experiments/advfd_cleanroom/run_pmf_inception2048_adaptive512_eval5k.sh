#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=${1:?usage: $0 OUTPUT_ROOT [DEVICE]}
DEVICE=${2:-cuda:3}
WARMSTART=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_prefix_v1/warmstart.pt
LOG="$ROOT/paired_eval5k.log"

COMMON=(
  --stage evaluate
  --output-root "$ROOT"
  --warmstart-file "$WARMSTART"
  --device "$DEVICE"
  --batch-size 32
  --feature-dim 2048
  --adaptive-feature-dim 512
  --warmstart-samples 50000
  --eval-samples 5000
  --balanced-eval-labels
  --per-sample-eval-noise
  --eval-noise-seed 42
  --quantize-eval-images
  --adaptive-eval-samples 5000
  --num-workers 4
  --seed 260823
  --no-amp
)

exec > >(tee "$LOG") 2>&1

PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --variant base \
  --evaluation-tag paired5k \
  "${COMMON[@]}"

for step in 1000 2000 3000 4000 5000; do
  checkpoint=$(printf '%s/static/checkpoint_step%06d.pt' "$ROOT" "$step")
  tag=$(printf 'step%06d_paired5k' "$step")
  PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.run_pmf_pilot \
    --variant static \
    --checkpoint "$checkpoint" \
    --evaluation-tag "$tag" \
    "${COMMON[@]}"
done

for step in 1000 2000 3000 4000 5000; do
  checkpoint=$(printf '%s/real/checkpoint_step%06d.pt' "$ROOT" "$step")
  tag=$(printf 'step%06d_paired5k' "$step")
  PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.run_pmf_pilot \
    --variant real \
    --checkpoint "$checkpoint" \
    --evaluation-tag "$tag" \
    "${COMMON[@]}"
done

PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.summarize_pmf_prefix \
  --output-root "$ROOT"
