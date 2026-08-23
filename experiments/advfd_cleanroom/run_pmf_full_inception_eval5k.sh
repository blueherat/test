#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_prefix_v1
LOG="$ROOT/static_prefix1000_eval5k.log"

while [[ ! -f "$ROOT/static/evaluation.json" ]]; do
  sleep 5
done

COMMON=(
  --stage evaluate
  --output-root "$ROOT"
  --device cuda:3
  --batch-size 32
  --feature-dim 2048
  --warmstart-samples 50000
  --eval-samples 5000
  --adaptive-eval-samples 512
  --num-workers 4
  --seed 260823
)

exec > >(tee "$LOG") 2>&1

python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --variant base --evaluation-tag paired5k "${COMMON[@]}"

for step in 250 500 750 1000; do
  checkpoint=$(printf '%s/static/checkpoint_step%06d.pt' "$ROOT" "$step")
  tag=$(printf 'step%06d_paired5k' "$step")
  python -m experiments.advfd_cleanroom.run_pmf_pilot \
    --variant static \
    --checkpoint "$checkpoint" \
    --evaluation-tag "$tag" \
    "${COMMON[@]}"
done
