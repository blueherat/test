#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_projected_pilot_v1
LOG="$ROOT/eval5k_sweep.log"
exec > >(tee -a "$LOG") 2>&1

COMMON=(
  --stage evaluate
  --output-root "$ROOT"
  --device cuda:3
  --batch-size 16
  --feature-dim 64
  --warmstart-samples 2048
  --eval-samples 5000
  --adaptive-eval-samples 512
  --num-workers 4
  --seed 260823
)

python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --variant base --evaluation-tag paired5k "${COMMON[@]}"

for variant in static raw real; do
  for step in 40 80 120; do
    checkpoint=$(printf '%s/%s/checkpoint_step%06d.pt' "$ROOT" "$variant" "$step")
    tag=$(printf 'step%06d_paired5k' "$step")
    python -m experiments.advfd_cleanroom.run_pmf_pilot \
      --variant "$variant" \
      --checkpoint "$checkpoint" \
      --evaluation-tag "$tag" \
      "${COMMON[@]}"
  done
done
