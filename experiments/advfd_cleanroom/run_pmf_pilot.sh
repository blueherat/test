#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_projected_pilot_v1
mkdir -p "$ROOT"
exec > >(tee -a "$ROOT/pipeline.log") 2>&1

COMMON=(
  --output-root "$ROOT"
  --device cuda:3
  --batch-size 8
  --feature-dim 64
  --warmstart-samples 2048
  --eval-samples 512
  --num-workers 4
  --steps 120
  --adaptive-start 10
  --adaptive-warmup 40
  --lr-warmup 10
  --save-every 40
  --seed 260823
)

python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --stage warmstart "${COMMON[@]}"

python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --stage evaluate --variant base "${COMMON[@]}"

for variant in static raw real; do
  python -m experiments.advfd_cleanroom.run_pmf_pilot \
    --stage train --variant "$variant" "${COMMON[@]}"
done
