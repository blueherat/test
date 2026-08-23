#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_prefix_v1
WARMSTART="$ROOT/warmstart.pt"
LOG="$ROOT/static_prefix1000.log"

while [[ ! -f "$WARMSTART" ]]; do
  sleep 5
done

PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --stage train \
  --variant static \
  --output-root "$ROOT" \
  --device cuda:3 \
  --batch-size 32 \
  --feature-dim 2048 \
  --warmstart-samples 50000 \
  --eval-samples 5000 \
  --num-workers 4 \
  --steps 1000 \
  --generator-lr 1e-6 \
  --lr-warmup 6250 \
  --schedule-total-steps 125000 \
  --adaptive-start 1000 \
  --adaptive-warmup 4000 \
  --save-every 250 \
  --log-every 25 \
  --seed 260823 \
  2>&1 | tee "$LOG"
