#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_adaptive512_prefix5000_v1
WARMSTART=/data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_pmf_inception2048_prefix_v1/warmstart.pt
DEVICE=${DEVICE:-cuda:3}

mkdir -p "$ROOT"

COMMON=(
  --stage train
  --output-root "$ROOT"
  --warmstart-file "$WARMSTART"
  --device "$DEVICE"
  --batch-size 32
  --feature-dim 2048
  --adaptive-feature-dim 512
  --warmstart-samples 50000
  --eval-samples 5000
  --adaptive-eval-samples 512
  --num-workers 4
  --steps 5000
  --generator-lr 1e-6
  --critic-lr 2e-6
  --static-ema 0.999
  --adaptive-ema 0.99
  --adaptive-weight 0.05
  --adaptive-start 1000
  --adaptive-warmup 4000
  --critic-frequency 2
  --lr-warmup 6250
  --schedule-total-steps 125000
  --log-every 25
  --seed 260823
)

PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --variant static \
  --save-every 1000 \
  "${COMMON[@]}" \
  2>&1 | tee "$ROOT/static_prefix5000.log"

PYTHONPATH=/home/zhoushunyu/eqvae python -m experiments.advfd_cleanroom.run_pmf_pilot \
  --variant real \
  --save-every 500 \
  "${COMMON[@]}" \
  2>&1 | tee "$ROOT/real_prefix5000.log"

PYTHONPATH=/home/zhoushunyu/eqvae \
  bash experiments/advfd_cleanroom/run_pmf_inception2048_adaptive512_eval5k.sh \
  "$ROOT" "$DEVICE"
