#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

LOG_DIR=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/logs
mkdir -p "$LOG_DIR"

python experiments/run_imagenet100_sit_static_pair_fid5k.py \
  --output-root /home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_static_pair_v_to_jit_x_step400000_seed0 \
  --scales 0 1 -0.05 -0.1 -0.2 -0.3 -0.5 -0.75 -1.0 0.1 0.25 0.5 0.75 1.25 1.5 \
  --num-samples 5000 \
  --per-rank-batch-size 8 \
  --vae-decode-batch-size 2 \
  --cuda-allocator-limit-gib 4 \
  --sampling-cuda-visible-devices 3 \
  --fid-batch-size 8 \
  --fid-gpu-memory-fraction 0.25 \
  --fid-cuda-visible-devices 3 \
  --gpu-memory-ceiling-mib 8192 \
  --memory-poll-interval 0.25 \
  --global-seed 0 \
  2>&1 | tee "$LOG_DIR/static_pair_v_to_jit_x_400k_fid5k.log"
