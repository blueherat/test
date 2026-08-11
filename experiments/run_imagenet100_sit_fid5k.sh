#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# Canonical low-memory ImageNet-100 SiT sampling/FID protocol.
# Every GPU is guarded below 9 GiB; sampling uses all four GPUs.
STEP="${STEP:-800000}"
NUM_SAMPLES="${NUM_SAMPLES:-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_lowmem_v1}"

exec python experiments/run_imagenet100_sit_fid_curve.py \
  --steps "${STEP}" \
  --output-root "${OUTPUT_ROOT}" \
  --num-samples "${NUM_SAMPLES}" \
  --sampling-cuda-visible-devices 0,1,2,3 \
  --per-rank-batch-size 64 \
  --vae-decode-batch-size 4 \
  --cuda-allocator-limit-gib 7.5 \
  --fid-cuda-visible-devices 0 \
  --fid-batch-size 8 \
  --fid-gpu-memory-fraction 0.30 \
  --gpu-memory-ceiling-mib 9216 \
  --memory-poll-interval 0.25
