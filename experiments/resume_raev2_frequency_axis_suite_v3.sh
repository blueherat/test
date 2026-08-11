#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --standalone --nproc_per_node=4 \
  experiments/run_raev2_frequency_axis_extrapolation_suite_v3.py \
  --output-root /data/users/zhoushunyu/eqvae/experiments/frequency_axis_suite_v1_full \
  --protocol-json experiments/configs/frequency_axis_extrapolation_suite_v1_full.json \
  --stages sample,decode,evaluate,report \
  --resume \
  --reference-samples 1024 \
  --rollout-samples 1024 \
  --pulse-samples 128 \
  --sample-count 5000 \
  --per-rank-batch 1 \
  --metric-batch-size 64 \
  --preview-count 32 \
  --precision fp32 \
  --distributed-timeout-minutes 360
