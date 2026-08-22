#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_progressive-innovation-split6_seed0}"
WEAK_STEPS="${WEAK_STEPS:-100000}"
INNOVATION_STEPS="${INNOVATION_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-50000}"
VALIDATION_EVERY="${VALIDATION_EVERY:-5000}"
NPROC="${NPROC:-2}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  experiments/train_imagenet100_sit_progressive_innovation.py \
  --phase weak \
  --output-dir "${ROOT}/phase_weak" \
  --split-depth 6 \
  --global-batch-size 256 \
  --max-steps "${WEAK_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --validation-every "${VALIDATION_EVERY}" \
  --resume auto

STAGE1_CHECKPOINT="${ROOT}/phase_weak/checkpoints/step_$(printf '%08d' "${WEAK_STEPS}").pt"
test -f "${STAGE1_CHECKPOINT}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  experiments/train_imagenet100_sit_progressive_innovation.py \
  --phase innovation \
  --stage1-checkpoint "${STAGE1_CHECKPOINT}" \
  --output-dir "${ROOT}/phase_innovation" \
  --split-depth 6 \
  --global-batch-size 256 \
  --max-steps "${INNOVATION_STEPS}" \
  --save-every "${SAVE_EVERY}" \
  --validation-every "${VALIDATION_EVERY}" \
  --resume auto
