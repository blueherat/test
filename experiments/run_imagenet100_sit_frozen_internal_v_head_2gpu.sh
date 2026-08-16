#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

GPU_LIST="${CUDA_VISIBLE_DEVICES:-1,2}"
NPROC="$(awk -F, '{print NF}' <<<"${GPU_LIST}")"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
VALIDATION_EVERY="${VALIDATION_EVERY:-5000}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
INTERNAL_DEPTH="${INTERNAL_DEPTH:-8}"
PREDICTION_TARGET="${PREDICTION_TARGET:-velocity}"
CLEAN_VELOCITY_DENOMINATOR_FLOOR="${CLEAN_VELOCITY_DENOMINATOR_FLOOR:-0.05}"
SEED="${SEED:-0}"
SOURCE_STATE_KEY="${SOURCE_STATE_KEY:-ema}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_seed0/checkpoints/step_00800000.pt}"
if [[ "${PREDICTION_TARGET}" == "velocity" ]]; then
  TARGET_TAG="v"
elif [[ "${PREDICTION_TARGET}" == "clean" ]]; then
  TARGET_TAG="x"
elif [[ "${PREDICTION_TARGET}" == "epsilon" ]]; then
  TARGET_TAG="eps"
else
  echo "Unsupported PREDICTION_TARGET=${PREDICTION_TARGET}" >&2
  exit 2
fi
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_v800-${SOURCE_STATE_KEY}_frozen-internal-${TARGET_TAG}-depth${INTERNAL_DEPTH}_seed${SEED}}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/zhoushunyu/data/eqvae/torchinductor_cache}"

mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTPUT_DIR}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  experiments/train_imagenet100_sit_frozen_internal_v_head.py \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --source-state-key "${SOURCE_STATE_KEY}" \
  --internal-depth "${INTERNAL_DEPTH}" \
  --prediction-target "${PREDICTION_TARGET}" \
  --clean-velocity-denominator-floor "${CLEAN_VELOCITY_DENOMINATOR_FLOOR}" \
  --output-dir "${OUTPUT_DIR}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --precision bf16 \
  --compile \
  --compile-mode default \
  --allow-tf32 \
  --num-workers 4 \
  --prefetch-factor 4 \
  --log-every "${LOG_EVERY}" \
  --validation-every "${VALIDATION_EVERY}" \
  --validation-batches "${VALIDATION_BATCHES}" \
  --save-every "${SAVE_EVERY}" \
  --resume auto \
  "$@"
