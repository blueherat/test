#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="$(awk -F, '{print NF}' <<<"${GPU_LIST}")"
SIT_MODEL="${SIT_MODEL:-SiT-S/2}"
MODEL_TAG="${SIT_MODEL//\//-}"
MODEL_TAG="${MODEL_TAG,,}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
VALIDATION_EVERY="${VALIDATION_EVERY:-5000}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-8}"
LOG_EVERY="${LOG_EVERY:-50}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/${MODEL_TAG}_dual-output_seed${SEED}}"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/zhoushunyu/data/eqvae/torchinductor_cache}"

mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"

exec torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  experiments/train_imagenet100_sit_dual_output.py \
  --model "${SIT_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
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
