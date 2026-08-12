#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# JiT-style prediction-parameterization control on the existing SiT protocol.
# This intentionally keeps SiT's Uniform[0,1) time distribution so that the
# only scientific change from the velocity baseline is the native x output and
# its clamped common-velocity loss. It is not an exact full JiT recipe.
GPU_LIST="${CUDA_VISIBLE_DEVICES:-2,3}"
if [[ "${GPU_LIST}" != "2,3" ]]; then
  echo "This run is restricted to physical GPUs 2,3; got CUDA_VISIBLE_DEVICES=${GPU_LIST}" >&2
  exit 2
fi

SEED="${SEED:-0}"
MAX_STEP="${MAX_STEP:-400000}"
DENOMINATOR_FLOOR="${DENOMINATOR_FLOOR:-0.05}"
BASE_DIR="${BASE_DIR:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
RUN_DIR="${BASE_DIR}/runs/sit-s-2_x-velocity-loss-floor0p05_seed${SEED}"
FID_ROOT="${BASE_DIR}/fid5k_single-target_x-velocity-floor0p05_seed${SEED}"
PIPELINE_LOG="${RUN_DIR}/pipeline_to_${MAX_STEP}.log"
MILESTONES=(100000 200000 300000 400000)
COMPLETED_STEPS=()

export CUDA_VISIBLE_DEVICES="2,3"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/eqvae_inductor_jit_x_2gpu}"

mkdir -p "${RUN_DIR}"
echo "[$(date --iso-8601=seconds)] start x-output/common-velocity-loss control" \
  | tee -a "${PIPELINE_LOG}"

for TARGET_STEP in "${MILESTONES[@]}"; do
  if (( TARGET_STEP > MAX_STEP )); then
    break
  fi
  echo "[$(date --iso-8601=seconds)] train x/velocity to ${TARGET_STEP}" \
    | tee -a "${PIPELINE_LOG}"
  PREDICTION_TARGET=x \
  LOSS_SPACE=velocity \
  DENOMINATOR_FLOOR="${DENOMINATOR_FLOOR}" \
  SIT_MODEL=SiT-S/2 \
  GLOBAL_BATCH_SIZE=256 \
  MAX_STEPS="${TARGET_STEP}" \
  SAVE_EVERY=100000 \
  SEED="${SEED}" \
  OUTPUT_DIR="${RUN_DIR}" \
  bash experiments/run_imagenet100_sit_4gpu.sh \
    2>&1 | tee -a "${RUN_DIR}/train_to_${TARGET_STEP}.log"

  echo "[$(date --iso-8601=seconds)] FID-5K at ${TARGET_STEP}" \
    | tee -a "${PIPELINE_LOG}"
  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps "${TARGET_STEP}" \
    --run-dir "${RUN_DIR}" \
    --output-root "${FID_ROOT}" \
    --sampling-cuda-visible-devices 2,3 \
    --per-rank-batch-size 64 \
    --vae-decode-batch-size 4 \
    --cuda-allocator-limit-gib 7.5 \
    --fid-cuda-visible-devices 2 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.30 \
    --gpu-memory-ceiling-mib 9216 \
    --memory-poll-interval 0.25 \
    2>&1 | tee -a "${PIPELINE_LOG}"
  COMPLETED_STEPS+=("${TARGET_STEP}")
done

if (( ${#COMPLETED_STEPS[@]} > 0 )); then
  STEP_LIST="$(IFS=,; echo "${COMPLETED_STEPS[*]}")"
  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps "${STEP_LIST}" \
    --run-dir "${RUN_DIR}" \
    --output-root "${FID_ROOT}" \
    --sampling-cuda-visible-devices 2,3 \
    --fid-cuda-visible-devices 2 \
    2>&1 | tee -a "${PIPELINE_LOG}"
  touch "${RUN_DIR}/COMPLETE_${MAX_STEP}_WITH_FID"
fi

echo "[$(date --iso-8601=seconds)] complete through ${MAX_STEP}" \
  | tee -a "${PIPELINE_LOG}"
