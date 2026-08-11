#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# Independent single-head controls for the failed x+epsilon dual-head SiT run.
# The original velocity baseline and both controls share SiT-S/2, ImageNet-100
# latent cache, seed, four-rank DDP, global batch 256, optimizer, EMA and steps.
# Each endpoint target uses its native elementwise MSE, matching the two losses
# used by the dual-head experiment without shared-trunk interference.
TARGETS=(x epsilon)
MILESTONES=(100000 200000)
SEED="${SEED:-0}"
DENOMINATOR_FLOOR="${DENOMINATOR_FLOOR:-0.001}"
BASE_DIR="${BASE_DIR:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
PIPELINE_LOG="${BASE_DIR}/single_target_native_overnight_seed${SEED}.log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${BASE_DIR}/runs"
echo "[$(date --iso-8601=seconds)] start independent x/epsilon native-loss controls" \
  | tee -a "${PIPELINE_LOG}"

for TARGET in "${TARGETS[@]}"; do
  RUN_DIR="${BASE_DIR}/runs/sit-s-2_${TARGET}-native-loss_seed${SEED}"
  FID_ROOT="${BASE_DIR}/fid5k_single-target_${TARGET}-native_seed${SEED}"
  mkdir -p "${RUN_DIR}"

  for TARGET_STEP in "${MILESTONES[@]}"; do
    echo "[$(date --iso-8601=seconds)] ${TARGET}/native -> step ${TARGET_STEP}" \
      | tee -a "${PIPELINE_LOG}"
    PREDICTION_TARGET="${TARGET}" \
    LOSS_SPACE=native \
    DENOMINATOR_FLOOR="${DENOMINATOR_FLOOR}" \
    SIT_MODEL=SiT-S/2 \
    GLOBAL_BATCH_SIZE=256 \
    MAX_STEPS="${TARGET_STEP}" \
    SAVE_EVERY=100000 \
    SEED="${SEED}" \
    OUTPUT_DIR="${RUN_DIR}" \
    bash experiments/run_imagenet100_sit_4gpu.sh \
      2>&1 | tee -a "${RUN_DIR}/train_to_${TARGET_STEP}.log"

    echo "[$(date --iso-8601=seconds)] ${TARGET}/native FID-5K at ${TARGET_STEP}" \
      | tee -a "${PIPELINE_LOG}"
    python experiments/run_imagenet100_sit_fid_curve.py \
      --steps "${TARGET_STEP}" \
      --run-dir "${RUN_DIR}" \
      --output-root "${FID_ROOT}" \
      --sampling-cuda-visible-devices 0,1,2,3 \
      --per-rank-batch-size 64 \
      --vae-decode-batch-size 4 \
      --cuda-allocator-limit-gib 7.5 \
      --fid-cuda-visible-devices 0 \
      --fid-batch-size 8 \
      --fid-gpu-memory-fraction 0.30 \
      --gpu-memory-ceiling-mib 9216 \
      --memory-poll-interval 0.25 \
      2>&1 | tee -a "${PIPELINE_LOG}"
  done

  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps 100000,200000 \
    --run-dir "${RUN_DIR}" \
    --output-root "${FID_ROOT}" \
    2>&1 | tee -a "${PIPELINE_LOG}"
  touch "${RUN_DIR}/COMPLETE_200K_WITH_FID"
done

touch "${BASE_DIR}/COMPLETE_SINGLE_TARGET_NATIVE_200K_WITH_FID"
echo "[$(date --iso-8601=seconds)] complete independent x/epsilon controls" \
  | tee -a "${PIPELINE_LOG}"
