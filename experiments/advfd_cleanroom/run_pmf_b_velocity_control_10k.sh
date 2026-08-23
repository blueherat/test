#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
GLOBAL_BATCH="${GLOBAL_BATCH:-72}"
if (( GLOBAL_BATCH % NPROC != 0 )); then
  echo "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NPROC=${NPROC}" >&2
  exit 2
fi
LOCAL_BATCH=$((GLOBAL_BATCH / NPROC))

TOTAL_STEPS="${TOTAL_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-6250}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_velocity_control_10k_v1}"
EXP_NAME="${EXP_NAME:-pmf_b_local_velocity_w${NPROC}_b${LOCAL_BATCH}_10k}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
PACKED_DATA="${PACKED_DATA:-/data/shared/imagenet-1k/random_access_v1}"
LOAD_FROM="${LOAD_FROM:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PRINT_FREQ="${PRINT_FREQ:-20}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p "${OUTPUT_ROOT}/${EXP_NAME}"

torchrun --standalone --nproc-per-node="${NPROC}" \
  experiments/advfd_cleanroom/train_pmf_velocity_continuation.py \
  --official-root "${OFFICIAL_ROOT}" \
  --packed-data "${PACKED_DATA}" \
  --load-from "${LOAD_FROM}" \
  --output-dir "${OUTPUT_ROOT}" \
  --exp-name "${EXP_NAME}" \
  --batch-size "${LOCAL_BATCH}" \
  --total-steps "${TOTAL_STEPS}" \
  --warmup-steps "${WARMUP_STEPS}" \
  --save-steps 5000 "${TOTAL_STEPS}" \
  --lr 1e-6 --min-lr 0 \
  --weight-decay 0 --beta1 0.9 --beta2 0.95 \
  --seed 1 --dtype bf16 --grad-checkpointing \
  --num-workers "${NUM_WORKERS}" --print-freq "${PRINT_FREQ}" \
  --auto-resume \
  2>&1 | tee -a "${OUTPUT_ROOT}/${EXP_NAME}/launcher.log"
