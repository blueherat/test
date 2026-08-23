#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

EVAL_SESSION="${EVAL_SESSION:-advfd_official_pmf_b_eval_after_10k}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1}"
TRAIN_EXP="${TRAIN_EXP:-pmf_b_sim_advinc_officialavg_w3_b24_q50k_10k}"
CHECKPOINT_ROOT="${TRAIN_ROOT}/eqvae_advfd_reproduction/${TRAIN_EXP}/checkpoints"
EVAL_ROOT="${EVAL_ROOT:-${TRAIN_ROOT}/official_eval5k}"
AUDIT_ROOT="${AUDIT_ROOT:-${TRAIN_ROOT}/fresh_feature_audit5k}"
EVAL_SUCCESS_MARKER="${EVAL_ROOT}/_SUCCESS"
EVAL_FAILURE_MARKER="${EVAL_ROOT}/_FAILED"
AUDIT_SUCCESS_MARKER="${AUDIT_ROOT}/_SUCCESS"
AUDIT_FAILURE_MARKER="${AUDIT_ROOT}/_FAILED"
GPU_IDS="${GPU_IDS:-1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_IMAGES="${NUM_IMAGES:-5000}"

while [[ ! -f "${EVAL_SUCCESS_MARKER}" ]]; do
  if [[ -f "${EVAL_FAILURE_MARKER}" ]]; then
    echo "Evaluation failed; refusing to treat a vanished tmux session as success." >&2
    exit 1
  fi
  if ! tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
    echo "Evaluation session ended without success marker: ${EVAL_SUCCESS_MARKER}" >&2
    exit 1
  fi
  sleep 60
done

mkdir -p "${AUDIT_ROOT}"
rm -f "${AUDIT_SUCCESS_MARKER}" "${AUDIT_FAILURE_MARKER}"
trap 'status=$?; if (( status != 0 )); then printf "exit_status=%s\n" "${status}" > "${AUDIT_FAILURE_MARKER}"; fi' EXIT
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:/home/zhoushunyu/eqvae:${PYTHONPATH:-}"

find_generated_folder() {
  local condition="$1"
  find "${EVAL_ROOT}" -type d \
    -path "*/${condition}/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7" \
    -print -quit
}

run_audit() {
  local condition="$1"
  local checkpoint="$2"
  local generated_folder
  generated_folder="$(find_generated_folder "${condition}")"
  if [[ -z "${generated_folder}" ]]; then
    echo "Generated folder not found for ${condition} under ${EVAL_ROOT}" >&2
    exit 1
  fi
  local image_count
  image_count="$(find "${generated_folder}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if (( image_count < NUM_IMAGES )); then
    echo "Expected ${NUM_IMAGES} generated images for ${condition}, found ${image_count}" >&2
    exit 1
  fi

  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/audit_official_advfd_features.py \
    --checkpoint "${checkpoint}" \
    --generated-folder "${generated_folder}" \
    --imagenet-root /data/shared/imagenet-1k \
    --real-split validation \
    --num-images "${NUM_IMAGES}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers 4 \
    --output-json "${AUDIT_ROOT}/${condition}.json" \
    2>&1 | tee "${AUDIT_ROOT}/${condition}.log"
}

run_audit step_0005000 "${CHECKPOINT_ROOT}/step_0005000.pth"
run_audit step_0009999 "${CHECKPOINT_ROOT}/step_0009999.pth"
touch "${AUDIT_SUCCESS_MARKER}"
