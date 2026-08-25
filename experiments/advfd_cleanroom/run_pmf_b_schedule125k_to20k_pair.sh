#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
LOCAL_BATCH="${LOCAL_BATCH:-24}"
SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS:-125000}"
MID_STEPS="${MID_STEPS:-10000}"
TARGET_STEPS="${TARGET_STEPS:-20000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1250}"
WAIT_SESSION="${WAIT_SESSION:-advfd_pmf_sched125k_to10k}"
POLL_SECONDS="${POLL_SECONDS:-60}"

for steps in "${MID_STEPS}" "${TARGET_STEPS}"; do
  if (( steps % STEPS_PER_EPOCH != 0 )); then
    echo "step count ${steps} must be divisible by ${STEPS_PER_EPOCH}" >&2
    exit 1
  fi
done
if (( MID_STEPS >= TARGET_STEPS )); then
  echo "MID_STEPS must be smaller than TARGET_STEPS" >&2
  exit 1
fi

MID_EPOCHS="$((MID_STEPS / STEPS_PER_EPOCH))"
TARGET_EPOCHS="$((TARGET_STEPS / STEPS_PER_EPOCH))"
# The official AdvFD checkpoint utility formats zero-based loop indices with
# seven digits (for example, 10K updates -> step_0009999.pth).
printf -v MID_INDEX "%07d" "$((MID_STEPS - 1))"
printf -v TARGET_INDEX "%07d" "$((TARGET_STEPS - 1))"

ADV_ROOT="${ADV_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w4_b24_25k_v1}"
ADV_EXP="${ADV_EXP:-pmf_b_sim_advinc_officialavg_w4_b24_q50k_25k}"
ADV_CKPT_DIR="${ADV_ROOT}/eqvae_advfd_reproduction/${ADV_EXP}/checkpoints"
ADV_MID_CKPT="${ADV_CKPT_DIR}/step_${MID_INDEX}.pth"
ADV_TARGET_CKPT="${ADV_CKPT_DIR}/step_${TARGET_INDEX}.pth"

STATIC_ROOT="${STATIC_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w4_b24_schedule125k_to20k_v1}"
STATIC_EXP="${STATIC_EXP:-pmf_b_sim_static_officialavg_w4_b24_q50k_schedule125k_to20k}"
STATIC_CKPT_DIR="${STATIC_ROOT}/eqvae_advfd_static_reproduction/${STATIC_EXP}/checkpoints"
STATIC_MID_CKPT="${STATIC_CKPT_DIR}/step_${MID_INDEX}.pth"
STATIC_TARGET_CKPT="${STATIC_CKPT_DIR}/step_${TARGET_INDEX}.pth"

PAIR_ROOT="${PAIR_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_schedule125k_to20k_paired5k_v1}"
PIPELINE_ROOT="${PIPELINE_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_schedule125k_to20k_pipeline_v1}"
PRESERVED_ROOT="${PIPELINE_ROOT}/preserved_checkpoints"
SUCCESS_MARKER="${PIPELINE_ROOT}/_SUCCESS"
FAILURE_MARKER="${PIPELINE_ROOT}/_FAILED"
ADV_MID_PRESERVED="${PRESERVED_ROOT}/advfd_step_${MID_INDEX}.pth"
STATIC_MID_PRESERVED="${PRESERVED_ROOT}/static_step_${MID_INDEX}.pth"

mkdir -p "${PIPELINE_ROOT}" "${PRESERVED_ROOT}"
rm -f "${SUCCESS_MARKER}" "${FAILURE_MARKER}"
exec > >(tee -a "${PIPELINE_ROOT}/pipeline.log") 2>&1
trap 'status=$?; if (( status != 0 )); then printf "exit_status=%s\n" "${status}" > "${FAILURE_MARKER}"; fi' EXIT

preserve_checkpoint() {
  local source="$1"
  local destination="$2"
  if [[ -e "${destination}" ]]; then
    if [[ -e "${source}" ]] && \
       [[ "$(stat -c '%d:%i' "${source}")" != "$(stat -c '%d:%i' "${destination}")" ]]; then
      echo "Refusing to replace a different preserved checkpoint: ${destination}" >&2
      exit 1
    fi
    return
  fi
  if [[ ! -f "${source}" ]]; then
    echo "Missing checkpoint to preserve: ${source}" >&2
    exit 1
  fi
  ln "${source}" "${destination}"
  echo "Preserved checkpoint: ${destination}"
}

run_advfd() {
  local epochs="$1"
  GPU_IDS="${GPU_IDS}" \
  LOCAL_BATCH="${LOCAL_BATCH}" \
  QUEUE_SIZE=50000 \
  EPOCHS="${epochs}" \
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH}" \
  FD_ADV_START_STEP=1000 \
  FD_ADV_WARMUP_STEPS=4000 \
  FD_ADV_UPDATE_FREQ=2 \
  PRINT_FREQ=20 \
  SAVE_FREQ=5000 \
  MILESTONE_INTERVAL=4 \
  NUM_WORKERS=4 \
  OUTPUT_ROOT="${ADV_ROOT}" \
  EXP_NAME="${ADV_EXP}" \
  SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS}" \
  bash experiments/advfd_cleanroom/run_pmf_b_official_code.sh
}

run_static() {
  local epochs="$1"
  GPU_IDS="${GPU_IDS}" \
  LOCAL_BATCH="${LOCAL_BATCH}" \
  QUEUE_SIZE=50000 \
  EPOCHS="${epochs}" \
  STEPS_PER_EPOCH="${STEPS_PER_EPOCH}" \
  PRINT_FREQ=20 \
  SAVE_FREQ=5000 \
  MILESTONE_INTERVAL=4 \
  NUM_WORKERS=4 \
  OUTPUT_ROOT="${STATIC_ROOT}" \
  EXP_NAME="${STATIC_EXP}" \
  SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS}" \
  bash experiments/advfd_cleanroom/run_pmf_b_official_static_code.sh
}

evaluate_pair() {
  local tag="$1"
  local static_checkpoint="$2"
  local advfd_checkpoint="$3"
  local output_root="${PAIR_ROOT}/${tag}"
  if [[ -f "${output_root}/_SUCCESS" ]]; then
    echo "Reusing completed paired evaluation: ${output_root}"
    return
  fi
  GPU_IDS="${GPU_IDS}" \
  ROOT="${output_root}" \
  STATIC_CHECKPOINT="${static_checkpoint}" \
  ADVFD_CHECKPOINT="${advfd_checkpoint}" \
  GEN_BATCH=32 \
  FD_BATCH=16 \
  NUM_IMAGES=5000 \
  bash experiments/advfd_cleanroom/eval_pmf_b_official_10k_fdr3.sh
}

echo "Protocol: pMF-B, global batch $((LOCAL_BATCH * NPROC)), schedule horizon ${SCHEDULE_TOTAL_STEPS}, checkpoints ${MID_STEPS}/${TARGET_STEPS}."

# Another tmux job is currently producing the AdvFD 10K checkpoint. Waiting
# here avoids two writers touching the same checkpoint directory.
if [[ ! -f "${ADV_MID_CKPT}" && ! -f "${ADV_MID_PRESERVED}" ]] && \
   tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; then
  echo "Waiting for tmux session ${WAIT_SESSION} to finish AdvFD ${MID_STEPS}..."
  while [[ ! -f "${ADV_MID_CKPT}" ]] && tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
  # The checkpoint is written atomically, but leave the old launcher enough
  # time to finish its final barrier before opening the same directory again.
  if [[ -f "${ADV_MID_CKPT}" ]]; then
    sleep 30
  fi
fi
if [[ ! -f "${ADV_MID_CKPT}" && ! -f "${ADV_MID_PRESERVED}" ]]; then
  echo "AdvFD ${MID_STEPS} checkpoint was not produced; resuming it now."
  run_advfd "${MID_EPOCHS}"
fi

preserve_checkpoint "${ADV_MID_CKPT}" "${ADV_MID_PRESERVED}"

if [[ ! -f "${ADV_TARGET_CKPT}" ]]; then
  echo "Continuing AdvFD ${MID_STEPS} -> ${TARGET_STEPS}."
  run_advfd "${TARGET_EPOCHS}"
fi
if [[ ! -f "${ADV_TARGET_CKPT}" ]]; then
  echo "Missing final AdvFD checkpoint: ${ADV_TARGET_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${STATIC_MID_CKPT}" && ! -f "${STATIC_MID_PRESERVED}" && ! -f "${STATIC_TARGET_CKPT}" ]]; then
  echo "Training static FD-Loss 0 -> ${MID_STEPS} from the original pMF-B checkpoint."
  run_static "${MID_EPOCHS}"
fi
preserve_checkpoint "${STATIC_MID_CKPT}" "${STATIC_MID_PRESERVED}"

if [[ ! -f "${STATIC_TARGET_CKPT}" ]]; then
  echo "Continuing static FD-Loss ${MID_STEPS} -> ${TARGET_STEPS}."
  run_static "${TARGET_EPOCHS}"
fi
if [[ ! -f "${STATIC_TARGET_CKPT}" ]]; then
  echo "Missing final static checkpoint: ${STATIC_TARGET_CKPT}" >&2
  exit 1
fi

echo "Evaluating the paired ${MID_STEPS}-step checkpoints."
evaluate_pair "step_${MID_STEPS}" "${STATIC_MID_PRESERVED}" "${ADV_MID_PRESERVED}"

echo "Evaluating the paired ${TARGET_STEPS}-step checkpoints."
evaluate_pair "step_${TARGET_STEPS}" "${STATIC_TARGET_CKPT}" "${ADV_TARGET_CKPT}"

touch "${SUCCESS_MARKER}"
echo "Completed AdvFD/static FD-Loss training and paired evaluation."
