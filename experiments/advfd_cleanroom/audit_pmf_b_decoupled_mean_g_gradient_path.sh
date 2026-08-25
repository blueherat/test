#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

TRAIN_SESSION="${TRAIN_SESSION:-advfd_mean_g_5k}"
RUN_ROOT="${RUN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_w3_b24_5k_v1}"
EXP_NAME="${EXP_NAME:-pmf_b_sim_advinc_full_d_mean_g_officialavg_w3_b24_q50k_5k}"
CHECKPOINT_ROOT="${RUN_ROOT}/eqvae_advfd_reproduction/${EXP_NAME}/checkpoints"
MANIFEST="${RUN_ROOT}/${EXP_NAME}_adapter_manifest.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_generator_component_gradient_audit_v1/mean_g_path}"
DEVICE="${DEVICE:-cuda:3}"
BATCH_SIZE="${BATCH_SIZE:-24}"
TRIALS="${TRIALS:-1}"

mkdir -p "${OUTPUT_ROOT}"

shopt -s nullglob
while true; do
  audited_checkpoint=false
  checkpoints=("${CHECKPOINT_ROOT}"/step_*.pth)
  for checkpoint in "${checkpoints[@]}"; do
    filename="$(basename "${checkpoint}")"
    step="${filename#step_}"
    step="${step%.pth}"
    output="${OUTPUT_ROOT}/step_${step}_b${BATCH_SIZE}_t${TRIALS}.json"
    if [[ -f "${output}" ]]; then
      continue
    fi
    python experiments/advfd_cleanroom/audit_pmf_generator_component_gradients.py \
      --checkpoint "${checkpoint}" \
      --adapter-manifest "${MANIFEST}" \
      --batch-size "${BATCH_SIZE}" \
      --trials "${TRIALS}" \
      --parameter-gradient-trials "${TRIALS}" \
      --device "${DEVICE}" \
      --output-json "${output}"
    audited_checkpoint=true
  done
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    if [[ "${audited_checkpoint}" == false ]]; then
      break
    fi
  fi
  sleep 30
done

touch "${OUTPUT_ROOT}/_SUCCESS"
