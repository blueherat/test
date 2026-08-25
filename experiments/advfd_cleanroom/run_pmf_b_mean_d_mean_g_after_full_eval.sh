#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

PREREQUISITE_SUCCESS="${PREREQUISITE_SUCCESS:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_paired5k_v1/_SUCCESS}"
PREREQUISITE_SESSION="${PREREQUISITE_SESSION:-advfd_mean_g_eval5k}"

while [[ ! -f "${PREREQUISITE_SUCCESS}" ]]; do
  if ! tmux has-session -t "${PREREQUISITE_SESSION}" 2>/dev/null; then
    echo "Prerequisite evaluation ended without ${PREREQUISITE_SUCCESS}" >&2
    exit 1
  fi
  sleep 60
done

GPU_IDS="${GPU_IDS:-0,1,2}" \
LOCAL_BATCH="${LOCAL_BATCH:-24}" \
QUEUE_SIZE="${QUEUE_SIZE:-50000}" \
EPOCHS="${EPOCHS:-4}" \
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1250}" \
SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS:-10000}" \
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_mean_d_mean_g_w3_b24_5k_v1}" \
EXP_NAME="${EXP_NAME:-pmf_b_sim_advinc_mean_d_mean_g_officialavg_w3_b24_q50k_5k}" \
FD_ADV_CRITIC_COMPONENT=mean \
FD_ADV_GENERATOR_COMPONENT=mean \
bash experiments/advfd_cleanroom/run_pmf_b_decoupled_mean_g.sh
