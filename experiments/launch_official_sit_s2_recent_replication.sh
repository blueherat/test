#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

SESSION_NAME="${SESSION_NAME:-official_sit_s2_repro}"
PROFILE="${PROFILE:-full}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/official_sit_s2_recent_replication_v1}"
MASTER_LOG="${OUTPUT_ROOT}/pipeline.log"
GPU_MEMORY_CEILING_MIB="${GPU_MEMORY_CEILING_MIB:-22528}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
tmux new-session -d -s "${SESSION_NAME}" \
  "bash -lc 'cd /home/zhoushunyu/eqvae && exec python experiments/run_official_sit_s2_recent_replication.py --profile ${PROFILE} --gpus ${GPU_LIST} --gpu-memory-ceiling-mib ${GPU_MEMORY_CEILING_MIB} --output-root ${OUTPUT_ROOT} 2>&1 | tee -a ${MASTER_LOG}'"

echo "started tmux session: ${SESSION_NAME}"
echo "log: ${MASTER_LOG}"
echo "attach: tmux attach -t ${SESSION_NAME}"
