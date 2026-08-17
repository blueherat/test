#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhoushunyu/eqvae
PYTHON=/home/zhoushunyu/miniconda3/envs/myenv/bin/python
SESSION=sit_multiscale_guidance
OUTPUT=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1
LOG=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/multiscale_guidance_study_v1/pipeline.log

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${OUTPUT}"
COMMAND="cd ${ROOT} && set -o pipefail && exec ${PYTHON} experiments/run_imagenet100_sit_multiscale_guidance_study.py --profile full --output-root ${OUTPUT} --gpu 2 --evaluation-gpus 2,3 --confirm-samples 0 2>&1 | tee -a ${LOG}"
tmux new-session -d -s "${SESSION}" "bash -lc '$COMMAND'"

echo "Started ${SESSION} on physical GPU2"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log: ${LOG}"
