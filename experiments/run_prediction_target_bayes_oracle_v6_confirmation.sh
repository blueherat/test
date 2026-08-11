#!/usr/bin/env bash
set -euo pipefail

ARCHITECTURE="${ARCHITECTURE:?set ARCHITECTURE from the frozen discovery result}"
HIDDEN="${HIDDEN:?set HIDDEN from the frozen discovery result}"
STEP="${STEP:?set STEP from the frozen discovery result}"
GAMMA="${GAMMA:?set GAMMA from the frozen discovery result}"
ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/experiments/prediction_target_bayes_oracle_v6_confirmation/${ARCHITECTURE}_H${HIDDEN}_S${STEP}_G${GAMMA}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20261001 20261002 20261003 20261004 20261005)

mkdir -p "${ROOT}/logs"

run_seed() {
  local gpu="$1"
  local seed="$2"
  local log="${ROOT}/logs/seed${seed}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON_BIN}" \
    experiments/run_prediction_target_bayes_oracle_v6_trajectory.py \
    --output-root "${ROOT}/runs" \
    --D 64 \
    --components 32 \
    --sigma-tangent 0.35 \
    --sigma-normal 0.03 \
    --architectures "${ARCHITECTURE}" \
    --hidden-dims "${HIDDEN}" \
    --milestones "${STEP}" \
    --batch-size 2048 \
    --log-every 500 \
    --validation-samples 4096 \
    --teacher-samples 8192 \
    --sample-count 5000 \
    --sample-steps 100 \
    --gammas="${GAMMA}" \
    --geometry-gammas="${GAMMA}" \
    --seeds "${seed}" \
    --device cuda \
    --save-samples \
    --resume >"${log}" 2>&1
}

pids=()
for gpu in 0 1 2 3; do
  run_seed "${gpu}" "${SEEDS[$gpu]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "A first-wave confirmation seed failed. Inspect ${ROOT}/logs." >&2
  exit "${status}"
fi

run_seed 0 "${SEEDS[4]}"

PYTHONPATH=. "${PYTHON_BIN}" \
  experiments/summarize_prediction_target_bayes_oracle_v6.py \
  --input-root "${ROOT}/runs" \
  --output-dir "${ROOT}/aggregate"

echo "[$(date -Iseconds)] frozen five-seed confirmation complete: ${ROOT}"
