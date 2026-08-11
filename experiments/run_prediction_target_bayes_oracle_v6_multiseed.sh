#!/usr/bin/env bash
set -euo pipefail

ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/experiments/prediction_target_bayes_oracle_v6_trajectory}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260901 20260902 20260903 20260904)

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
    --architectures residual \
    --hidden-dims 64,80,96,128 \
    --milestones 6000,10000,15000,20000,30000 \
    --batch-size 2048 \
    --log-every 500 \
    --validation-samples 4096 \
    --teacher-samples 8192 \
    --sample-count 5000 \
    --sample-steps 100 \
    --gammas=-0.03,0.01,0.03,0.1 \
    --geometry-gammas=-0.03,0.03 \
    --seeds "${seed}" \
    --device cuda \
    --save-samples \
    --resume >"${log}" 2>&1
}

pids=()
for index in "${!SEEDS[@]}"; do
  run_seed "${index}" "${SEEDS[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one trajectory seed failed. Inspect ${ROOT}/logs." >&2
  exit "${status}"
fi

PYTHONPATH=. "${PYTHON_BIN}" \
  experiments/summarize_prediction_target_bayes_oracle_v6.py \
  --input-root "${ROOT}/runs" \
  --output-dir "${ROOT}/aggregate"

echo "[$(date -Iseconds)] trajectory discovery complete: ${ROOT}"
