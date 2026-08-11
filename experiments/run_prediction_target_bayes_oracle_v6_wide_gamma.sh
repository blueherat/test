#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/zhoushunyu/data/eqvae/experiments/prediction_target_bayes_oracle_v6_trajectory/runs}"
ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/experiments/prediction_target_bayes_oracle_v6_wide_gamma}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260901 20260902 20260903 20260904)

mkdir -p "${ROOT}/logs"

run_seed() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON_BIN}" \
    experiments/reevaluate_prediction_target_bayes_oracle_v6_wide_gamma.py \
    --input-root "${SOURCE_ROOT}" \
    --output-root "${ROOT}/runs" \
    --architectures residual \
    --hidden-dims 64,80,96,128 \
    --steps 6000,10000,15000,20000,30000 \
    --gammas=0.01,0.03,0.1,0.2,0.3,0.5,0.78,1.0 \
    --geometry-gammas=0.1,0.5 \
    --seeds "${seed}" \
    --device cuda \
    --resume >"${ROOT}/logs/seed${seed}.log" 2>&1
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
  echo "At least one wide-gamma seed failed. Inspect ${ROOT}/logs." >&2
  exit "${status}"
fi

PYTHONPATH=. "${PYTHON_BIN}" \
  experiments/summarize_prediction_target_bayes_oracle_v6.py \
  --input-root "${ROOT}/runs" \
  --output-dir "${ROOT}/aggregate"

echo "[$(date -Iseconds)] wide-gamma checkpoint evaluation complete: ${ROOT}"
