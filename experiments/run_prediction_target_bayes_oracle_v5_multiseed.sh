#!/usr/bin/env bash
set -euo pipefail

ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/experiments/prediction_target_bayes_oracle_v5_formal}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS=(20260821 20260822 20260823 20260824)

mkdir -p "${ROOT}/logs"

run_seed() {
  local gpu="$1"
  local seed="$2"
  local run_root="${ROOT}/runs/seed${seed}"
  local log="${ROOT}/logs/seed${seed}.log"

  {
    echo "[$(date -Iseconds)] seed=${seed} GPU=${gpu}: tube distribution"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON_BIN}" \
      experiments/run_prediction_target_bayes_oracle_v5.py \
      --output-root "${run_root}/tube" \
      --D 64 \
      --components 32 \
      --sigma-tangent 0.35 \
      --sigma-normal 0.03 \
      --architectures jit_relu,plain,residual,residual_skip \
      --hidden-dims 64,128 \
      --train-steps 6000 \
      --batch-size 2048 \
      --log-every 500 \
      --validation-samples 4096 \
      --teacher-samples 8192 \
      --sample-count 5000 \
      --sample-steps 100 \
      --gammas=-0.1,-0.03,-0.01,0.01,0.03,0.1 \
      --geometry-gammas=-0.03,0.03 \
      --seeds "${seed}" \
      --device cuda \
      --resume

    echo "[$(date -Iseconds)] seed=${seed} GPU=${gpu}: strict-support control"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=. "${PYTHON_BIN}" \
      experiments/run_prediction_target_bayes_oracle_v5.py \
      --output-root "${run_root}/strict_support" \
      --D 64 \
      --components 32 \
      --sigma-tangent 0.35 \
      --sigma-normal 0.0 \
      --architectures jit_relu,plain,residual_skip \
      --hidden-dims 64,128 \
      --train-steps 6000 \
      --batch-size 2048 \
      --log-every 500 \
      --validation-samples 4096 \
      --teacher-samples 8192 \
      --sample-count 5000 \
      --sample-steps 100 \
      --gammas=-0.1,-0.03,-0.01,0.01,0.03,0.1 \
      --geometry-gammas=-0.03,0.03 \
      --seeds "${seed}" \
      --device cuda \
      --resume
    echo "[$(date -Iseconds)] seed=${seed}: complete"
  } >"${log}" 2>&1
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
  echo "At least one seed failed. Inspect ${ROOT}/logs." >&2
  exit "${status}"
fi

PYTHONPATH=. "${PYTHON_BIN}" \
  experiments/summarize_prediction_target_bayes_oracle_v5.py \
  --input-root "${ROOT}/runs" \
  --output-dir "${ROOT}/aggregate"

echo "[$(date -Iseconds)] all v5 exact-Bayes runs complete: ${ROOT}"
