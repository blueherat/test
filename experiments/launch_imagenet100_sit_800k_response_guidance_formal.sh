#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/formal_n5000}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"

wait_for_gpu() {
  local gpu="$1"
  while [[ -n "$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for GPU $gpu"
    sleep 30
  done
}

run_condition() {
  local gpu="$1" seed="$2" mode="$3"
  local tag args
  if [[ "$mode" == "response" ]]; then
    tag="response_r1p3_n5000_seed${seed}"
    args=(
      --mode factorized --gamma 1 --nominal-scale 1 --orthogonal-scale 0
      --response-scale 1.3
    )
  else
    tag="closed_g1p125_n5000_seed${seed}"
    args=(--mode closed --gamma 1.125)
  fi
  local output="$ROOT/x800/$tag"
  wait_for_gpu "$gpu"
  echo "[$(date --iso-8601=seconds)] $tag GPU=$gpu"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" --other-checkpoint "$X800" \
    "${args[@]}" --num-samples 5000 --global-seed "$seed" \
    --cuda-visible-devices "$gpu" --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 8192 --output-dir "$output" \
    >"$LOG_DIR/${tag}.log" 2>&1
}

run_condition 0 0 response & pid0=$!
run_condition 1 0 closed & pid1=$!
run_condition 2 1 response & pid2=$!
run_condition 3 1 closed & pid3=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
wait "$pid2" || status=$?
wait "$pid3" || status=$?
[[ "$status" == "0" ]] || exit "$status"

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/FORMAL_COMPLETE"
echo "[$(date --iso-8601=seconds)] response guidance formal comparison complete"
