#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/joint_response_direction_n1000_seed0}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
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

float_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

run_job() {
  local gpu="$1" rho="$2" lambda="$3"
  local tag="joint_r$(float_tag "$rho")_l$(float_tag "$lambda")"
  local output="$ROOT/x800/${tag}_n${SAMPLES}_seed${SEED}"
  echo "[$(date --iso-8601=seconds)] $tag GPU=$gpu"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" --other-checkpoint "$X800" \
    --mode factorized --gamma 1 --nominal-scale 1 \
    --orthogonal-scale "$lambda" --response-scale "$rho" \
    --num-samples "$SAMPLES" --global-seed "$SEED" \
    --cuda-visible-devices "$gpu" --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 8192 --output-dir "$output" \
    >"$LOG_DIR/${tag}.log" 2>&1
}

worker_2() {
  wait_for_gpu 2
  for rho in 1.2 1.3 1.4; do
    for lambda in 0.25 0.5; do
      run_job 2 "$rho" "$lambda"
    done
  done
}

worker_3() {
  wait_for_gpu 3
  for rho in 1.2 1.3 1.4; do
    for lambda in 0.75 1.0; do
      run_job 3 "$rho" "$lambda"
    done
  done
}

worker_2 & pid2=$!
worker_3 & pid3=$!
status=0
wait "$pid2" || status=$?
wait "$pid3" || status=$?
[[ "$status" == "0" ]] || exit "$status"

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/JOINT_SCREEN_COMPLETE"
echo "[$(date --iso-8601=seconds)] response-direction joint screen complete"
