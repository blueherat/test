#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/refinement_n1000_seed0}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
V_RUN="$BASE/runs/sit-s-2_seed0/checkpoints"
V800="$V_RUN/step_00800000.pt"
V500="$V_RUN/step_00500000.pt"
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
  local family="$1" checkpoint="$2" gpu="$3" mode="$4" coefficient="$5"
  local mismatch=()
  [[ "$family" == "v500" ]] && mismatch+=(--allow-step-mismatch)
  local tag args
  if [[ "$mode" == "closed" ]]; then
    tag="closed_g$(float_tag "$coefficient")"
    args=(--mode closed --gamma "$coefficient")
  else
    tag="response_r$(float_tag "$coefficient")"
    args=(
      --mode factorized --gamma 1 --nominal-scale 1 --orthogonal-scale 0
      --response-scale "$coefficient"
    )
  fi
  local output="$ROOT/$family/${tag}_n${SAMPLES}_seed${SEED}"
  echo "[$(date --iso-8601=seconds)] $family $tag GPU=$gpu"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" --other-checkpoint "$checkpoint" \
    "${mismatch[@]}" "${args[@]}" \
    --num-samples "$SAMPLES" --global-seed "$SEED" \
    --cuda-visible-devices "$gpu" --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 8192 --output-dir "$output" \
    >"$LOG_DIR/${family}_${tag}.log" 2>&1
}

worker_0() {
  wait_for_gpu 0
  for rho in 1.2 1.3 1.4; do
    run_job x800 "$X800" 0 response "$rho"
  done
}

worker_1() {
  wait_for_gpu 1
  for rho in 1.35 1.4 1.45; do
    run_job v500 "$V500" 1 response "$rho"
  done
  run_job x800 "$X800" 1 closed 1.125
}

worker_2() {
  wait_for_gpu 2
  for rho in 1.55 1.6 1.7; do
    run_job v500 "$V500" 2 response "$rho"
  done
  run_job x800 "$X800" 2 closed 1.375
}

worker_3() {
  wait_for_gpu 3
  for gamma in 3 3.5 4; do
    run_job v500 "$V500" 3 closed "$gamma"
  done
}

worker_0 & pid0=$!
worker_1 & pid1=$!
worker_2 & pid2=$!
worker_3 & pid3=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
wait "$pid2" || status=$?
wait "$pid3" || status=$?
[[ "$status" == "0" ]] || exit "$status"

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/REFINEMENT_COMPLETE"
echo "[$(date --iso-8601=seconds)] response refinement complete"
