#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/response_screen_n1000_seed0}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
V_RUN="$BASE/runs/sit-s-2_seed0/checkpoints"
V800="$V_RUN/step_00800000.pt"
V500="$V_RUN/step_00500000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$V500" "$X800"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local pids
    pids="$(
      nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
        | tr -d '[:space:]'
    )"
    if [[ -z "$pids" ]]; then
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] waiting for GPU $gpu (pids=$pids)"
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
  local family="$1"
  local checkpoint="$2"
  local gpu="$3"
  local mode="$4"
  local gamma="$5"
  local rho="$6"
  local mismatch=()
  [[ "$family" == "v500" ]] && mismatch+=(--allow-step-mismatch)
  local tag
  if [[ "$mode" == "closed" ]]; then
    tag="closed_g$(float_tag "$gamma")"
  else
    tag="response_r$(float_tag "$rho")"
  fi
  local output="$ROOT/$family/${tag}_n${SAMPLES}_seed${SEED}"
  echo "[$(date --iso-8601=seconds)] $family $tag GPU=$gpu"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$checkpoint" \
    "${mismatch[@]}" \
    --mode "$mode" \
    --gamma "$gamma" \
    --nominal-scale 1 \
    --orthogonal-scale 0 \
    --response-scale "$rho" \
    --num-samples "$SAMPLES" \
    --global-seed "$SEED" \
    --cuda-visible-devices "$gpu" \
    --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 8192 \
    --output-dir "$output" \
    >"$LOG_DIR/${family}_${tag}.log" 2>&1
}

run_response_slice() {
  local family="$1"
  local checkpoint="$2"
  local gpu="$3"
  shift 3
  local rho
  for rho in "$@"; do
    run_job "$family" "$checkpoint" "$gpu" factorized 1 "$rho"
  done
}

run_closed_slice() {
  local family="$1"
  local checkpoint="$2"
  local gpu="$3"
  shift 3
  local gamma
  for gamma in "$@"; do
    run_job "$family" "$checkpoint" "$gpu" closed "$gamma" 1
  done
}

worker_0() {
  wait_for_gpu 0
  run_response_slice x800 "$X800" 0 0.5 0.75 1
  run_closed_slice x800 "$X800" 0 1.75 2
}

worker_1() {
  wait_for_gpu 1
  run_response_slice x800 "$X800" 1 1.25 1.5 2
}

worker_2() {
  wait_for_gpu 2
  run_response_slice v500 "$V500" 2 0.5 0.75 1
  run_closed_slice v500 "$V500" 2 1.75 2 2.5
}

worker_3() {
  wait_for_gpu 3
  run_response_slice v500 "$V500" 3 1.25 1.5 2
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
  --root "$ROOT" \
  >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/SCREEN_COMPLETE"
echo "[$(date --iso-8601=seconds)] response guidance screen complete"
