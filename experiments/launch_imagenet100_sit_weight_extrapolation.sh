#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$BASE/weight_extrapolation_v800_v500_v1}"
CHECKPOINT_ROOT="$EXPERIMENT_ROOT/checkpoints"
FID_ROOT="$EXPERIMENT_ROOT/fid1k_seed0"
DIAGNOSTIC_ROOT="$EXPERIMENT_ROOT/local_velocity_comparison"
LOG_ROOT="$EXPERIMENT_ROOT/logs"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
SCALES=(0.5 1 1.5 2 2.5 3)

mkdir -p "$CHECKPOINT_ROOT" "$FID_ROOT" "$DIAGNOSTIC_ROOT" "$LOG_ROOT"

float_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

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

"$PYTHON_BIN" experiments/build_imagenet100_sit_weight_extrapolation_checkpoints.py \
  --output-dir "$CHECKPOINT_ROOT" \
  --scales "$(IFS=,; echo "${SCALES[*]}")" \
  >"$LOG_ROOT/build_checkpoints.log" 2>&1

wait_for_gpu 0
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" \
  experiments/compare_imagenet100_sit_weight_velocity_extrapolation.py \
  --device cuda:0 \
  --output-dir "$DIAGNOSTIC_ROOT" \
  --scales "$(IFS=,; echo "${SCALES[*]}")" \
  >"$LOG_ROOT/local_velocity_comparison.log" 2>&1

run_scale() {
  local gamma="$1"
  local gpu="$2"
  local tag
  tag="$(float_tag "$gamma")"
  local checkpoint="$CHECKPOINT_ROOT/v800_plus_g${tag}_v800_minus_v500_ema.pt"
  local output="$FID_ROOT/g${tag}_n${SAMPLES}_seed${SEED}"
  wait_for_gpu "$gpu"
  echo "[$(date --iso-8601=seconds)] gamma=$gamma GPU=$gpu"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_weight_extrapolation_fid1k.py \
    --checkpoint "$checkpoint" \
    --output-dir "$output" \
    --num-samples "$SAMPLES" \
    --global-seed "$SEED" \
    --cuda-visible-devices "$gpu" \
    --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 12288 \
    >"$LOG_ROOT/fid1k_g${tag}.log" 2>&1
}

(run_scale 0.5 0; run_scale 2 0) & pid0=$!
(run_scale 1 1; run_scale 2.5 1) & pid1=$!
(run_scale 1.5 3; run_scale 3 3) & pid3=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
wait "$pid3" || status=$?
[[ "$status" == "0" ]] || exit "$status"

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_weight_extrapolation.py \
  --root "$FID_ROOT" \
  --scales "${SCALES[@]}" \
  --num-samples "$SAMPLES" \
  --global-seed "$SEED" \
  >"$LOG_ROOT/summary.log" 2>&1
touch "$EXPERIMENT_ROOT/COMPLETE"
echo "[$(date --iso-8601=seconds)] weight extrapolation experiment complete"
