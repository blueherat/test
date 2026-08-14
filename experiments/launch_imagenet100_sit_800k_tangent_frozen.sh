#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/tangent_frozen_800k_v1}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
V500="$BASE/runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
GPU_X="${GPU_X:-0}"
GPU_V="${GPU_V:-1}"
SAMPLES="${SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
HEUN_STEPS="${HEUN_STEPS:-100}"
SEED="${SEED:-20260814}"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$X800" "$V500"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

run_family() {
  local family="$1"
  local gpu="$2"
  echo "[$(date --iso-8601=seconds)] start family=$family gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
    experiments/run_imagenet100_sit_finite_guidance.py \
    --study tangent_frozen \
    --direction "$family" \
    --anchor-checkpoint "$V800" \
    --x800-checkpoint "$X800" \
    --v500-checkpoint "$V500" \
    --output-root "$ROOT" \
    --num-samples "$SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --heun-steps "$HEUN_STEPS" \
    --seed "$SEED" \
    --device cuda:0 \
    >"$LOG_DIR/${family}_n${SAMPLES}_seed${SEED}.log" 2>&1
  echo "[$(date --iso-8601=seconds)] complete family=$family"
}

run_family x800 "$GPU_X" &
pid_x=$!
run_family v500 "$GPU_V" &
pid_v=$!

status=0
wait "$pid_x" || status=$?
wait "$pid_v" || status=$?
[[ "$status" == "0" ]] || exit "$status"

touch "$ROOT/FIRST_ROUND_COMPLETE"
echo "[$(date --iso-8601=seconds)] all tangent-frozen jobs complete"
