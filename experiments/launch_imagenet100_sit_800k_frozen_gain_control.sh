#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/frozen_gain_control_n1000_seed0}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
GPU="${GPU:-1}"
GAINS="${GAINS:-0.5 0.75 1.25 1.5 1.75 2.0 2.5 3.0}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for GPU $GPU"
  sleep 30
done

float_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

for gamma in $GAINS; do
  tag="frozen_g$(float_tag "$gamma")"
  output="$ROOT/x800/${tag}_n${SAMPLES}_seed${SEED}"
  echo "[$(date --iso-8601=seconds)] $tag GPU=$GPU"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" --other-checkpoint "$X800" \
    --mode frozen --gamma "$gamma" \
    --num-samples "$SAMPLES" --global-seed "$SEED" \
    --cuda-visible-devices "$GPU" --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 8192 --output-dir "$output" \
    >"$LOG_DIR/${tag}.log" 2>&1
done

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/FROZEN_GAIN_CONTROL_COMPLETE"
echo "[$(date --iso-8601=seconds)] frozen gain control complete"
