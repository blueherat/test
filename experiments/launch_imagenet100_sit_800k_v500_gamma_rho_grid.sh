#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/v500_gamma_rho_grid_n1000_seed0}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
GPU="${GPU:-3}"
GAMMAS="${GAMMAS:-1.0 1.5 2.0 2.5 3.0}"
RHOS="${RHOS:-1.0 1.05 1.10 1.15 1.20 1.25 1.30 1.35 1.40 1.45 1.50}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
V500="$BASE/runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$V500"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

free_mib="$(
  nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits \
    | tr -d '[:space:]'
)"
if (( free_mib < 10000 )); then
  echo "GPU $GPU has only ${free_mib} MiB free; require at least 10000 MiB" >&2
  exit 3
fi

float_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

for gamma in $GAMMAS; do
  for rho in $RHOS; do
    tag="response_g$(float_tag "$gamma")_r$(float_tag "$rho")"
    output="$ROOT/v500/${tag}_n${SAMPLES}_seed${SEED}"
    echo "[$(date --iso-8601=seconds)] $tag GPU=$GPU"
    "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
      --anchor-checkpoint "$V800" \
      --other-checkpoint "$V500" \
      --allow-step-mismatch \
      --mode factorized \
      --gamma "$gamma" \
      --nominal-scale 1 \
      --orthogonal-scale 0 \
      --response-scale "$rho" \
      --num-samples "$SAMPLES" \
      --global-seed "$SEED" \
      --cuda-visible-devices "$GPU" \
      --cuda-allocator-limit-gib 4 \
      --gpu-memory-ceiling-mib 8192 \
      --output-dir "$output" \
      >"$LOG_DIR/${tag}.log" 2>&1
  done
done

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" \
  >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/GRID_COMPLETE"
touch "$ROOT/FINE_GRID_COMPLETE"
echo "[$(date --iso-8601=seconds)] v500 gamma-rho grid complete"
