#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/nominal_guidance_transfer_800k_v1}"
V_RUN="$BASE/runs/sit-s-2_seed0/checkpoints"
V800="$V_RUN/step_00800000.pt"
V500="$V_RUN/step_00500000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
GPU_X="${GPU_X:-2}"
GPU_V="${GPU_V:-3}"
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
    sleep 60
  done
}

run_family() {
  local family="$1"
  local checkpoint="$2"
  local gpu="$3"
  local mismatch=()
  [[ "$family" == "v500" ]] && mismatch+=(--allow-step-mismatch)
  for seed in 0 1; do
    for mode in gain_only direction_only; do
      local output="$ROOT/causal_formal/${family}/${mode}_n5000_seed${seed}"
      wait_for_gpu "$gpu"
      echo "[$(date --iso-8601=seconds)] formal $family $mode seed=$seed GPU=$gpu"
      "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
        --anchor-checkpoint "$V800" \
        --other-checkpoint "$checkpoint" \
        "${mismatch[@]}" \
        --mode "$mode" \
        --gamma 1 \
        --num-samples 5000 \
        --global-seed "$seed" \
        --cuda-visible-devices "$gpu" \
        --output-dir "$output" \
        >"$LOG_DIR/formal_${family}_${mode}_seed${seed}.log" 2>&1
    done
  done
}

run_family x800 "$X800" "$GPU_X" &
pid_x=$!
run_family v500 "$V500" "$GPU_V" &
pid_v=$!
status=0
wait "$pid_x" || status=$?
wait "$pid_v" || status=$?
[[ "$status" == "0" ]] || exit "$status"

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_nominal_transfer_causal.py \
  --root "$ROOT" \
  --replay-seeds 0,1 \
  --formal-seeds 0,1 \
  --screen-seed 0 \
  >"$LOG_DIR/nominal_transfer_formal_summary.log" 2>&1

touch "$ROOT/FORMAL_COMPLETE"
echo "[$(date --iso-8601=seconds)] formal 800K transfer study complete"
