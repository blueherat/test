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
GPU="${GPU:-3}"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$V500" "$X800"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

wait_for_geometry() {
  while [[ ! -f "$ROOT/GEOMETRY_COMPLETE" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for 800K geometry"
    sleep 60
  done
}

wait_for_gpu() {
  while true; do
    local pids
    pids="$(
      nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
        | tr -d '[:space:]'
    )"
    if [[ -z "$pids" ]]; then
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] waiting for GPU $GPU (pids=$pids)"
    sleep 60
  done
}

run_intervention() {
  local family="$1"
  local checkpoint="$2"
  local mode="$3"
  local samples="$4"
  local seed="$5"
  local mismatch=()
  [[ "$family" == "v500" ]] && mismatch+=(--allow-step-mismatch)
  local output="$ROOT/causal_screen/${family}/${mode}_n${samples}_seed${seed}"
  wait_for_gpu
  echo "[$(date --iso-8601=seconds)] intervention $family $mode n=$samples seed=$seed"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$checkpoint" \
    "${mismatch[@]}" \
    --mode "$mode" \
    --num-samples "$samples" \
    --global-seed "$seed" \
    --cuda-visible-devices "$GPU" \
    --output-dir "$output" \
    >"$LOG_DIR/causal_${family}_${mode}_n${samples}_seed${seed}.log" 2>&1
}

run_donor() {
  local family="$1"
  local checkpoint="$2"
  local mode="$3"
  local samples="$4"
  local seed="$5"
  local mismatch=()
  [[ "$family" == "v500" ]] && mismatch+=(--allow-step-mismatch)
  local output="$ROOT/causal_screen/${family}/donor_${mode}_n${samples}_seed${seed}"
  wait_for_gpu
  echo "[$(date --iso-8601=seconds)] donor $family $mode n=$samples seed=$seed"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_donor_fid.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$checkpoint" \
    "${mismatch[@]}" \
    --donor-mode "$mode" \
    --num-samples "$samples" \
    --global-seed "$seed" \
    --cuda-visible-devices "$GPU" \
    --output-dir "$output" \
    >"$LOG_DIR/donor_${family}_${mode}_n${samples}_seed${seed}.log" 2>&1
}

wait_for_geometry
"$PYTHON_BIN" experiments/summarize_imagenet100_sit_nominal_transfer_geometry.py \
  --root "$ROOT" \
  --families x800,v500 \
  --seeds 0,1 \
  >"$LOG_DIR/nominal_transfer_summary.log" 2>&1

for seed in 0 1; do
  run_intervention x800 "$X800" replay 5000 "$seed"
  run_intervention v500 "$V500" replay 5000 "$seed"
done

for family in x800 v500; do
  checkpoint="$X800"
  [[ "$family" == "v500" ]] && checkpoint="$V500"
  for mode in frozen gain_only direction_only closed; do
    run_intervention "$family" "$checkpoint" "$mode" 1000 0
  done
  for mode in paired same_noise_other_class other_noise_same_class other_noise_other_class; do
    run_donor "$family" "$checkpoint" "$mode" 1000 0
  done
done

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_nominal_transfer_causal.py \
  --root "$ROOT" \
  --replay-seeds 0,1 \
  --screen-seed 0 \
  >"$LOG_DIR/nominal_transfer_causal_summary.log" 2>&1

touch "$ROOT/CAUSAL_SCREEN_COMPLETE"
echo "[$(date --iso-8601=seconds)] causal screen complete"
