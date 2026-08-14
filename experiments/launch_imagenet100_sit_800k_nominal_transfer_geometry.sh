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
    local output="$ROOT/geometry/${family}_seed${seed}"
    if [[ -f "$output/manifest.json" \
       && -f "$output/nominal_transfer_by_time.csv" \
       && -f "$output/segment_transfer_by_time.csv" \
       && -f "$output/endpoint_latents.pt" ]]; then
      echo "[$(date --iso-8601=seconds)] reuse $family seed=$seed"
      continue
    fi
    wait_for_gpu "$gpu"
    echo "[$(date --iso-8601=seconds)] start $family seed=$seed GPU=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      experiments/run_imagenet100_sit_nominal_transfer_geometry.py \
      --anchor-checkpoint "$V800" \
      --other-checkpoint "$checkpoint" \
      "${mismatch[@]}" \
      --num-samples 512 \
      --batch-size 8 \
      --seed "$seed" \
      --gamma 1 \
      --output-dir "$output" \
      --device cuda:0 \
      >"$LOG_DIR/${family}_seed${seed}.log" 2>&1
    echo "[$(date --iso-8601=seconds)] complete $family seed=$seed"
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

touch "$ROOT/GEOMETRY_COMPLETE"
echo "[$(date --iso-8601=seconds)] all geometry jobs complete"
