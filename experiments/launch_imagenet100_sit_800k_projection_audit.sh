#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sit-800k-projection}"

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/finite_guidance_800k_projection_audit_v1}"
COMPACT_ROOT="${COMPACT_ROOT:-$BASE/finite_guidance_800k_compact_replication_v1}"
PAIR_ROOT="$COMPACT_ROOT/seed0/x_pair"
LOG_DIR="$ROOT/logs"
V_RUN="$BASE/runs/sit-s-2_seed0/checkpoints"
V800="$V_RUN/step_00800000.pt"
V500="$V_RUN/step_00500000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
WAIT_FOR_FREE_GPUS="${WAIT_FOR_FREE_GPUS:-0}"
TWO_GPU_MODE="${TWO_GPU_MODE:-0}"

mkdir -p "$LOG_DIR"
for path in "$V800" "$V500" "$X800" "$REFERENCE"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done
for directory in "$PAIR_ROOT/static_s0" "$PAIR_ROOT/static_sm1"; do
  [[ -d "$directory" ]] || { echo "Missing paired compact artifact: $directory" >&2; exit 2; }
done

if [[ "$WAIT_FOR_FREE_GPUS" == "1" ]]; then
  while true; do
    busy_gpus=()
    gpu_ids=("$GPU0" "$GPU1")
    if [[ "$TWO_GPU_MODE" != "1" ]]; then
      gpu_ids+=("$GPU2" "$GPU3")
    fi
    for gpu in "${gpu_ids[@]}"; do
      running_gpu_pids="$(
        nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits
      )"
      if [[ -n "${running_gpu_pids//[[:space:]]/}" ]]; then
        busy_gpus+=("$gpu")
      fi
    done
    [[ "${#busy_gpus[@]}" == "0" ]] && break
    echo "[$(date --iso-8601=seconds)] waiting for GPUs: ${busy_gpus[*]}"
    sleep 60
  done
fi

# The generic comparison summarizer expects this conventional path. Reuse the
# already paired v500 full-guidance artifact instead of sampling it again.
if [[ ! -e "$ROOT/same_target_v500" ]]; then
  ln -s "$COMPACT_ROOT/seed0/vweak_closed" "$ROOT/same_target_v500"
fi

run_decomposition() {
  local family="$1"
  local mode="$2"
  local gpu="$3"
  local other
  local output
  local mismatch_args=()
  if [[ "$family" == "x800" ]]; then
    other="$X800"
    output="$ROOT/$mode"
  else
    other="$V500"
    output="$ROOT/v500_direction_decomposition/$mode"
    mismatch_args+=(--allow-step-mismatch)
  fi
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$other" \
    "${mismatch_args[@]}" \
    --control-mode "$mode" \
    --scales -1 \
    --reference "$REFERENCE" \
    --output-root "$output" \
    --sampling-cuda-visible-devices "$gpu" \
    --fid-cuda-visible-devices "$gpu" \
    --per-rank-batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 8192 \
    --memory-poll-interval 0.25 \
    --global-seed 0 \
    >"$LOG_DIR/${family}_${mode}.log" 2>&1
}

echo "[$(date --iso-8601=seconds)] phase 1: parallel/orthogonal FID-5K" | tee "$LOG_DIR/master.log"
status=0
if [[ "$TWO_GPU_MODE" == "1" ]]; then
  (run_decomposition x800 parallel_pair "$GPU0"; run_decomposition v500 parallel_pair "$GPU0") &
  pid_lane0=$!
  (run_decomposition x800 orthogonal_pair "$GPU1"; run_decomposition v500 orthogonal_pair "$GPU1") &
  pid_lane1=$!
  wait "$pid_lane0" || status=$?
  wait "$pid_lane1" || status=$?
else
  run_decomposition x800 parallel_pair "$GPU0" &
  pid_x_parallel=$!
  run_decomposition x800 orthogonal_pair "$GPU1" &
  pid_x_orthogonal=$!
  run_decomposition v500 parallel_pair "$GPU2" &
  pid_v_parallel=$!
  run_decomposition v500 orthogonal_pair "$GPU3" &
  pid_v_orthogonal=$!
  for pid in "$pid_x_parallel" "$pid_x_orthogonal" "$pid_v_parallel" "$pid_v_orthogonal"; do
    wait "$pid" || status=$?
  done
fi
[[ "$status" == "0" ]] || exit "$status"

COMMON_ROOT="$ROOT/common_unique_x800_v500"
run_component() {
  local component="$1"
  local gpu="$2"
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$X800" \
    --reference-checkpoint "$V500" \
    --common-unique-component "$component" \
    --allow-reference-step-mismatch \
    --scales 1 \
    --reference "$REFERENCE" \
    --output-root "$COMMON_ROOT/$component" \
    --sampling-cuda-visible-devices "$gpu" \
    --fid-cuda-visible-devices "$gpu" \
    --per-rank-batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 8192 \
    --memory-poll-interval 0.25 \
    --global-seed 0 \
    >"$LOG_DIR/${component}.log" 2>&1
}

echo "[$(date --iso-8601=seconds)] phase 2: reciprocal common/unique FID-5K" | tee -a "$LOG_DIR/master.log"
status=0
if [[ "$TWO_GPU_MODE" == "1" ]]; then
  (run_component x_common_on_v "$GPU0"; run_component v_common_on_x "$GPU0") &
  pid_lane0=$!
  (run_component x_unique_to_v "$GPU1"; run_component v_unique_to_x "$GPU1") &
  pid_lane1=$!
  wait "$pid_lane0" || status=$?
  wait "$pid_lane1" || status=$?
else
  run_component x_common_on_v "$GPU0" &
  pid_x_common=$!
  run_component x_unique_to_v "$GPU1" &
  pid_x_unique=$!
  run_component v_common_on_x "$GPU2" &
  pid_v_common=$!
  run_component v_unique_to_x "$GPU3" &
  pid_v_unique=$!
  for pid in "$pid_x_common" "$pid_x_unique" "$pid_v_common" "$pid_v_unique"; do
    wait "$pid" || status=$?
  done
fi
[[ "$status" == "0" ]] || exit "$status"

echo "[$(date --iso-8601=seconds)] phase 3: paired field geometry and summaries" | tee -a "$LOG_DIR/master.log"
env CUDA_VISIBLE_DEVICES="$GPU0" OMP_NUM_THREADS=1 \
  python experiments/analyze_imagenet100_sit_400k_direction_geometry.py \
    --anchor "$V800" \
    --x-other "$X800" \
    --v-other "$V500" \
    --anchor-label v800 \
    --x-label x800 \
    --v-label v500 \
    --samples 512 \
    --batch-size 16 \
    --output-dir "$ROOT/direction_geometry_x800_v500" \
    --device cuda:0 \
    >"$LOG_DIR/direction_geometry_x800_v500.log" 2>&1

python experiments/summarize_imagenet100_sit_future_common_unique.py \
  --root "$COMMON_ROOT" \
  --audit-root "$ROOT" \
  --pair-root "$PAIR_ROOT" \
  --anchor-label v800 \
  --x-label x800 \
  --v-label v500 \
  >"$LOG_DIR/common_unique_summary.log" 2>&1

python experiments/summarize_imagenet100_sit_400k_direction_comparison.py \
  --audit-root "$ROOT" \
  --pair-root "$PAIR_ROOT" \
  --output-dir "$ROOT/direction_comparison_x800_v500" \
  --anchor-label v800 \
  --x-label x800 \
  --v-label v500 \
  >"$LOG_DIR/direction_comparison.log" 2>&1

touch "$ROOT/COMPLETE"
echo "[$(date --iso-8601=seconds)] complete: $ROOT" | tee -a "$LOG_DIR/master.log"
