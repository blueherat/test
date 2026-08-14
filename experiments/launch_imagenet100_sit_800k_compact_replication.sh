#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sit-800k-compact}"

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/finite_guidance_800k_compact_replication_v1}"
LOG_DIR="$ROOT/logs"
V_RUN="$BASE/runs/sit-s-2_seed0/checkpoints"
V800="$V_RUN/step_00800000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
V400="$V_RUN/step_00400000.pt"
V500="$V_RUN/step_00500000.pt"
V600="$V_RUN/step_00600000.pt"
V700="$V_RUN/step_00700000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
SELECTION="$ROOT/quality_match.json"

mkdir -p "$LOG_DIR"
for path in "$V800" "$X800" "$V400" "$V500" "$V600" "$V700" "$REFERENCE"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done

run_pair() {
  local gpu="$1"
  local seed="$2"
  local other="$3"
  local output="$4"
  local allow_mismatch="$5"
  shift 5
  local mismatch_args=()
  [[ "$allow_mismatch" == "yes" ]] && mismatch_args+=(--allow-step-mismatch)
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$other" \
    "${mismatch_args[@]}" \
    --scales "$@" \
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
    --global-seed "$seed"
}

run_frozen() {
  local gpu="$1"
  local seed="$2"
  local other="$3"
  local output="$4"
  local allow_mismatch="$5"
  local mismatch_args=()
  [[ "$allow_mismatch" == "yes" ]] && mismatch_args+=(--allow-step-mismatch)
  python experiments/run_imagenet100_sit_frozen_guidance_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$other" \
    "${mismatch_args[@]}" \
    --gamma 1 \
    --output-dir "$output" \
    --reference "$REFERENCE" \
    --cuda-visible-devices "$gpu" \
    --num-samples 5000 \
    --batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --global-seed "$seed" \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 8192 \
    --memory-poll-interval 0.25
}

echo "[$(date --iso-8601=seconds)] phase 0: endpoint quality matching" | tee "$LOG_DIR/master.log"
run_pair 0 0 "$X800" "$ROOT/selection/x800_endpoint" no 1 \
  >"$LOG_DIR/x800_endpoint.log" 2>&1 &
pid_x=$!
run_pair 1 0 "$V500" "$ROOT/selection/v500_endpoint" yes 1 \
  >"$LOG_DIR/v500_endpoint.log" 2>&1 &
pid_v500=$!
run_pair 2 0 "$V600" "$ROOT/selection/v600_endpoint" yes 1 \
  >"$LOG_DIR/v600_endpoint.log" 2>&1 &
pid_v600=$!
(
  run_pair 3 0 "$V400" "$ROOT/selection/v400_endpoint" yes 1
  run_pair 3 0 "$V700" "$ROOT/selection/v700_endpoint" yes 1
) >"$LOG_DIR/v400_v700_endpoints.log" 2>&1 &
pid_v400_v700=$!
status=0
wait "$pid_x" || status=$?
wait "$pid_v500" || status=$?
wait "$pid_v600" || status=$?
wait "$pid_v400_v700" || status=$?
[[ "$status" == "0" ]] || exit "$status"

python experiments/summarize_imagenet100_sit_800k_compact_replication.py select \
  --x-endpoint-summary "$ROOT/selection/x800_endpoint/field_control_fid5k.json" \
  --v-candidate "400=$ROOT/selection/v400_endpoint/field_control_fid5k.json" \
  --v-candidate "500=$ROOT/selection/v500_endpoint/field_control_fid5k.json" \
  --v-candidate "600=$ROOT/selection/v600_endpoint/field_control_fid5k.json" \
  --v-candidate "700=$ROOT/selection/v700_endpoint/field_control_fid5k.json" \
  --output "$SELECTION" \
  >"$LOG_DIR/quality_match.log" 2>&1

V_WEAK="$(jq -r '.matched.checkpoint' "$SELECTION")"
V_LABEL="$(jq -r '.matched.label' "$SELECTION")"
echo "[$(date --iso-8601=seconds)] matched weak: $V_LABEL ($V_WEAK)" | tee -a "$LOG_DIR/master.log"

run_x_lane() {
  local gpu="$1"
  local seed="$2"
  run_pair "$gpu" "$seed" "$X800" "$ROOT/seed${seed}/x_pair" no 0 -1
}

run_aux_lane() {
  local gpu="$1"
  local seed="$2"
  run_pair "$gpu" "$seed" "$V_WEAK" "$ROOT/seed${seed}/vweak_closed" yes -1
  run_frozen "$gpu" "$seed" "$X800" "$ROOT/seed${seed}/x_frozen" no
  run_frozen "$gpu" "$seed" "$V_WEAK" "$ROOT/seed${seed}/vweak_frozen" yes
}

echo "[$(date --iso-8601=seconds)] phase 1: two-seed paired closed/frozen replication" | tee -a "$LOG_DIR/master.log"
run_x_lane 0 0 >"$LOG_DIR/seed0_x_lane.log" 2>&1 &
pid_s0x=$!
run_x_lane 1 1 >"$LOG_DIR/seed1_x_lane.log" 2>&1 &
pid_s1x=$!
run_aux_lane 2 0 >"$LOG_DIR/seed0_aux_lane.log" 2>&1 &
pid_s0a=$!
run_aux_lane 3 1 >"$LOG_DIR/seed1_aux_lane.log" 2>&1 &
pid_s1a=$!

status=0
for pid in "$pid_s0x" "$pid_s1x" "$pid_s0a" "$pid_s1a"; do
  wait "$pid" || status=$?
done
[[ "$status" == "0" ]] || exit "$status"

python experiments/summarize_imagenet100_sit_800k_compact_replication.py final \
  --root "$ROOT" \
  --selection "$SELECTION" \
  >"$LOG_DIR/final_summary.log" 2>&1

touch "$ROOT/COMPLETE"
echo "[$(date --iso-8601=seconds)] complete: $ROOT" | tee -a "$LOG_DIR/master.log"
