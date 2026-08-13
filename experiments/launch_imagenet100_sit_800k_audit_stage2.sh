#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sit-800k-stage2}"

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
PAIR_ROOT="${PAIR_ROOT:-$BASE/fid5k_static_pair_v_to_jit_x_step800000_seed0}"
AUDIT_ROOT="${AUDIT_ROOT:-$BASE/fid5k_step800k_floor_audit_seed0}"
LOG_DIR="$AUDIT_ROOT/logs"
V_RUN="$BASE/runs/sit-s-2_seed0/checkpoints"
V800="$V_RUN/step_00800000.pt"
V400="$V_RUN/step_00400000.pt"
V700="$V_RUN/step_00700000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"

mkdir -p "$LOG_DIR"
for path in "$V800" "$V400" "$V700" "$X800"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done
[[ -f "$AUDIT_ROOT/COMPLETE_STAGE1" ]] || {
  echo "Stage 1 is not complete: $AUDIT_ROOT/COMPLETE_STAGE1" >&2
  exit 2
}

python experiments/summarize_imagenet100_sit_800k_mechanism_audit.py \
  --full-sweep-root "$PAIR_ROOT" \
  --audit-root "$AUDIT_ROOT" \
  >"$LOG_DIR/combined_mechanism_audit.log" 2>&1

matched_step="$(python - "$AUDIT_ROOT/combined_mechanism_audit.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
row = payload["matched_same_target_weak"]
if row is None:
    raise SystemExit("no matched same-target weak model")
print(int(row["other_checkpoint_step"]))
PY
)"
matched_k=$((matched_step / 1000))
v_label="v${matched_k}"
V_WEAK="$V_RUN/step_$(printf '%08d' "$matched_step").pt"
[[ -f "$V_WEAK" ]] || { echo "Missing matched weak checkpoint: $V_WEAK" >&2; exit 2; }
echo "[$(date --iso-8601=seconds)] matched weak=$v_label checkpoint=$V_WEAK" | tee "$LOG_DIR/stage2.log"

run_decomposition() {
  local mode="$1"
  local gpu="$2"
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$V_WEAK" \
    --allow-step-mismatch \
    --control-mode "$mode" \
    --scales -1 \
    --output-root "$AUDIT_ROOT/${v_label}_direction_decomposition/${mode}" \
    --sampling-cuda-visible-devices "$gpu" \
    --fid-cuda-visible-devices "$gpu" \
    --per-rank-batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 8192 \
    --global-seed 0 \
    >"$LOG_DIR/${v_label}_${mode}.log" 2>&1
}

run_decomposition parallel_pair 0 &
pid_parallel=$!
run_decomposition orthogonal_pair 1 &
pid_orthogonal=$!

env CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 \
  python experiments/analyze_imagenet100_sit_400k_direction_geometry.py \
    --anchor "$V800" \
    --x-other "$X800" \
    --v-other "$V_WEAK" \
    --anchor-label v800 \
    --x-label x800 \
    --v-label "$v_label" \
    --samples 512 \
    --batch-size 16 \
    --output-dir "$AUDIT_ROOT/direction_geometry_x800_${v_label}" \
    --device cuda:0 \
    >"$LOG_DIR/direction_geometry_x800_${v_label}.log" 2>&1 &
pid_geometry=$!

env CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 \
  python experiments/analyze_imagenet100_sit_future_training_direction.py \
    --anchor "$V800" \
    --training-reference "$V400" \
    --update-direction anchor_minus_reference \
    --x-other "$X800" \
    --v-other "$V_WEAK" \
    --anchor-label v800 \
    --training-reference-label v400 \
    --x-label x800 \
    --v-label "$v_label" \
    --samples 512 \
    --batch-size 16 \
    --output-dir "$AUDIT_ROOT/training_update_arrival_v800_minus_v400" \
    --device cuda:0 \
    >"$LOG_DIR/training_update_arrival_v800_minus_v400.log" 2>&1 &
pid_update_long=$!

status=0
for pid in "$pid_parallel" "$pid_orthogonal" "$pid_geometry" "$pid_update_long"; do
  wait "$pid" || status=$?
done
[[ "$status" == "0" ]] || exit "$status"

COMMON_ROOT="$AUDIT_ROOT/common_unique_x800_${v_label}"
run_component() {
  local component="$1"
  local gpu="$2"
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$X800" \
    --reference-checkpoint "$V_WEAK" \
    --common-unique-component "$component" \
    --allow-reference-step-mismatch \
    --scales 1 \
    --output-root "$COMMON_ROOT/$component" \
    --sampling-cuda-visible-devices "$gpu" \
    --fid-cuda-visible-devices "$gpu" \
    --per-rank-batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 8192 \
    --global-seed 0 \
    >"$LOG_DIR/${component}.log" 2>&1
}

run_component x_common_on_v 0 &
pid_x_common=$!
run_component x_unique_to_v 1 &
pid_x_unique=$!
run_component v_common_on_x 2 &
pid_v_common=$!
run_component v_unique_to_x 3 &
pid_v_unique=$!

status=0
for pid in "$pid_x_common" "$pid_x_unique" "$pid_v_common" "$pid_v_unique"; do
  wait "$pid" || status=$?
done
[[ "$status" == "0" ]] || exit "$status"

python experiments/summarize_imagenet100_sit_future_common_unique.py \
  --root "$COMMON_ROOT" \
  --audit-root "$AUDIT_ROOT" \
  --pair-root "$PAIR_ROOT" \
  --anchor-label v800 \
  --x-label x800 \
  --v-label "$v_label" \
  >"$LOG_DIR/common_unique_summary.log" 2>&1

python experiments/summarize_imagenet100_sit_400k_direction_comparison.py \
  --audit-root "$AUDIT_ROOT" \
  --pair-root "$PAIR_ROOT" \
  --output-dir "$AUDIT_ROOT/direction_comparison_x800_${v_label}" \
  --anchor-label v800 \
  --x-label x800 \
  --v-label "$v_label" \
  >"$LOG_DIR/direction_comparison.log" 2>&1

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  python experiments/analyze_imagenet100_sit_future_training_direction.py \
    --anchor "$V800" \
    --training-reference "$V700" \
    --update-direction anchor_minus_reference \
    --x-other "$X800" \
    --v-other "$V_WEAK" \
    --anchor-label v800 \
    --training-reference-label v700 \
    --x-label x800 \
    --v-label "$v_label" \
    --samples 512 \
    --batch-size 16 \
    --output-dir "$AUDIT_ROOT/training_update_arrival_v800_minus_v700" \
    --device cuda:0 \
    >"$LOG_DIR/training_update_arrival_v800_minus_v700.log" 2>&1

touch "$AUDIT_ROOT/COMPLETE_STAGE2"
echo "[$(date --iso-8601=seconds)] stage2 complete matched=$v_label" | tee -a "$LOG_DIR/stage2.log"
