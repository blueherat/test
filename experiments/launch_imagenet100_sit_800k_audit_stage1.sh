#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-sit-800k-stage1}"

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
PAIR_ROOT="${PAIR_ROOT:-$BASE/fid5k_static_pair_v_to_jit_x_step800000_seed0}"
AUDIT_ROOT="${AUDIT_ROOT:-$BASE/fid5k_step800k_floor_audit_seed0}"
LOG_DIR="$AUDIT_ROOT/logs"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"

mkdir -p "$LOG_DIR"
for path in "$V800" "$X800" "$REFERENCE"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done

run_full_pair() {
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$X800" \
    --reference "$REFERENCE" \
    --output-root "$PAIR_ROOT" \
    --sampling-cuda-visible-devices 0 \
    --fid-cuda-visible-devices 0 \
    --per-rank-batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 8192 \
    --memory-poll-interval 0.25 \
    --global-seed 0 \
    >"$LOG_DIR/full_pair.log" 2>&1
}

echo "[$(date --iso-8601=seconds)] stage1 start" | tee -a "$LOG_DIR/stage1.log"

run_full_pair &
pid_pair=$!

GPU=1 GROUP=mechanism ROOT="$AUDIT_ROOT" \
  bash experiments/run_imagenet100_sit_800k_floor_audit.sh \
  >"$LOG_DIR/driver_mechanism.log" 2>&1 &
pid_mechanism=$!

(
  GPU=2 GROUP=inference_floor ROOT="$AUDIT_ROOT" \
    bash experiments/run_imagenet100_sit_800k_floor_audit.sh
  GPU=2 GROUP=autoguidance WEAK_STEPS="400 500" ROOT="$AUDIT_ROOT" \
    bash experiments/run_imagenet100_sit_800k_floor_audit.sh
) >"$LOG_DIR/driver_gpu2.log" 2>&1 &
pid_gpu2=$!

GPU=3 GROUP=autoguidance WEAK_STEPS="600 700" ROOT="$AUDIT_ROOT" \
  bash experiments/run_imagenet100_sit_800k_floor_audit.sh \
  >"$LOG_DIR/driver_gpu3.log" 2>&1 &
pid_gpu3=$!

status=0
for pid in "$pid_pair" "$pid_mechanism" "$pid_gpu2" "$pid_gpu3"; do
  wait "$pid" || status=$?
done

if [[ "$status" == "0" ]]; then
  touch "$AUDIT_ROOT/COMPLETE_STAGE1"
fi
echo "[$(date --iso-8601=seconds)] stage1 exit=$status" | tee -a "$LOG_DIR/stage1.log"
exit "$status"
