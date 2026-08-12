#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

BASE=/home/zhoushunyu/data/eqvae/imagenet_sit_flow
V_RUN="$BASE/runs/sit-s-2_seed0"
X_RUN="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0"
V_FID="$BASE/fid5k_continuation_v_900k_1000k_seed0"
X_FID="$BASE/fid5k_continuation_x_velocity_floor0p05_500k_1000k_seed0"
REFERENCE="$BASE/fid5k/sit-s-2_step100000_seed0/reference_imagenet100_validation_n5000.npz"
LOG_DIR="$BASE/logs/sit_v_x_to1000k"
V_DONE="$V_RUN/COMPLETE_1000K_TRAINING"
X_DONE="$X_RUN/COMPLETE_1000K_TRAINING"
PIPELINE_DONE="$BASE/COMPLETE_SIT_V_X_TO_1000K_WITH_FID"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
STATE_DIR="$LOG_DIR/state_$RUN_ID"
V_STATUS="$STATE_DIR/v_training.status"
X_STATUS="$STATE_DIR/x_training.status"
X_EVAL_STATUS="$STATE_DIR/x_evaluation.status"
V_EVAL_STATUS="$STATE_DIR/v_evaluation.status"

mkdir -p "$LOG_DIR" "$STATE_DIR"

# Completion markers are recreated only by successful producers in this run.
rm -f \
  "$V_DONE" \
  "$X_DONE" \
  "$X_FID/COMPLETE_500K_TO_1000K_WITH_FID" \
  "$V_FID/COMPLETE_900K_TO_1000K_WITH_FID" \
  "$PIPELINE_DONE"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 2
  fi
}

require_file "$V_RUN/checkpoints/step_00800000.pt"
require_file "$X_RUN/checkpoints/step_00400000.pt"
require_file "$REFERENCE"

train_v() {
  echo "[$(date --iso-8601=seconds)] resume SiT-v 800K -> 1000K on GPUs 0,1,2,3"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  SIT_MODEL=SiT-S/2 \
  PREDICTION_TARGET=velocity \
  LOSS_SPACE=velocity \
  DENOMINATOR_FLOOR=0.001 \
  GLOBAL_BATCH_SIZE=256 \
  MAX_STEPS=1000000 \
  SAVE_EVERY=100000 \
  SEED=0 \
  OUTPUT_DIR="$V_RUN" \
  TORCHINDUCTOR_CACHE_DIR="$BASE/torchinductor_cache" \
    bash experiments/run_imagenet100_sit_4gpu.sh \
    2>&1 | tee -a "$V_RUN/train_800k_to_1000k.log"
  touch "$V_DONE"
  echo "[$(date --iso-8601=seconds)] SiT-v training complete"
}

train_x() {
  echo "[$(date --iso-8601=seconds)] resume JiT-style x 400K -> 1000K on GPUs 2,3"
  CUDA_VISIBLE_DEVICES=2,3 \
  SIT_MODEL=SiT-S/2 \
  PREDICTION_TARGET=x \
  LOSS_SPACE=velocity \
  DENOMINATOR_FLOOR=0.05 \
  GLOBAL_BATCH_SIZE=256 \
  MAX_STEPS=1000000 \
  SAVE_EVERY=100000 \
  SEED=0 \
  OUTPUT_DIR="$X_RUN" \
  TORCHINDUCTOR_CACHE_DIR=/tmp/eqvae_inductor_jit_x_2gpu \
    bash experiments/run_imagenet100_sit_4gpu.sh \
    2>&1 | tee -a "$X_RUN/train_400k_to_1000k.log"
  touch "$X_DONE"
  echo "[$(date --iso-8601=seconds)] JiT-style x training complete"
}

write_status() {
  local path="$1"
  local status="$2"
  printf '%s\n' "$status" >"$path.tmp"
  mv "$path.tmp" "$path"
}

run_with_status() {
  local status_path="$1"
  shift
  set +e
  (
    set -euo pipefail
    "$@"
  )
  local status=$?
  write_status "$status_path" "$status"
  return "$status"
}

wait_for_file() {
  local path="$1"
  local producer_status="$2"
  local description="$3"
  while [[ ! -f "$path" ]]; do
    if [[ -f "$producer_status" ]]; then
      local status
      status="$(<"$producer_status")"
      if (( status != 0 )); then
        echo "$description cannot be produced: producer exited with $status" >&2
      else
        echo "$description is missing after its producer exited successfully" >&2
      fi
      return 1
    fi
    sleep 15
  done
}

evaluate_x_milestones() {
  wait_for_file "$V_DONE" "$V_STATUS" "SiT-v 1000K completion marker"
  for step in 500000 600000 700000 800000 900000 1000000; do
    checkpoint="$X_RUN/checkpoints/step_$(printf '%08d' "$step").pt"
    wait_for_file "$checkpoint" "$X_STATUS" "JiT-style x checkpoint $step"
    echo "[$(date --iso-8601=seconds)] evaluate JiT-style x step $step on GPUs 0,1"
    python experiments/run_imagenet100_sit_fid_curve.py \
      --steps "$step" \
      --run-dir "$X_RUN" \
      --output-root "$X_FID" \
      --reference "$REFERENCE" \
      --sampling-cuda-visible-devices 0,1 \
      --per-rank-batch-size 64 \
      --vae-decode-batch-size 4 \
      --cuda-allocator-limit-gib 7.5 \
      --fid-cuda-visible-devices 0 \
      --fid-batch-size 8 \
      --fid-gpu-memory-fraction 0.30 \
      --gpu-memory-ceiling-mib 9216 \
      --memory-poll-interval 0.25 \
      2>&1 | tee -a "$LOG_DIR/x_fid5k.log"
  done
  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps 500000,600000,700000,800000,900000,1000000 \
    --run-dir "$X_RUN" \
    --output-root "$X_FID" \
    --reference "$REFERENCE" \
    --sampling-cuda-visible-devices 0,1 \
    --fid-cuda-visible-devices 0 \
    2>&1 | tee -a "$LOG_DIR/x_fid5k.log"
  touch "$X_FID/COMPLETE_500K_TO_1000K_WITH_FID"
}

evaluate_v_milestones() {
  wait_for_file "$V_DONE" "$V_STATUS" "SiT-v 1000K completion marker"
  wait_for_file "$X_DONE" "$X_STATUS" "JiT-style x 1000K completion marker"
  wait_for_file \
    "$X_FID/COMPLETE_500K_TO_1000K_WITH_FID" \
    "$X_EVAL_STATUS" \
    "JiT-style x FID completion marker"
  echo "[$(date --iso-8601=seconds)] evaluate SiT-v 900K and 1000K on four GPUs"
  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps 900000,1000000 \
    --run-dir "$V_RUN" \
    --output-root "$V_FID" \
    --reference "$REFERENCE" \
    --sampling-cuda-visible-devices 0,1,2,3 \
    --per-rank-batch-size 64 \
    --vae-decode-batch-size 4 \
    --cuda-allocator-limit-gib 7.5 \
    --fid-cuda-visible-devices 0 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.30 \
    --gpu-memory-ceiling-mib 9216 \
    --memory-poll-interval 0.25 \
    2>&1 | tee -a "$LOG_DIR/v_fid5k.log"
  touch "$V_FID/COMPLETE_900K_TO_1000K_WITH_FID"
}

run_with_status "$V_STATUS" train_v >"$LOG_DIR/v_training_driver.log" 2>&1 &
V_PID=$!
sleep 5
run_with_status "$X_STATUS" train_x >"$LOG_DIR/x_training_driver.log" 2>&1 &
X_PID=$!
run_with_status "$X_EVAL_STATUS" evaluate_x_milestones \
  >"$LOG_DIR/x_evaluation_driver.log" 2>&1 &
X_EVAL_PID=$!
run_with_status "$V_EVAL_STATUS" evaluate_v_milestones \
  >"$LOG_DIR/v_evaluation_driver.log" 2>&1 &
V_EVAL_PID=$!

echo "[$(date --iso-8601=seconds)] launched v=$V_PID x=$X_PID x_eval=$X_EVAL_PID v_eval=$V_EVAL_PID"

status=0
for child_pid in "$V_PID" "$X_PID" "$X_EVAL_PID" "$V_EVAL_PID"; do
  if wait "$child_pid"; then
    continue
  else
    child_status=$?
    if (( status == 0 )); then
      status=$child_status
    fi
  fi
done

if (( status != 0 )); then
  echo "[$(date --iso-8601=seconds)] pipeline failed with status $status" >&2
  exit "$status"
fi

touch "$PIPELINE_DONE"
echo "[$(date --iso-8601=seconds)] all training and FID milestones complete"
