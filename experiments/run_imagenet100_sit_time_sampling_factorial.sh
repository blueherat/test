#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
MODE="${MODE:-all}"
V_UNIFORM_RUN="$BASE/runs/sit-s-2_seed0"
X_UNIFORM_RUN="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0"
V_LOGIT_RUN="$BASE/runs/sit-s-2_velocity-velocity-loss_t-logit-normal-m-0p8-s-0p8_seed0"
X_LOGIT_RUN="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_t-logit-normal-m-0p8-s-0p8_seed0"
FID_ROOT="$BASE/fid5k_time_sampling_factorial_seed0"
REFERENCE="$BASE/fid5k/sit-s-2_step100000_seed0/reference_imagenet100_validation_n5000.npz"
LOG_DIR="$BASE/logs/sit_time_sampling_factorial"
X_UNIFORM_STOP_MARKER="$X_UNIFORM_RUN/STOPPED_AT_STEP_00800000"
TRAINING_STATUS="$LOG_DIR/training.status"
EVALUATION_STATUS="$LOG_DIR/evaluation.status"
COMPLETE_MARKER="$BASE/COMPLETE_SIT_TIME_SAMPLING_FACTORIAL_300K"
FID_COMPLETE_MARKER="$BASE/COMPLETE_SIT_TIME_SAMPLING_FACTORIAL_FID5K"

# The historical v+Uniform run did not save 200K (it saved 180K and 210K),
# so the four-way comparison has common checkpoints at 100K and 300K only.
V_UNIFORM_EVAL_STEPS="${V_UNIFORM_EVAL_STEPS:-100000,300000}"
X_UNIFORM_EVAL_STEPS="${X_UNIFORM_EVAL_STEPS:-100000,200000,300000,400000,500000,600000,700000,800000}"
V_LOGIT_EVAL_STEPS="${V_LOGIT_EVAL_STEPS:-100000,200000,300000}"
X_LOGIT_EVAL_STEPS="${X_LOGIT_EVAL_STEPS:-100000,200000,300000}"

mkdir -p "$LOG_DIR"

require_file() {
  [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

require_file "$REFERENCE"
require_file "$V_UNIFORM_RUN/checkpoints/step_00300000.pt"
require_file "$X_UNIFORM_RUN/checkpoints/step_00400000.pt"

write_status() {
  local path="$1"
  local value="$2"
  printf '%s\n' "$value" >"$path.tmp"
  mv "$path.tmp" "$path"
}

wait_for_file() {
  local path="$1"
  local producer_status="${2:-}"
  while [[ ! -f "$path" ]]; do
    if [[ -n "$producer_status" && -f "$producer_status" ]]; then
      local status
      status="$(<"$producer_status")"
      if (( status != 0 )); then
        echo "Producer failed with status $status before creating $path" >&2
      else
        echo "Producer exited successfully without creating $path" >&2
      fi
      return 1
    fi
    sleep 15
  done
}

train_one() {
  local target="$1"
  local floor="$2"
  local output_dir="$3"
  local cache_dir="$4"
  mkdir -p "$output_dir" "$cache_dir"
  echo "[$(date --iso-8601=seconds)] train $target + JiT logit-normal to 300K on GPUs 0,1"
  CUDA_VISIBLE_DEVICES=0,1 \
  SIT_MODEL=SiT-S/2 \
  PREDICTION_TARGET="$target" \
  LOSS_SPACE=velocity \
  DENOMINATOR_FLOOR="$floor" \
  TIME_SAMPLER=logit_normal \
  TIME_LOGIT_MEAN=-0.8 \
  TIME_LOGIT_STD=0.8 \
  GLOBAL_BATCH_SIZE=256 \
  MAX_STEPS=300000 \
  SAVE_EVERY=100000 \
  SEED=0 \
  OUTPUT_DIR="$output_dir" \
  TORCHINDUCTOR_CACHE_DIR="$cache_dir" \
    bash experiments/run_imagenet100_sit_4gpu.sh \
    2>&1 | tee -a "$output_dir/train.log"
}

training_sequence() {
  train_one \
    velocity 0.001 "$V_LOGIT_RUN" "$BASE/torchinductor_cache_time_factorial_v"
  touch "$V_LOGIT_RUN/COMPLETE_300K_TRAINING"
  train_one \
    x 0.05 "$X_LOGIT_RUN" "$BASE/torchinductor_cache_time_factorial_x"
  touch "$X_LOGIT_RUN/COMPLETE_300K_TRAINING"
}

run_fid_step() {
  local run_dir="$1"
  local output_root="$2"
  local step="$3"
  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps "$step" \
    --run-dir "$run_dir" \
    --output-root "$output_root" \
    --reference "$REFERENCE" \
    --sampling-cuda-visible-devices 2 \
    --per-rank-batch-size 64 \
    --vae-decode-batch-size 4 \
    --cuda-allocator-limit-gib 7.5 \
    --fid-cuda-visible-devices 2 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.30 \
    --gpu-memory-ceiling-mib 9216 \
    --memory-poll-interval 0.25 \
    2>&1 | tee -a "$LOG_DIR/fid5k.log"
}

evaluate_series() {
  local run_dir="$1"
  local output_root="$2"
  local steps_csv="$3"
  local producer_status="${4:-}"
  IFS=',' read -ra steps <<<"$steps_csv"
  for step in "${steps[@]}"; do
    local checkpoint
    checkpoint="$run_dir/checkpoints/step_$(printf '%08d' "$step").pt"
    wait_for_file "$checkpoint" "$producer_status"
    run_fid_step "$run_dir" "$output_root" "$step"
  done
  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps "$steps_csv" \
    --run-dir "$run_dir" \
    --output-root "$output_root" \
    --reference "$REFERENCE" \
    --sampling-cuda-visible-devices 2 \
    --per-rank-batch-size 64 \
    --vae-decode-batch-size 4 \
    --cuda-allocator-limit-gib 7.5 \
    --fid-cuda-visible-devices 2 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.30 \
    --gpu-memory-ceiling-mib 9216 \
    --memory-poll-interval 0.25 \
    2>&1 | tee -a "$LOG_DIR/fid5k.log"
}

evaluation_sequence() {
  wait_for_file "$X_UNIFORM_STOP_MARKER"
  echo "[$(date --iso-8601=seconds)] GPU 2 released; start paired single-GPU FID protocol"
  evaluate_series \
    "$V_UNIFORM_RUN" "$FID_ROOT/v_uniform" "$V_UNIFORM_EVAL_STEPS"
  evaluate_series \
    "$X_UNIFORM_RUN" "$FID_ROOT/x_uniform" \
    "$X_UNIFORM_EVAL_STEPS"
  evaluate_series \
    "$V_LOGIT_RUN" "$FID_ROOT/v_logit_normal" \
    "$V_LOGIT_EVAL_STEPS" "$TRAINING_STATUS"
  evaluate_series \
    "$X_LOGIT_RUN" "$FID_ROOT/x_logit_normal" \
    "$X_LOGIT_EVAL_STEPS" "$TRAINING_STATUS"
}

case "$MODE" in
  all|training|evaluation) ;;
  *)
    echo "Unsupported MODE=$MODE (expected all, training, or evaluation)" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "training" ]]; then
  set +e
  training_sequence
  TRAIN_RESULT=$?
  set -e
  write_status "$TRAINING_STATUS" "$TRAIN_RESULT"
  exit "$TRAIN_RESULT"
fi

if [[ "$MODE" == "evaluation" ]]; then
  set +e
  evaluation_sequence
  EVAL_RESULT=$?
  set -e
  write_status "$EVALUATION_STATUS" "$EVAL_RESULT"
  if (( EVAL_RESULT != 0 )); then
    exit "$EVAL_RESULT"
  fi
  touch "$FID_COMPLETE_MARKER"
  if [[ -f "$TRAINING_STATUS" ]] && [[ "$(<"$TRAINING_STATUS")" == "0" ]]; then
    touch "$COMPLETE_MARKER"
  fi
  exit 0
fi

(
  set -euo pipefail
  training_sequence
) >"$LOG_DIR/training_driver.log" 2>&1 &
TRAIN_PID=$!
(
  set -euo pipefail
  evaluation_sequence
) >"$LOG_DIR/evaluation_driver.log" 2>&1 &
EVAL_PID=$!

echo "[$(date --iso-8601=seconds)] launched training=$TRAIN_PID evaluation=$EVAL_PID"

set +e
wait "$TRAIN_PID"
TRAIN_RESULT=$?
write_status "$TRAINING_STATUS" "$TRAIN_RESULT"
wait "$EVAL_PID"
EVAL_RESULT=$?
write_status "$EVALUATION_STATUS" "$EVAL_RESULT"
set -e

if (( TRAIN_RESULT != 0 || EVAL_RESULT != 0 )); then
  echo "factorial pipeline failed: training=$TRAIN_RESULT evaluation=$EVAL_RESULT" >&2
  exit 1
fi

touch "$FID_COMPLETE_MARKER"
touch "$COMPLETE_MARKER"
echo "[$(date --iso-8601=seconds)] time-sampling factorial complete"
