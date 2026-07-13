#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu}
CONFIG=${CONFIG:-$EQVAE_ROOT/experiments/configs/finetune_ditdh_s_dinov2_adapter_from_official_ep5.yaml}
DATA_PATH=${DATA_PATH:-$EQVAE_SHARED_DATA_ROOT/imagenet-1k}
RESULTS_DIR=${RESULTS_DIR:-$EQVAE_STAGE2_TRAINING}
SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$SAMPLE_ROOT/logs}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-6255}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
NPROC=${NPROC:-4}

RUN_DIR="$RESULTS_DIR/$RUN_NAME"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}_train.log"
CHAIN_LOG="$LOG_DIR/${RUN_NAME}_chain.log"
FINAL_CHECKPOINT="$RUN_DIR/checkpoints/step-$(printf '%07d' "$MAX_TRAIN_STEPS").pt"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "FINETUNE_CHAIN_EXIT $? $(date)"' EXIT

echo "FINETUNE_CHAIN_START $(date)"
echo "RUN_NAME $RUN_NAME"
echo "CONFIG $CONFIG"
echo "DATA_PATH $DATA_PATH"
echo "RESULTS_DIR $RESULTS_DIR"
echo "TRAIN_GPUS $TRAIN_GPUS"
echo "NPROC $NPROC"
echo "MAX_TRAIN_STEPS $MAX_TRAIN_STEPS"

if [ ! -s "$FINAL_CHECKPOINT" ]; then
  echo "FINETUNE_TRAIN_START $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  EXPERIMENT_NAME="$RUN_NAME" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$NPROC" src/train.py \
      --config "$CONFIG" \
      --data-path "$DATA_PATH" \
      --results-dir "$RESULTS_DIR" \
      --image-size 256 \
      --precision fp32 \
      --max-train-steps "$MAX_TRAIN_STEPS" > "$TRAIN_LOG" 2>&1
  echo "FINETUNE_TRAIN_DONE $(date)"
else
  echo "FINETUNE_FINAL_CHECKPOINT_EXISTS $FINAL_CHECKPOINT"
fi

if [ ! -s "$FINAL_CHECKPOINT" ] && [ -s "$RUN_DIR/checkpoints/ep-last.pt" ]; then
  echo "FINETUNE_LINK_EP_LAST_TO_STEP $FINAL_CHECKPOINT $(date)"
  ln -f "$RUN_DIR/checkpoints/ep-last.pt" "$FINAL_CHECKPOINT" || cp -f "$RUN_DIR/checkpoints/ep-last.pt" "$FINAL_CHECKPOINT"
fi

if [ ! -s "$FINAL_CHECKPOINT" ]; then
  echo "FINETUNE_FINAL_CHECKPOINT_MISSING $FINAL_CHECKPOINT"
  exit 2
fi
