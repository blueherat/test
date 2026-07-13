#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-dinov2_adapter_imagenet_full_e2_light_r1_strict_id5_lr5e6_from_e1}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-$EQVAE_ARTIFACTS_DIR/latent_adapter/dinov2_adapter_imagenet_full_e1_light_r1_strict_id5_lr1e5/adapter.pt}
DATA_ROOT=${DATA_ROOT:-$EQVAE_SHARED_DATA_ROOT}
DATASET_PATH=${DATASET_PATH:-$EQVAE_SHARED_DATA_ROOT/imagenet-1k}
LOG_DIR=${LOG_DIR:-$EQVAE_STAGE2_SAMPLES/logs}
ADAPTER_GPUS=${ADAPTER_GPUS:-0,1,2,3}
NPROC=${NPROC:-4}

TRAIN_COUNT=${TRAIN_COUNT:-1280000}
VAL_COUNT=${VAL_COUNT:-1024}
TEST_COUNT=${TEST_COUNT:-2048}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
EPOCHS=${EPOCHS:-1}
LR=${LR:-5e-6}
IDENTITY_WEIGHT=${IDENTITY_WEIGHT:-5.0}
EQUIV_WEIGHT=${EQUIV_WEIGHT:-1.0}
INVERSE_WEIGHT=${INVERSE_WEIGHT:-1.0}
BLOCKS=${BLOCKS:-2}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-64}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-250}

LOG_PATH="$LOG_DIR/${RUN_NAME}.log"
CHAIN_LOG="$LOG_DIR/${RUN_NAME}_chain.log"
PID_PATH="$LOG_DIR/${RUN_NAME}.pid"
RUN_DIR="$EQVAE_ARTIFACTS_DIR/latent_adapter/$RUN_NAME"

mkdir -p "$LOG_DIR"
echo "$$" > "$PID_PATH"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "LIGHTADAPTER_EXIT $? $(date)"' EXIT

echo "LIGHTADAPTER_START $(date)"
echo "RUN_NAME $RUN_NAME"
echo "INIT_CHECKPOINT $INIT_CHECKPOINT"
echo "DATASET_PATH $DATASET_PATH"
echo "ADAPTER_GPUS $ADAPTER_GPUS"
echo "NPROC $NPROC"
echo "TRAIN_COUNT $TRAIN_COUNT"
echo "VAL_COUNT $VAL_COUNT"
echo "TEST_COUNT $TEST_COUNT"
echo "BATCH_SIZE $BATCH_SIZE"
echo "EPOCHS $EPOCHS"
echo "LR $LR"
echo "IDENTITY_WEIGHT $IDENTITY_WEIGHT"
echo "PROGRESS_INTERVAL $PROGRESS_INTERVAL"

if [ -s "$RUN_DIR/adapter.pt" ]; then
  echo "LIGHTADAPTER_EXISTS $RUN_DIR/adapter.pt"
  exit 0
fi

cd "$EQVAE_ROOT"
CUDA_VISIBLE_DEVICES="$ADAPTER_GPUS" PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node="$NPROC" experiments/latent_equiv_adapter.py \
    --data-root "$DATA_ROOT" \
    --dataset-name imagenet_parquet \
    --dataset-split train \
    --dataset-path "$DATASET_PATH" \
    --eval-dataset-name imagenet_parquet \
    --eval-dataset-split validation \
    --eval-dataset-path "$DATASET_PATH" \
    --image-size 256 \
    --model-key rae_dinov2 \
    --train-count "$TRAIN_COUNT" \
    --val-count "$VAL_COUNT" \
    --test-count "$TEST_COUNT" \
    --sequential-split \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --blocks "$BLOCKS" \
    --hidden-channels "$HIDDEN_CHANNELS" \
    --train-transforms flip_h flip_v rot180 \
    --eval-transform flip_h \
    --equiv-weight "$EQUIV_WEIGHT" \
    --inverse-weight "$INVERSE_WEIGHT" \
    --identity-weight "$IDENTITY_WEIGHT" \
    --init-checkpoint "$INIT_CHECKPOINT" \
    --run-name "$RUN_NAME" \
    --viz-count 8 \
    --fid-count 64 \
    --progress-interval "$PROGRESS_INTERVAL" \
    --skip-fid > "$LOG_PATH" 2>&1

echo "LIGHTADAPTER_DONE $(date)"
echo "CHECKPOINT $RUN_DIR/adapter.pt"
