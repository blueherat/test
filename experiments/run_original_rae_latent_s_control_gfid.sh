#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-ditdh_s_dinov2_original_rae_latent_control_s20000_4gpu}
CONFIG=${CONFIG:-$EQVAE_ROOT/experiments/configs/rae_stage2_ditdh_s_dinov2_imagenet256_parquet.yaml}
DATA_PATH=${DATA_PATH:-$EQVAE_SHARED_DATA_ROOT/imagenet-1k}
RESULTS_DIR=${RESULTS_DIR:-$EQVAE_STAGE2_TRAINING}
SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$SAMPLE_ROOT/logs}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
SAMPLE_BATCH=${SAMPLE_BATCH:-16}
SAMPLE_FOLDER=${SAMPLE_FOLDER:-${RUN_NAME}_n50000_adm}
REF=${REF:-$EQVAE_ADM_REF}
INCEPTION=${INCEPTION:-$EQVAE_ADM_INCEPTION}
ADM_PY=${ADM_PY:-$EQVAE_ADM_PY}

RUN_DIR="$RESULTS_DIR/$RUN_NAME"
CHAIN_LOG="$LOG_DIR/${RUN_NAME}_chain.log"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}_train.log"
SAMPLE_LOG="$LOG_DIR/${SAMPLE_FOLDER}.log"
ADM_LOG="$LOG_DIR/${SAMPLE_FOLDER}_adm_fid.log"
SAMPLE_CFG="$RUN_DIR/sampling_step${MAX_TRAIN_STEPS}.yaml"
CHECKPOINT="$RUN_DIR/checkpoints/step-$(printf '%07d' "$MAX_TRAIN_STEPS").pt"
SAMPLE_NPZ="$SAMPLE_ROOT/${SAMPLE_FOLDER}.npz"
OUT_JSON="$SAMPLE_ROOT/${SAMPLE_FOLDER}_adm_fid.json"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$SAMPLE_ROOT"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "CONTROL_CHAIN_EXIT $? $(date)"' EXIT

echo "CONTROL_CHAIN_START $(date)"
echo "RUN_NAME $RUN_NAME"
echo "CONFIG $CONFIG"
echo "DATA_PATH $DATA_PATH"
echo "RESULTS_DIR $RESULTS_DIR"

if [ ! -s "$CHECKPOINT" ]; then
  echo "CONTROL_TRAIN_START $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  EXPERIMENT_NAME="$RUN_NAME" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$NPROC" src/train.py \
      --config "$CONFIG" \
      --data-path "$DATA_PATH" \
      --results-dir "$RESULTS_DIR" \
      --image-size 256 \
      --precision fp32 \
      --max-train-steps "$MAX_TRAIN_STEPS" > "$TRAIN_LOG" 2>&1
  echo "CONTROL_TRAIN_DONE $(date)"
else
  echo "CONTROL_CHECKPOINT_EXISTS $CHECKPOINT"
fi

if [ ! -s "$CHECKPOINT" ]; then
  echo "CONTROL_CHECKPOINT_MISSING $CHECKPOINT"
  exit 2
fi

if [ ! -s "$SAMPLE_CFG" ]; then
  echo "CONTROL_SAMPLING_CONFIG_START $(date)"
  cd "$EQVAE_ROOT"
  python experiments/make_rae_sampling_config.py \
    --base-config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$SAMPLE_CFG" \
    --guidance-scale 1.0 \
    --num-steps 50
  echo "CONTROL_SAMPLING_CONFIG_DONE $SAMPLE_CFG"
fi

if [ ! -s "$SAMPLE_NPZ" ]; then
  echo "CONTROL_SAMPLE_START batch=$SAMPLE_BATCH $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" SAVE_FOLDER="$SAMPLE_FOLDER" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$NPROC" src/sample_ddp.py \
      --config "$SAMPLE_CFG" \
      --sample-dir "$SAMPLE_ROOT" \
      --per-proc-batch-size "$SAMPLE_BATCH" \
      --num-fid-samples 50000 \
      --global-seed 0 \
      --precision fp32 \
      --label-sampling equal >> "$SAMPLE_LOG" 2>&1
  echo "CONTROL_SAMPLE_DONE $(date)"
else
  echo "CONTROL_SAMPLE_NPZ_EXISTS $SAMPLE_NPZ"
fi

if [ ! -s "$OUT_JSON" ]; then
  echo "CONTROL_ADM_START $(date)"
  cd "$EQVAE_ROOT"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$ADM_PY" experiments/compute_adm_fid.py \
    --reference "$REF" \
    --samples "$SAMPLE_NPZ" \
    --batch-size 64 \
    --inception-path "$INCEPTION" \
    --output "$OUT_JSON" > "$ADM_LOG" 2>&1
  echo "CONTROL_ADM_DONE $OUT_JSON $(date)"
else
  echo "CONTROL_ADM_JSON_EXISTS $OUT_JSON"
fi
