#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu}
CONFIG=${CONFIG:-$EQVAE_ROOT/experiments/configs/finetune_ditdh_s_dinov2_adapter_from_official_ep5.yaml}
RESULTS_DIR=${RESULTS_DIR:-$EQVAE_STAGE2_TRAINING}
SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$SAMPLE_ROOT/logs}
FINAL_STEP=${FINAL_STEP:-6255}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-50000}
SAMPLE_GPUS=${SAMPLE_GPUS:-0,1,2,3}
SAMPLE_NPROC=${SAMPLE_NPROC:-4}
SAMPLE_BATCH=${SAMPLE_BATCH:-2}
CHECKPOINT_WEIGHT_KEY=${CHECKPOINT_WEIGHT_KEY:-model}
REF=${REF:-$EQVAE_ADM_REF}
INCEPTION=${INCEPTION:-$EQVAE_ADM_INCEPTION}
ADM_PY=${ADM_PY:-$EQVAE_ADM_PY}
ADM_GPU=${ADM_GPU:-1}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
NUM_STEPS=${NUM_STEPS:-50}
POLL_SECONDS=${POLL_SECONDS:-120}

RUN_DIR="$RESULTS_DIR/$RUN_NAME"
CHECKPOINT="$RUN_DIR/checkpoints/step-$(printf '%07d' "$FINAL_STEP").pt"
WEIGHT_SUFFIX=""
if [ "$CHECKPOINT_WEIGHT_KEY" != "auto" ]; then
  WEIGHT_SUFFIX="_${CHECKPOINT_WEIGHT_KEY}"
fi
SAMPLE_FOLDER="${RUN_NAME}_step$(printf '%07d' "$FINAL_STEP")${WEIGHT_SUFFIX}_n${NUM_FID_SAMPLES}_adm"
SAMPLE_CFG="$RUN_DIR/sampling_step${FINAL_STEP}${WEIGHT_SUFFIX}_n${NUM_FID_SAMPLES}.yaml"
MATERIALIZED_CHECKPOINT="$RUN_DIR/checkpoints/step-$(printf '%07d' "$FINAL_STEP")${WEIGHT_SUFFIX}_stage2.pt"
SAMPLE_NPZ="$SAMPLE_ROOT/${SAMPLE_FOLDER}.npz"
OUT_JSON="$SAMPLE_ROOT/${SAMPLE_FOLDER}_adm_fid.json"
CHAIN_LOG="$LOG_DIR/${SAMPLE_FOLDER}_chain.log"
SAMPLE_LOG="$LOG_DIR/${SAMPLE_FOLDER}.log"
ADM_LOG="$LOG_DIR/${SAMPLE_FOLDER}_adm_fid.log"
GRID_PNG="$SAMPLE_ROOT/${SAMPLE_FOLDER}_grid.png"

mkdir -p "$LOG_DIR" "$SAMPLE_ROOT"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "FINAL50K_CHAIN_EXIT $? $(date)"' EXIT

echo "FINAL50K_CHAIN_START $(date)"
echo "RUN_NAME $RUN_NAME"
echo "CONFIG $CONFIG"
echo "CHECKPOINT $CHECKPOINT"
echo "NUM_FID_SAMPLES $NUM_FID_SAMPLES"
echo "SAMPLE_GPUS $SAMPLE_GPUS"
echo "SAMPLE_NPROC $SAMPLE_NPROC"
echo "SAMPLE_BATCH $SAMPLE_BATCH"
echo "CHECKPOINT_WEIGHT_KEY $CHECKPOINT_WEIGHT_KEY"

while [ ! -s "$CHECKPOINT" ]; do
  echo "WAIT_FINAL_CHECKPOINT path=$CHECKPOINT $(date)"
  sleep "$POLL_SECONDS"
done
echo "FINAL_CHECKPOINT_READY $CHECKPOINT $(date)"

if [ ! -s "$SAMPLE_CFG" ]; then
  cd "$EQVAE_ROOT"
  if [ "$CHECKPOINT_WEIGHT_KEY" = "auto" ]; then
    python experiments/make_rae_sampling_config.py \
      --base-config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output "$SAMPLE_CFG" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --num-steps "$NUM_STEPS"
  else
    python experiments/make_rae_sampling_config.py \
      --base-config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --output "$SAMPLE_CFG" \
      --guidance-scale "$GUIDANCE_SCALE" \
      --num-steps "$NUM_STEPS" \
      --checkpoint-weight-key "$CHECKPOINT_WEIGHT_KEY" \
      --materialized-checkpoint "$MATERIALIZED_CHECKPOINT"
  fi
  echo "FINAL50K_SAMPLING_CONFIG_READY $SAMPLE_CFG"
else
  echo "FINAL50K_SAMPLING_CONFIG_EXISTS $SAMPLE_CFG"
fi

while [ ! -s "$SAMPLE_NPZ" ]; do
  echo "FINAL50K_SAMPLE_START folder=$SAMPLE_FOLDER $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  set +e
  CUDA_VISIBLE_DEVICES="$SAMPLE_GPUS" SAVE_FOLDER="$SAMPLE_FOLDER" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$SAMPLE_NPROC" src/sample_ddp.py \
      --config "$SAMPLE_CFG" \
      --sample-dir "$SAMPLE_ROOT" \
      --per-proc-batch-size "$SAMPLE_BATCH" \
      --num-fid-samples "$NUM_FID_SAMPLES" \
      --global-seed 0 \
      --precision fp32 \
      --label-sampling equal >> "$SAMPLE_LOG" 2>&1
  code=$?
  set -e
  echo "FINAL50K_SAMPLE_EXIT code=$code $(date)"
  if [ "$code" -eq 0 ]; then
    break
  fi
  echo "FINAL50K_SAMPLE_RETRY $(date)"
  sleep 300
done
echo "FINAL50K_SAMPLE_READY $SAMPLE_NPZ $(date)"

if [ ! -s "$GRID_PNG" ]; then
  echo "FINAL50K_GRID_START $(date)"
  cd "$EQVAE_ROOT"
  python experiments/make_sample_grid.py \
    --sample-dir "$SAMPLE_ROOT/$SAMPLE_FOLDER" \
    --output "$GRID_PNG" \
    --num-images 64 \
    --cols 8 \
    --thumb-size 128
  echo "FINAL50K_GRID_READY $GRID_PNG $(date)"
else
  echo "FINAL50K_GRID_EXISTS $GRID_PNG"
fi

if [ ! -s "$OUT_JSON" ]; then
  echo "FINAL50K_ADM_START $(date)"
  cd "$EQVAE_ROOT"
  CUDA_VISIBLE_DEVICES="$ADM_GPU" PYTHONUNBUFFERED=1 "$ADM_PY" experiments/compute_adm_fid.py \
    --reference "$REF" \
    --samples "$SAMPLE_NPZ" \
    --batch-size 64 \
    --inception-path "$INCEPTION" \
    --output "$OUT_JSON" > "$ADM_LOG" 2>&1
  echo "FINAL50K_ADM_READY $OUT_JSON $(date)"
else
  echo "FINAL50K_ADM_JSON_EXISTS $OUT_JSON"
fi

echo "FINAL50K_CHAIN_DONE $(date)"
