#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu}
CONFIG=${CONFIG:-$EQVAE_ROOT/experiments/configs/finetune_ditdh_s_dinov2_adapter_from_official_ep5.yaml}
RESULTS_DIR=${RESULTS_DIR:-$EQVAE_STAGE2_TRAINING}
SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$SAMPLE_ROOT/logs}
EVAL_STEPS=${EVAL_STEPS:-625 1250 1875 2500 3125 3750 4375 5000 5625 6250}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-5000}
SAMPLE_GPUS=${SAMPLE_GPUS:-0,1,2,3}
SAMPLE_NPROC=${SAMPLE_NPROC:-4}
SAMPLE_BATCH=${SAMPLE_BATCH:-1}
FALLBACK_SAMPLE_GPUS=${FALLBACK_SAMPLE_GPUS:-}
FALLBACK_SAMPLE_NPROC=${FALLBACK_SAMPLE_NPROC:-}
CHECKPOINT_WEIGHT_KEY=${CHECKPOINT_WEIGHT_KEY:-model}
REF=${REF:-$EQVAE_ADM_REF}
INCEPTION=${INCEPTION:-$EQVAE_ADM_INCEPTION}
ADM_PY=${ADM_PY:-$EQVAE_ADM_PY}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
NUM_STEPS=${NUM_STEPS:-50}
POLL_SECONDS=${POLL_SECONDS:-60}

RUN_DIR="$RESULTS_DIR/$RUN_NAME"
WATCH_LOG="$LOG_DIR/${RUN_NAME}_eval_watcher.log"

mkdir -p "$LOG_DIR" "$SAMPLE_ROOT"
exec >> "$WATCH_LOG" 2>&1
trap 'echo "FINETUNE_EVAL_WATCHER_EXIT $? $(date)"' EXIT

echo "FINETUNE_EVAL_WATCHER_START $(date)"
echo "RUN_NAME $RUN_NAME"
echo "CONFIG $CONFIG"
echo "EVAL_STEPS $EVAL_STEPS"
echo "NUM_FID_SAMPLES $NUM_FID_SAMPLES"
echo "SAMPLE_GPUS $SAMPLE_GPUS"
echo "SAMPLE_NPROC $SAMPLE_NPROC"
echo "SAMPLE_BATCH $SAMPLE_BATCH"
echo "FALLBACK_SAMPLE_GPUS $FALLBACK_SAMPLE_GPUS"
echo "FALLBACK_SAMPLE_NPROC $FALLBACK_SAMPLE_NPROC"
echo "CHECKPOINT_WEIGHT_KEY $CHECKPOINT_WEIGHT_KEY"

for step in $EVAL_STEPS; do
  ckpt="$RUN_DIR/checkpoints/step-$(printf '%07d' "$step").pt"
  weight_suffix=""
  if [ "$CHECKPOINT_WEIGHT_KEY" != "auto" ]; then
    weight_suffix="_${CHECKPOINT_WEIGHT_KEY}"
  fi
  sample_folder="${RUN_NAME}_step$(printf '%07d' "$step")${weight_suffix}_n${NUM_FID_SAMPLES}_adm"
  sample_cfg="$RUN_DIR/sampling_step${step}${weight_suffix}_n${NUM_FID_SAMPLES}.yaml"
  materialized_ckpt="$RUN_DIR/checkpoints/step-$(printf '%07d' "$step")${weight_suffix}_stage2.pt"
  sample_npz="$SAMPLE_ROOT/${sample_folder}.npz"
  out_json="$SAMPLE_ROOT/${sample_folder}_adm_fid.json"
  sample_log="$LOG_DIR/${sample_folder}.log"
  adm_log="$LOG_DIR/${sample_folder}_adm_fid.log"
  grid_png="$SAMPLE_ROOT/${sample_folder}_grid.png"

  while [ ! -s "$ckpt" ]; do
    echo "WAIT_CHECKPOINT step=$step path=$ckpt $(date)"
    sleep "$POLL_SECONDS"
  done
  echo "CHECKPOINT_READY step=$step path=$ckpt $(date)"

  if [ ! -s "$sample_cfg" ]; then
    cd "$EQVAE_ROOT"
    if [ "$CHECKPOINT_WEIGHT_KEY" = "auto" ]; then
      python experiments/make_rae_sampling_config.py \
        --base-config "$CONFIG" \
        --checkpoint "$ckpt" \
        --output "$sample_cfg" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --num-steps "$NUM_STEPS"
    else
      python experiments/make_rae_sampling_config.py \
        --base-config "$CONFIG" \
        --checkpoint "$ckpt" \
        --output "$sample_cfg" \
        --guidance-scale "$GUIDANCE_SCALE" \
        --num-steps "$NUM_STEPS" \
        --checkpoint-weight-key "$CHECKPOINT_WEIGHT_KEY" \
        --materialized-checkpoint "$materialized_ckpt"
    fi
    echo "SAMPLING_CONFIG_READY $sample_cfg"
  fi

  sample_gpus_current="$SAMPLE_GPUS"
  sample_nproc_current="$SAMPLE_NPROC"
  sample_attempt=0
  while [ ! -s "$sample_npz" ]; do
    sample_attempt=$((sample_attempt + 1))
    echo "SAMPLE_START step=$step folder=$sample_folder $(date)"
    cd "$EQVAE_ROOT/external/RAE"
    set +e
    CUDA_VISIBLE_DEVICES="$sample_gpus_current" SAVE_FOLDER="$sample_folder" PYTHONPATH=src PYTHONUNBUFFERED=1 \
      torchrun --standalone --nproc_per_node="$sample_nproc_current" src/sample_ddp.py \
        --config "$sample_cfg" \
        --sample-dir "$SAMPLE_ROOT" \
        --per-proc-batch-size "$SAMPLE_BATCH" \
        --num-fid-samples "$NUM_FID_SAMPLES" \
        --global-seed 0 \
        --precision fp32 \
        --label-sampling equal >> "$sample_log" 2>&1
    code=$?
    set -e
    echo "SAMPLE_EXIT step=$step code=$code $(date)"
    if [ "$code" -eq 0 ]; then
      break
    fi
    if [ "$sample_attempt" -eq 1 ] && [ -n "$FALLBACK_SAMPLE_GPUS" ] && [ -n "$FALLBACK_SAMPLE_NPROC" ]; then
      echo "SAMPLE_FALLBACK step=$step from_gpus=$sample_gpus_current to_gpus=$FALLBACK_SAMPLE_GPUS from_nproc=$sample_nproc_current to_nproc=$FALLBACK_SAMPLE_NPROC $(date)"
      sample_gpus_current="$FALLBACK_SAMPLE_GPUS"
      sample_nproc_current="$FALLBACK_SAMPLE_NPROC"
    fi
    echo "SAMPLE_RETRY step=$step $(date)"
    sleep 300
  done
  echo "SAMPLE_READY step=$step npz=$sample_npz $(date)"

  if [ ! -s "$grid_png" ]; then
    echo "GRID_START step=$step $(date)"
    cd "$EQVAE_ROOT"
    python experiments/make_sample_grid.py \
      --sample-dir "$SAMPLE_ROOT/$sample_folder" \
      --output "$grid_png" \
      --num-images 64 \
      --cols 8 \
      --thumb-size 128
    echo "GRID_READY step=$step png=$grid_png $(date)"
  else
    echo "GRID_EXISTS step=$step png=$grid_png"
  fi

  if [ ! -s "$out_json" ]; then
    echo "ADM_START step=$step $(date)"
    cd "$EQVAE_ROOT"
    CUDA_VISIBLE_DEVICES="${ADM_GPU:-1}" PYTHONUNBUFFERED=1 "$ADM_PY" experiments/compute_adm_fid.py \
      --reference "$REF" \
      --samples "$sample_npz" \
      --batch-size 64 \
      --inception-path "$INCEPTION" \
      --output "$out_json" > "$adm_log" 2>&1
    echo "ADM_READY step=$step json=$out_json $(date)"
  else
    echo "ADM_JSON_EXISTS step=$step json=$out_json"
  fi
done

echo "FINETUNE_EVAL_WATCHER_DONE $(date)"
