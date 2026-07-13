#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

ROOT=${ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
SAVE_FOLDER=${SAVE_FOLDER:-official_ditdh_xl_noag_n50000_adm}
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml}
REF=${REF:-$EQVAE_ADM_REF}
INCEPTION=${INCEPTION:-$EQVAE_ADM_INCEPTION}
ADM_PY=${ADM_PY:-$EQVAE_ADM_PY}
BATCH_SIZES=${BATCH_SIZES:-"16 8 4"}

CHAIN_LOG="$LOG_DIR/${SAVE_FOLDER}_chain.log"
SAMPLE_LOG="$LOG_DIR/${SAVE_FOLDER}.log"
EVAL_LOG="$LOG_DIR/${SAVE_FOLDER}_adm_fid.log"
SAMPLE_NPZ="$ROOT/${SAVE_FOLDER}.npz"
OUT_JSON="$ROOT/${SAVE_FOLDER}_adm_fid.json"

mkdir -p "$LOG_DIR"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "NOAG_CHAIN_EXIT $? $(date)"' EXIT

echo "NOAG_CHAIN_START $(date)"

if [ ! -s "$SAMPLE_NPZ" ]; then
  for batch in $BATCH_SIZES; do
    echo "NOAG_SAMPLE_TRY batch=$batch $(date)"
    cd "$EQVAE_ROOT/external/RAE"
    set +e
    CUDA_VISIBLE_DEVICES=0,1,2,3 SAVE_FOLDER="$SAVE_FOLDER" PYTHONPATH=src PYTHONUNBUFFERED=1 \
      torchrun --standalone --nproc_per_node=4 src/sample_ddp.py \
        --config "$CONFIG" \
        --sample-dir "$ROOT" \
        --per-proc-batch-size "$batch" \
        --num-fid-samples 50000 \
        --global-seed 0 \
        --precision fp32 \
        --label-sampling equal >> "$SAMPLE_LOG" 2>&1
    code=$?
    set -e
    echo "NOAG_SAMPLE_EXIT batch=$batch code=$code $(date)"
    if [ "$code" -eq 0 ] && [ -s "$SAMPLE_NPZ" ]; then
      break
    fi
    echo "NOAG_SAMPLE_FALLBACK_FROM_BATCH $batch $(date)"
    sleep 30
  done
fi

if [ ! -s "$SAMPLE_NPZ" ]; then
  echo "NOAG_SAMPLE_MISSING $SAMPLE_NPZ"
  exit 2
fi

if [ -s "$OUT_JSON" ]; then
  echo "NOAG_ADM_JSON_EXISTS $OUT_JSON"
  exit 0
fi

cd "$EQVAE_ROOT"
echo "NOAG_ADM_START $(date)"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$ADM_PY" experiments/compute_adm_fid.py \
  --reference "$REF" \
  --samples "$SAMPLE_NPZ" \
  --batch-size 64 \
  --inception-path "$INCEPTION" \
  --output "$OUT_JSON" > "$EVAL_LOG" 2>&1
echo "NOAG_ADM_DONE $OUT_JSON $(date)"
