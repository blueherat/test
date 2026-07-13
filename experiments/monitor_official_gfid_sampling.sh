#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
SAVE_FOLDER=${SAVE_FOLDER:-official_ditdh_xl_ag_n50000_adm}
SAMPLE_DIR="$SAMPLE_ROOT/$SAVE_FOLDER"
SAMPLE_NPZ="$SAMPLE_ROOT/${SAVE_FOLDER}.npz"
LOG_DIR="$SAMPLE_ROOT/logs"
SAMPLE_LOG="$LOG_DIR/${SAVE_FOLDER}.log"
MONITOR_LOG="$LOG_DIR/${SAVE_FOLDER}_monitor.log"
CONFIG=${CONFIG:-configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml}

mkdir -p "$LOG_DIR"
exec >> "$MONITOR_LOG" 2>&1
trap 'echo "MONITOR_EXIT $? $(date)"' EXIT

echo "MONITOR_START $(date)"
echo "SAMPLE_DIR $SAMPLE_DIR"

sample_is_running() {
  pgrep -af "src/sample_ddp.py --config ${CONFIG}" >/dev/null
}

while true; do
  if [ -s "$SAMPLE_NPZ" ] && grep -q "EXIT 0" "$SAMPLE_LOG"; then
    echo "OFFICIAL_SAMPLE_COMPLETE $SAMPLE_NPZ $(date)"
    exit 0
  fi

  if sample_is_running; then
    echo "OFFICIAL_SAMPLE_RUNNING $(date)"
    sleep 300
    continue
  fi

  echo "OFFICIAL_SAMPLE_NOT_RUNNING_RESUME $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  {
    echo "RESUME_START $(date)"
    CUDA_VISIBLE_DEVICES=0,1,2,3 SAVE_FOLDER="$SAVE_FOLDER" PYTHONPATH=src PYTHONUNBUFFERED=1 \
      torchrun --standalone --nproc_per_node=4 src/sample_ddp.py \
        --config "$CONFIG" \
        --sample-dir "$SAMPLE_ROOT" \
        --per-proc-batch-size 4 \
        --num-fid-samples 50000 \
        --global-seed 0 \
        --precision fp32 \
        --label-sampling equal
    code=$?
    echo "EXIT ${code} $(date)"
    exit "$code"
  } >> "$SAMPLE_LOG" 2>&1 || {
    code=$?
    echo "OFFICIAL_SAMPLE_RESUME_FAILED $code $(date)"
    sleep 300
  }
done
