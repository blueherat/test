#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

GPU="${GPU:-2}"
ROOT="${ROOT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_step400k_floor_audit_seed0}"
LOG="$ROOT/driver_mechanism.log"

mkdir -p "$ROOT"
echo "[$(date --iso-8601=seconds)] mechanism-start gpu=$GPU" >> "$LOG"

set +e
GPU="$GPU" GROUP=mechanism ROOT="$ROOT" \
  SKIP_COMPLETE_SERIES="${SKIP_COMPLETE_SERIES:-0}" \
  bash experiments/run_imagenet100_sit_400k_floor_audit.sh >> "$LOG" 2>&1
status=$?
set -e

if [[ "$status" == "0" ]]; then
  python experiments/summarize_imagenet100_sit_400k_mechanism_audit.py \
    --allow-incomplete >> "$LOG" 2>&1 || status=$?
fi

echo "[$(date --iso-8601=seconds)] mechanism-exit=$status" >> "$LOG"
exit "$status"
