#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

IDLE_MAX_MIB="${IDLE_MAX_MIB:-512}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GROUP="${GROUP:-all}"
STATUS_ROOT="${ROOT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k_step400k_floor_audit_seed0}"
mkdir -p "$STATUS_ROOT"

if [[ -f "$STATUS_ROOT/COMPLETE_$GROUP" ]]; then
  echo "400K audit group $GROUP is already complete."
  exit 0
fi

while true; do
  GPU="$({
    nvidia-smi \
      --query-gpu=index,memory.used \
      --format=csv,noheader,nounits
  } | awk -F',' -v limit="$IDLE_MAX_MIB" '
    {
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      if (($2 + 0) < limit) { print $1; exit }
    }
  ')"
  if [[ -n "$GPU" ]]; then
    echo "[$(date --iso-8601=seconds)] GPU $GPU is idle; launching 400K floor audit"
    export GPU GROUP
    exec bash experiments/run_imagenet100_sit_400k_floor_audit.sh
  fi
  echo "[$(date --iso-8601=seconds)] no GPU below ${IDLE_MAX_MIB} MiB; waiting"
  sleep "$POLL_SECONDS"
done
