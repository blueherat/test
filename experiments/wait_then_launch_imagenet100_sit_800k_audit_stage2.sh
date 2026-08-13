#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
AUDIT_ROOT="${AUDIT_ROOT:-$BASE/fid5k_step800k_floor_audit_seed0}"
LOG_DIR="$AUDIT_ROOT/logs"
STAGE1_SESSION="${STAGE1_SESSION:-sit800-full-audit}"
mkdir -p "$LOG_DIR"

while [[ ! -f "$AUDIT_ROOT/COMPLETE_STAGE1" ]]; do
  if tmux has-session -t "$STAGE1_SESSION" 2>/dev/null; then
    pane_dead="$(tmux display-message -p -t "$STAGE1_SESSION:0" '#{pane_dead}')"
    pane_status="$(tmux display-message -p -t "$STAGE1_SESSION:0" '#{pane_dead_status}')"
    if [[ "$pane_dead" == "1" ]]; then
      echo "[$(date --iso-8601=seconds)] stage1 stopped without completion (status=$pane_status)" \
        | tee -a "$LOG_DIR/stage2_wait.log"
      exit 1
    fi
  else
    echo "[$(date --iso-8601=seconds)] stage1 session disappeared without completion" \
      | tee -a "$LOG_DIR/stage2_wait.log"
    exit 1
  fi
  echo "[$(date --iso-8601=seconds)] waiting for stage1" >> "$LOG_DIR/stage2_wait.log"
  sleep 60
done

echo "[$(date --iso-8601=seconds)] stage1 complete; launching stage2" \
  | tee -a "$LOG_DIR/stage2_wait.log"
exec bash experiments/launch_imagenet100_sit_800k_audit_stage2.sh
