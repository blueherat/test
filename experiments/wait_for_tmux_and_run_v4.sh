#!/usr/bin/env bash
set -euo pipefail

WAIT_SESSION=${1:?usage: $0 TMUX_SESSION GROUP SEEDS}
GROUP=${2:?usage: $0 TMUX_SESSION GROUP SEEDS}
SEEDS=${3:?usage: $0 TMUX_SESSION GROUP SEEDS}

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 30
done

exec /home/zhoushunyu/eqvae/experiments/run_prediction_target_toy_v4_worker.sh \
  "$GROUP" "$SEEDS"
