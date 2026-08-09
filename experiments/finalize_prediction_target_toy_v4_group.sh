#!/usr/bin/env bash
set -euo pipefail

WAIT_SESSION=${1:?usage: $0 TMUX_SESSION ROOT_NAME}
ROOT_NAME=${2:?usage: $0 TMUX_SESSION ROOT_NAME}

while tmux has-session -t "$WAIT_SESSION" 2>/dev/null; do
  sleep 60
done

cd /home/zhoushunyu/eqvae

ROOT="/home/zhoushunyu/data/eqvae/experiments/$ROOT_NAME"
python experiments/summarize_prediction_target_toy_v4.py \
  --input-root "$ROOT" \
  --output-dir "$ROOT/aggregate"
