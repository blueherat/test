#!/usr/bin/env bash
set -euo pipefail

for session in pt_v4_resume pt_v4_main_s08 pt_v4_main_s09_wait; do
  while tmux has-session -t "$session" 2>/dev/null; do
    sleep 60
  done
done

cd /home/zhoushunyu/eqvae

ROOT=/home/zhoushunyu/data/eqvae/experiments/prediction_target_toy_v4_main
python experiments/summarize_prediction_target_toy_v4.py \
  --input-root "$ROOT" \
  --output-dir "$ROOT/aggregate"
