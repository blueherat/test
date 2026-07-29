#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
session=raev2_lpl_cycle_to1000
experiment_dir="$data_root/experiments/raev2_lpl_pilot/lpl_official_800_strict_from10"
launcher_log="$experiment_dir/cycle_console.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA GPUs are not visible; refusing to create a dead long-running job" >&2
  exit 1
fi
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session"
  exit 0
fi
python_pattern="^(/home/zhoushunyu/data|/data/users/zhoushunyu)/eqvae/envs/raev2/bin/python .*train_raev2_strict_lpl.py"
if pgrep -f "$python_pattern" >/dev/null; then
  echo "an RAEv2 continuation process is already active" >&2
  exit 1
fi

mkdir -p "$experiment_dir"
tmux new-session -d -s "$session" \
  "bash -lc 'set -o pipefail; cd $repo; CUDA_VISIBLE_DEVICES=0,1,2,3 $python experiments/run_raev2_lpl_checkpoint_cycle.py --target-step 1000 --checkpoint-every 50 --sample-every 100 --sample-count 5000 --per-rank-batch 16 --min-free-gib 0.5 --compile-stage2 --experiment-name lpl_official_800_strict_from10 2>&1 | tee -a $launcher_log'"
tmux set-option -t "$session" remain-on-exit on

echo "started: $session"
echo "status:  $experiment_dir/cycle_status.json"
echo "events:  $experiment_dir/cycle_events.jsonl"
echo "log:     $launcher_log"
