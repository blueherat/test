#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
session=raev2_flow_lpl_long_150
pipeline_root="$data_root/experiments/raev2_lpl_pilot/long_flow_lpl_s150"
launcher_log="$pipeline_root/launcher.log"
flow_final_summary="$pipeline_root/samples_n5000_seed0/flow_s0150/sampling_summary.json"
flow_evaluation_status="$pipeline_root/flow_evaluation_status.json"

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
if [[ -f "$flow_final_summary" ]] && \
   ! grep -q '"state": "complete"' "$flow_evaluation_status" 2>/dev/null; then
  echo "Flow sampling is complete, but Flow evaluation must finish before LPL resumes" >&2
  exit 1
fi

mkdir -p "$pipeline_root"
tmux new-session -d -s "$session" \
  "bash -lc 'set -o pipefail; cd $repo; $python experiments/run_raev2_long_pipeline.py --target-step 150 --checkpoint-every 10 --sample-count 5000 --per-rank-batch 16 --min-free-gib 2.0 2>&1 | tee -a $launcher_log'"

echo "started: $session"
echo "status:  $pipeline_root/status.json"
echo "events:  $pipeline_root/events.jsonl"
echo "log:     $launcher_log"
