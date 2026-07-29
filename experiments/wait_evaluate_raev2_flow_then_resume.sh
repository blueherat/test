#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
pipeline_root="$data_root/experiments/raev2_lpl_pilot/long_flow_lpl_s150"
sample_root="$pipeline_root/samples_n5000_seed0"
log="$pipeline_root/logs/flow_eval_then_resume.log"
expected_summaries=16
min_free_mib=22800

mkdir -p "$pipeline_root/logs"
cd "$repo"

while true; do
  count=$(find "$sample_root" -mindepth 2 -maxdepth 2 \
    -name sampling_summary.json | wc -l)
  printf '%s flow_sample_summaries=%s/%s\n' \
    "$(date --iso-8601=seconds)" "$count" "$expected_summaries" >> "$log"
  if (( count >= expected_summaries )); then
    break
  fi
  sleep 60
done

sampler_pattern="^${python} .*sample_raev2_threeway.py.*${pipeline_root}"
while pgrep -f "$sampler_pattern" >/dev/null; do
  sleep 5
done
tmux kill-session -t raev2_flow_sampling_then_resume 2>/dev/null || true

CUDA_VISIBLE_DEVICES=0 "$python" experiments/evaluate_raev2_flow_curve.py \
  --pipeline-root "$pipeline_root" \
  --sample-count 5000 \
  --python "$python" >> "$log" 2>&1
printf '%s flow_evaluation_exit=0\n' "$(date --iso-8601=seconds)" >> "$log"

while true; do
  free_now=$(nvidia-smi --query-gpu=memory.free \
    --format=csv,noheader,nounits | sort -n | head -1)
  printf '%s waiting_lpl_min_free_mib=%s\n' \
    "$(date --iso-8601=seconds)" "$free_now" >> "$log"
  if [[ -n "$free_now" ]] && (( free_now >= min_free_mib )); then
    ./experiments/launch_raev2_long_pipeline.sh >> "$log" 2>&1
    exit $?
  fi
  sleep 300
done
