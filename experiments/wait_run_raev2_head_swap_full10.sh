#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
config="$repo/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
dino_repo="$data_root/models/RAEv2/dinov3_repo"
reference=/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz

upstream_metrics="$data_root/experiments/raev2_detach_raw_50step_evaluation/nonema_metrics_n5000_seed0.csv"
flow="$data_root/experiments/raev2_lpl_pilot/flow_official_10_strict/checkpoints/branch-0000010-global-0100090.pt"
treatment="$data_root/experiments/raev2_guidance_aware_10step/lpl10_gradcal_full/checkpoints/branch-0000010-global-0100090.pt"
old_flow="$data_root/experiments/raev2_guidance_aware_10step/samples_ig1p78_n1000_seed0/flow10_model/samples.npz"
old_treatment="$data_root/experiments/raev2_guidance_aware_10step/samples_ig1p78_n1000_seed0/full10_model/samples.npz"

result_root="$data_root/experiments/raev2_head_swap_full10"
samples="$result_root/samples_n1000_seed0"
metrics="$result_root/metrics_n1000_seed0.csv"
audit="$result_root/same_noise_and_parity_audit.json"
log="$result_root/run.log"

mkdir -p "$result_root"
exec >>"$log" 2>&1

timestamp() {
  date --iso-8601=seconds
}

echo "[$(timestamp)] waiting for detach/raw 50-step evaluation"
while [[ ! -f "$upstream_metrics" ]]; do
  sleep 60
done

while pgrep -f '[s]ample_raev2' >/dev/null; do
  echo "[$(timestamp)] waiting for active RAEv2 sampler"
  sleep 30
done

for required in \
  "$python" \
  "$config" \
  "$dino_repo" \
  "$reference" \
  "$flow" \
  "$treatment" \
  "$old_flow" \
  "$old_treatment"; do
  if [[ ! -e "$required" ]]; then
    echo "required path is missing: $required" >&2
    exit 1
  fi
done

if [[ -f "$metrics" ]]; then
  echo "[$(timestamp)] head-swap evaluation already completed: $metrics"
  exit 0
fi

cd "$repo"
samples_complete=true
for branch in flowF_flowD lplF_flowD flowF_lplD lplF_lplD; do
  branch_dir="$samples/$branch"
  for required in \
    "$branch_dir/samples.npz" \
    "$branch_dir/sampling_summary.json" \
    "$branch_dir/sampling_audit_rank0.json" \
    "$branch_dir/sampling_audit_rank1.json" \
    "$branch_dir/sampling_audit_rank2.json" \
    "$branch_dir/sampling_audit_rank3.json"; do
    if [[ ! -s "$required" ]]; then
      samples_complete=false
    fi
  done
done

if [[ "$samples_complete" == true ]]; then
  echo "[$(timestamp)] complete head-swap samples found; resuming at audit"
else
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$python" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    -m experiments.sample_raev2_head_swap \
    --config "$config" \
    --flow-checkpoint "$flow" \
    --treatment-checkpoint "$treatment" \
    --treatment-name lpl \
    --results-dir "$samples" \
    --sample-count 1000 \
    --per-rank-batch 16 \
    --sampling-seed 0 \
    --precision bf16 \
    --state-key model \
    --ig-scale 1.78 \
    --dino-repo-dir "$dino_repo"
fi

"$python" -c "
import json
from pathlib import Path

import numpy as np

from experiments.run_raev2_long_pipeline import verify_same_noise_protocol

paths = {
    'flowF_flowD': Path('$samples/flowF_flowD'),
    'lplF_flowD': Path('$samples/lplF_flowD'),
    'flowF_lplD': Path('$samples/flowF_lplD'),
    'lplF_lplD': Path('$samples/lplF_lplD'),
}
noise_report = verify_same_noise_protocol(
    paths,
    accepted_protocols=('raev2_head_swap_same_noise_v1',),
)
new_flow = np.load(paths['flowF_flowD'] / 'samples.npz')['arr_0']
new_lpl = np.load(paths['lplF_lplD'] / 'samples.npz')['arr_0']
old_flow = np.load('$old_flow')['arr_0']
old_lpl = np.load('$old_treatment')['arr_0']
flow_equal = bool(np.array_equal(new_flow, old_flow))
lpl_equal = bool(np.array_equal(new_lpl, old_lpl))
if not flow_equal or not lpl_equal:
    raise RuntimeError(
        f'pure-corner parity failed: flow={flow_equal}, lpl={lpl_equal}'
    )
report = {
    'protocol': 'raev2_head_swap_parity_v1',
    'same_noise': noise_report,
    'flow_corner_pixel_exact': flow_equal,
    'lpl_corner_pixel_exact': lpl_equal,
}
Path('$audit').write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + '\n',
    encoding='utf-8',
)
print(json.dumps(report, ensure_ascii=False))
"

CUDA_VISIBLE_DEVICES=0 "$python" experiments/evaluate_raev2_samples.py \
  --branch "flowF_flowD=$samples/flowF_flowD/samples.npz" \
  --branch "lplF_flowD=$samples/lplF_flowD/samples.npz" \
  --branch "flowF_lplD=$samples/flowF_lplD/samples.npz" \
  --branch "lplF_lplD=$samples/lplF_lplD/samples.npz" \
  --reference "$reference" \
  --output "$metrics" \
  --batch-size 64 \
  --seed 0

echo "[$(timestamp)] head-swap evaluation complete: $metrics"
