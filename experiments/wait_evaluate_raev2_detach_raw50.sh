#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
config="$repo/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
dino_repo="$data_root/models/RAEv2/dinov3_repo"
reference=/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz

flow50="$data_root/experiments/raev2_lpl_pilot/flow_official_150_strict_from30/checkpoints/branch-0000050-global-0100130.pt"
detach50="$data_root/experiments/raev2_prediction_detach_10step/lpl10_full_base_prediction_detach_final/checkpoints/branch-0000050-global-0100130.pt"
raw50="$data_root/experiments/raev2_raw_10step/raw10_gradcal_full_base/checkpoints/branch-0000050-global-0100130.pt"

result_root="$data_root/experiments/raev2_detach_raw_50step_evaluation"
samples="$result_root/nonema_samples_n5000_seed0"
metrics="$result_root/nonema_metrics_n5000_seed0.csv"
same_noise_audit="$result_root/nonema_same_noise_audit.json"
training_audit="$result_root/training_pair_audit.json"
log="$result_root/run.log"

mkdir -p "$result_root"
exec >>"$log" 2>&1

timestamp() {
  date --iso-8601=seconds
}

echo "[$(timestamp)] waiting for raw50 checkpoint"
while [[ ! -f "$raw50" ]]; do
  sleep 60
done

while pgrep -f '[t]rain_raev2_strict_lpl.py' >/dev/null; do
  echo "[$(timestamp)] raw50 exists; waiting for trainer shutdown"
  sleep 30
done

for required in \
  "$python" \
  "$config" \
  "$dino_repo" \
  "$reference" \
  "$flow50" \
  "$detach50" \
  "$raw50"; do
  if [[ ! -e "$required" ]]; then
    echo "required path is missing: $required" >&2
    exit 1
  fi
done

if [[ -f "$metrics" ]]; then
  echo "[$(timestamp)] evaluation already completed: $metrics"
  exit 0
fi
if pgrep -f '[s]ample_raev2_threeway.py' >/dev/null; then
  echo "another RAEv2 sampler is already active" >&2
  exit 1
fi

cd "$repo"
"$python" -c "
import json
from pathlib import Path

import numpy as np
import torch

paths = {
    'flow50': Path('$flow50'),
    'detach50': Path('$detach50'),
    'raw50': Path('$raw50'),
}

def load_metadata(path):
    checkpoint = torch.load(
        path, map_location='cpu', weights_only=False, mmap=True
    )
    metadata = checkpoint['raev2_lpl']
    result = {
        'step': int(checkpoint['step']),
        'source_step': int(metadata['source_step']),
        'source_sha256': metadata['source_sha256'],
        'config_sha256': metadata['config_sha256'],
        'branch_update': int(metadata['branch_update']),
        'objective': metadata['objective'],
        'data_indices_sha256': metadata['data_indices_sha256'],
        'paired_branch_stream_is_exact': bool(
            metadata['paired_branch_stream_is_exact']
        ),
        'rank_rng_states': metadata['rank_rng_states'],
    }
    del checkpoint
    return result

def rng_equal(left, right):
    if len(left) != len(right):
        return False
    for left_rank, right_rank in zip(left, right):
        if left_rank['rank'] != right_rank['rank']:
            return False
        if not torch.equal(left_rank['torch_cpu'], right_rank['torch_cpu']):
            return False
        if not torch.equal(left_rank['torch_cuda'], right_rank['torch_cuda']):
            return False
        left_np, right_np = left_rank['numpy'], right_rank['numpy']
        if (
            left_np[0] != right_np[0]
            or not np.array_equal(left_np[1], right_np[1])
            or left_np[2:] != right_np[2:]
        ):
            return False
        if left_rank['python'] != right_rank['python']:
            return False
    return True

loaded = {name: load_metadata(path) for name, path in paths.items()}
reference_metadata = loaded['flow50']
for name, metadata in loaded.items():
    assert metadata['step'] == 100130, (name, metadata['step'])
    assert metadata['source_step'] == 100080, (name, metadata['source_step'])
    assert metadata['branch_update'] == 50, (name, metadata['branch_update'])
    assert metadata['source_sha256'] == reference_metadata['source_sha256']
    assert metadata['config_sha256'] == reference_metadata['config_sha256']
    assert metadata['paired_branch_stream_is_exact']
    assert rng_equal(
        metadata['rank_rng_states'], reference_metadata['rank_rng_states']
    ), name

assert (
    loaded['detach50']['data_indices_sha256']
    == loaded['raw50']['data_indices_sha256']
)

report = {
    'protocol': 'raev2_flow_detach_raw50_training_pair_v1',
    'source_step': 100080,
    'branch_update': 50,
    'global_step': 100130,
    'all_rank_rng_states_equal': True,
    'detach_raw_segment_data_indices_equal': True,
    'checkpoints': {
        name: {
            key: value
            for key, value in metadata.items()
            if key != 'rank_rng_states'
        }
        for name, metadata in loaded.items()
    },
}
Path('$training_audit').write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + '\n',
    encoding='utf-8',
)
print(json.dumps(report, ensure_ascii=False))
"

echo "[$(timestamp)] paired checkpoint audit passed"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$python" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  experiments/sample_raev2_threeway.py \
  --config "$config" \
  --branch "flow50_model=$flow50" \
  --branch "detach50_model=$detach50" \
  --branch "raw50_model=$raw50" \
  --results-dir "$samples" \
  --sample-count 5000 \
  --per-rank-batch 16 \
  --sampling-seed 0 \
  --precision bf16 \
  --state-key model \
  --dino-repo-dir "$dino_repo"

"$python" -c "
import json
from pathlib import Path
from experiments.run_raev2_long_pipeline import verify_same_noise_protocol

report = verify_same_noise_protocol(
    {
        'flow50_model': Path('$samples/flow50_model'),
        'detach50_model': Path('$samples/detach50_model'),
        'raw50_model': Path('$samples/raw50_model'),
    }
)
Path('$same_noise_audit').write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + '\n',
    encoding='utf-8',
)
print(json.dumps(report, ensure_ascii=False))
"

CUDA_VISIBLE_DEVICES=0 "$python" experiments/evaluate_raev2_samples.py \
  --branch "flow50_model=$samples/flow50_model/samples.npz" \
  --branch "detach50_model=$samples/detach50_model/samples.npz" \
  --branch "raw50_model=$samples/raw50_model/samples.npz" \
  --reference "$reference" \
  --output "$metrics" \
  --batch-size 64 \
  --seed 0

echo "[$(timestamp)] evaluation complete: $metrics"
