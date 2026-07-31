#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
result_root="$data_root/experiments/raev2_guidance_aware_10step"
config="$repo/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
source_checkpoint="$data_root/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"
dino_repo="$data_root/models/RAEv2/dinov3_repo"
index_map="$data_root/datasets/raev2_imagenet_train_lexicographic_indices.npy"
calibration="$result_root/gradient_calibration.json"

if [[ ! -f "$calibration" ]]; then
  echo "missing gradient calibration: $calibration" >&2
  exit 1
fi

cd "$repo"
for target in full full_base guided guided_multiscale; do
  experiment="lpl10_gradcal_${target}"
  checkpoint="$result_root/$experiment/checkpoints/branch-0000010-global-0100090.pt"
  if [[ -f "$checkpoint" ]]; then
    echo "skip completed $experiment"
    continue
  fi
  weight=$(
    "$python" -c \
      "import json; print(json.load(open('$calibration'))['recommended_lpl_weights']['$target'])"
  )
  echo "start $experiment weight=$weight"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$python" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    experiments/train_raev2_strict_lpl.py \
    --config "$config" \
    --data-path /data/shared/imagenet-1k/data \
    --packed-data-path /data/shared/imagenet-1k/random_access_v1 \
    --index-map "$index_map" \
    --results-dir "$result_root" \
    --experiment-name "$experiment" \
    --source-checkpoint "$source_checkpoint" \
    --objective lpl \
    --lpl-target "$target" \
    --lpl-guidance-scale 1.78 \
    --lpl-multiscale-scales 1.0,1.39,1.78 \
    --lpl-weight "$weight" \
    --lpl-noise-threshold 3.0 \
    --lpl-max-samples-per-rank 1 \
    --max-updates 10 \
    --save-every 10 \
    --precision bf16 \
    --ema-device cpu \
    --num-workers 4 \
    --min-free-gib 0.5 \
    --dino-repo-dir "$dino_repo"
done
