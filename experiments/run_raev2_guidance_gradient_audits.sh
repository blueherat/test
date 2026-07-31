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
probe_weight=0.0001

cd "$repo"
for target in full full_base guided guided_multiscale; do
  experiment="grad_${target}"
  if [[ -f "$result_root/$experiment/gradient_audit.json" ]]; then
    echo "skip completed $experiment"
    continue
  fi
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
    --lpl-weight "$probe_weight" \
    --lpl-noise-threshold 3.0 \
    --lpl-max-samples-per-rank 1 \
    --max-updates 1 \
    --save-every 1 \
    --skip-checkpoint-save \
    --precision bf16 \
    --ema-device cpu \
    --num-workers 4 \
    --min-free-gib 0.5 \
    --gradient-audit-component lpl \
    --gradient-audit-microbatches 64 \
    --dino-repo-dir "$dino_repo"
done

"$python" experiments/summarize_raev2_gradient_audits.py \
  --flow-audit "$result_root/grad_flow/gradient_audit.json" \
  --lpl-audit "full=$result_root/grad_full/gradient_audit.json" \
  --lpl-audit "full_base=$result_root/grad_full_base/gradient_audit.json" \
  --lpl-audit "guided=$result_root/grad_guided/gradient_audit.json" \
  --lpl-audit "guided_multiscale=$result_root/grad_guided_multiscale/gradient_audit.json" \
  --target-ratio 0.20 \
  --output-json "$result_root/gradient_calibration.json" \
  --output-csv "$result_root/gradient_calibration.csv"
