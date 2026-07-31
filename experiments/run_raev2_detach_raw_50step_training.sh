#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
python="$data_root/envs/raev2/bin/python"
config="$repo/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
source_checkpoint="$data_root/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"
dino_repo="$data_root/models/RAEv2/dinov3_repo"
index_map="$data_root/datasets/raev2_imagenet_train_lexicographic_indices.npy"

train_branch() {
  local name=$1
  local result_root=$2
  local experiment=$3
  local initial_resume=$4
  local variant=$5
  local weight=$6
  local checkpoint_dir="$result_root/$experiment/checkpoints"
  local final_checkpoint="$result_root/$experiment/checkpoints/branch-0000050-global-0100130.pt"
  local resume="$initial_resume"
  local checkpoints=()

  if [[ -f "$final_checkpoint" ]]; then
    echo "[$(date --iso-8601=seconds)] skip completed $name: $final_checkpoint"
    return
  fi
  shopt -s nullglob
  checkpoints=("$checkpoint_dir"/branch-*-global-*.pt)
  shopt -u nullglob
  if ((${#checkpoints[@]} > 0)); then
    resume="${checkpoints[-1]}"
  fi
  if [[ ! -f "$resume" ]]; then
    echo "missing resume checkpoint for $name: $resume" >&2
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] start $name from $(basename "$resume") to step 50"
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
    --resume "$resume" \
    --objective lpl \
    --lpl-target full_base \
    --lpl-variant "$variant" \
    --lpl-gradient-mode direct \
    --lpl-guidance-scale 1.78 \
    --lpl-multiscale-scales 1.0,1.39,1.78 \
    --lpl-weight "$weight" \
    --lpl-noise-threshold 3.0 \
    --lpl-max-samples-per-rank 1 \
    --max-updates 50 \
    --save-every 10 \
    --precision bf16 \
    --ema-device cpu \
    --global-seed 42 \
    --num-workers 4 \
    --min-free-gib 0.5 \
    --dino-repo-dir "$dino_repo"
  echo "[$(date --iso-8601=seconds)] completed $name: $final_checkpoint"
}

cd "$repo"

detach_root="$data_root/experiments/raev2_prediction_detach_10step"
detach_experiment=lpl10_full_base_prediction_detach_final
detach_resume="$detach_root/$detach_experiment/checkpoints/branch-0000010-global-0100090.pt"
train_branch \
  detach \
  "$detach_root" \
  "$detach_experiment" \
  "$detach_resume" \
  prediction_detach \
  2.9384045033942286e-5

raw_root="$data_root/experiments/raev2_raw_10step"
raw_experiment=raw10_gradcal_full_base
raw_resume="$raw_root/$raw_experiment/checkpoints/branch-0000010-global-0100090.pt"
train_branch \
  raw \
  "$raw_root" \
  "$raw_experiment" \
  "$raw_resume" \
  raw \
  8.13964254606437e-8
