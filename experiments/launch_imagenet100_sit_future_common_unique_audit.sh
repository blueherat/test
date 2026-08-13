#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

base=/home/zhoushunyu/data/eqvae/imagenet_sit_flow
root="$base/fid5k_step400k_floor_audit_seed0/common_unique_x400_v270"
logs="$root/logs"
mkdir -p "$logs"

v400="$base/runs/sit-s-2_seed0/checkpoints/step_00400000.pt"
v800="$base/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
v270="$base/runs/sit-s-2_seed0/checkpoints/step_00270000.pt"
x400="$base/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00400000.pt"

run_component() {
  local component="$1"
  local gpu="$2"
  python experiments/run_imagenet100_sit_static_pair_fid5k.py \
    --anchor-checkpoint "$v400" \
    --other-checkpoint "$x400" \
    --reference-checkpoint "$v270" \
    --common-unique-component "$component" \
    --allow-reference-step-mismatch \
    --scales 1 \
    --output-root "$root/$component" \
    --sampling-cuda-visible-devices "$gpu" \
    --fid-cuda-visible-devices "$gpu" \
    --per-rank-batch-size 8 \
    --vae-decode-batch-size 2 \
    --cuda-allocator-limit-gib 4 \
    --gpu-memory-ceiling-mib 8192 \
    >"$logs/$component.log" 2>&1
}

run_component x_common_on_v 0 &
pid_x_common=$!
run_component x_unique_to_v 1 &
pid_x_unique=$!
run_component v_common_on_x 2 &
pid_v_common=$!
run_component v_unique_to_x 3 &
pid_v_unique=$!

wait "$pid_x_common"
wait "$pid_x_unique"
wait "$pid_v_common"
wait "$pid_v_unique"

env MPLCONFIGDIR=/tmp/matplotlib-sit-common-unique \
  python experiments/summarize_imagenet100_sit_future_common_unique.py \
    --root "$root" \
    >"$logs/common_unique_summary.log" 2>&1

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/matplotlib-sit-future \
  python experiments/analyze_imagenet100_sit_future_training_direction.py \
    --v400 "$v400" \
    --v800 "$v800" \
    --x400 "$x400" \
    --v270 "$v270" \
    --samples 512 \
    --batch-size 16 \
    --output-dir "$root/future_training_direction" \
    --device cuda:0 \
    >"$logs/future_training_direction.log" 2>&1
