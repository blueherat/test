#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

root=/data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_multiregime_screen
seeds=(20260817 20260818 20260819 20260820)
pids=()

mkdir -p "$root/logs"

for gpu in 0 1 2 3; do
  seed=${seeds[$gpu]}
  (
    # Phase A: strictly retest the only positive v3 regime with matched
    # sampling and metric randomness across conditions.
    env CUDA_VISIBLE_DEVICES="$gpu" \
      python experiments/run_prediction_target_extrapolation_toy_v4.py \
        --output-root "$root/run_seed${seed}/mid_linear" \
        --dims 16 \
        --curvatures 0 \
        --hidden-dims 256 \
        --loss-spaces v \
        --scale-mode constant_norm \
        --train-steps 30000 \
        --oracle-hidden-dim 1024 \
        --oracle-depth 6 \
        --oracle-train-steps 60000 \
        --batch-size 1024 \
        --eval-samples 8192 \
        --sample-count 5000 \
        --sample-batch-size 1000 \
        --sample-steps 200 \
        --gammas=-0.5,-0.25,-0.1,-0.03,-0.01,0.01,0.03,0.1,0.25,0.5 \
        --normalized-etas=-0.03,-0.01,0.01,0.03 \
        --generation-profile core \
        --bootstrap-reps 200 \
        --seeds "$seed" \
        --device cuda \
        --save-checkpoints \
        --resume

    # Phase B: make x-prediction itself capacity-limited using a curved
    # intrinsic-2D manifold whose fixed linear span can approach D.
    env CUDA_VISIBLE_DEVICES="$gpu" \
      python experiments/run_prediction_target_extrapolation_toy_v4.py \
        --output-root "$root/run_seed${seed}/curved_capacity" \
        --dims 512 \
        --curvatures 0.5 \
        --hidden-dims 64,256,1024 \
        --loss-spaces v \
        --scale-mode unit_rms \
        --frequency-scale 6.0 \
        --train-steps 15000 \
        --oracle-hidden-dim 1024 \
        --oracle-depth 6 \
        --oracle-train-steps 30000 \
        --batch-size 1024 \
        --eval-samples 4096 \
        --sample-count 3000 \
        --sample-batch-size 500 \
        --sample-steps 100 \
        --gammas=-0.1,-0.03,-0.01,0.01,0.03,0.1 \
        --normalized-etas=-0.03,-0.01,0.01,0.03 \
        --generation-profile core \
        --bootstrap-reps 100 \
        --seeds "$seed" \
        --device cuda \
        --save-checkpoints \
        --resume
  ) >"$root/logs/seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "At least one seed failed; rerun this script to resume." >&2
  exit "$status"
fi

python experiments/summarize_prediction_target_toy_v4.py \
  --input-root "$root"
