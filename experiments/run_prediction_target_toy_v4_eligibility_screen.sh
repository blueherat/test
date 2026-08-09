#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# Fill the missing capacity interval between H=64 and H=256. The existing
# oracle checkpoints and completed settings are reused in place.
root=/data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_multiregime_screen
seeds=(20260817 20260818 20260819 20260820)
pids=()

mkdir -p "$root/logs"

for gpu in 0 1 2 3; do
  seed=${seeds[$gpu]}
  (
    env CUDA_VISIBLE_DEVICES="$gpu" \
      python experiments/run_prediction_target_extrapolation_toy_v4.py \
        --output-root "$root/run_seed${seed}/curved_capacity" \
        --dims 512 \
        --curvatures 0.5 \
        --hidden-dims 96,128,160,192,224 \
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
        --gammas=-0.03,0.003,0.01,0.02,0.03,0.05,0.1 \
        --generation-profile core \
        --bootstrap-reps 100 \
        --seeds "$seed" \
        --device cuda \
        --save-checkpoints \
        --resume
  ) >"$root/logs/eligibility_seed${seed}.log" 2>&1 &
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
