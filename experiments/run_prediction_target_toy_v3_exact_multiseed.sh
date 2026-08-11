#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

root=/data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v3_exact_multiseed
seeds=(20260817 20260818 20260819 20260820)
pids=()

mkdir -p "$root/logs"

for gpu in 0 1 2 3; do
  seed=${seeds[$gpu]}
  (
    env CUDA_VISIBLE_DEVICES="$gpu" \
      python experiments/run_prediction_target_extrapolation_toy_v3.py \
        --output-root "$root/run_seed${seed}" \
        --dims 16 \
        --hidden-dim 256 \
        --depth 5 \
        --train-steps 30000 \
        --batch-size 1024 \
        --loss-space v \
        --sample-count 10000 \
        --sample-steps 200 \
        --gammas=0.01,0.03,0.1,0.25,0.5,1.0 \
        --seed "$seed" \
        --device cuda

    python experiments/reanalyze_prediction_target_toy_v3.py \
      --input-root "$root/run_seed${seed}" \
      --output-dir "$root/run_seed${seed}/reanalysis" \
      --dims 16 \
      --reference-count 10000 \
      --seed "$seed"
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
  echo "At least one exact-v3 seed failed." >&2
  exit "$status"
fi

python experiments/summarize_prediction_target_toy_v3_replays.py \
  --input-root "$root"
