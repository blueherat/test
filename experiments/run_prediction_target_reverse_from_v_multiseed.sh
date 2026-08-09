#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

root=/data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_reverse_from_v
seeds=(20260817 20260818 20260819 20260820)
pids=()

mkdir -p "$root/logs"

for gpu in 0 1 2 3; do
  seed=${seeds[$gpu]}
  (
    env CUDA_VISIBLE_DEVICES="$gpu" \
      python experiments/evaluate_prediction_target_reverse_from_v.py \
        --seed "$seed" \
        --device cuda \
        --output-root "$root" \
        --sample-count 10000 \
        --sample-batch-size 500 \
        --sample-steps 200 \
        --bootstrap-reps 300
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
  echo "At least one seed failed." >&2
  exit "$status"
fi

python experiments/evaluate_prediction_target_reverse_from_v.py \
  --aggregate-only \
  --output-root "$root"
