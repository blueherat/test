#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

root=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/cafm_tangent_predictivity_v1
mkdir -p "$root/logs"

pids=()
for gpu in 0 1 2 3; do
  seed=$gpu
  python -u -m experiments.train_imagenet100_sit_cafm_tangent_critic \
    --device "cuda:${gpu}" \
    --seed "$seed" \
    --output-dir "$root/critics_b256_drop10_fp32/seed${seed}" \
    --steps 1000 \
    --batch-size 32 \
    --accumulation-steps 8 \
    --workers 2 \
    --log-every 10 \
    --validate-every 100 \
    --validation-batches 64 \
    --save-every 250 \
    --class-dropout-probability 0.1 \
    --precision fp32 \
    >"$root/logs/train_b256_drop10_fp32_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
