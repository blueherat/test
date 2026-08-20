#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

root=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/cafm_tangent_predictivity_v1
mkdir -p "$root/logs"

pids=()
for gpu in 0 1 2 3; do
  seed=$gpu
  checkpoint="$root/critics_b256_drop10_fp32/seed${seed}/checkpoints/step_001000.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing fixed-protocol critic checkpoint: $checkpoint" >&2
    exit 1
  fi
  python -u -m experiments.audit_imagenet100_sit_cafm_predictivity \
    --device "cuda:${gpu}" \
    --critic-checkpoint "$checkpoint" \
    --output-dir "$root/audits_final_b256_drop10_fp32/seed${seed}" \
    --num-samples 4096 \
    --batch-size 32 \
    --critic-batch-size 16 \
    --workers 2 \
    --seed 91003 \
    >"$root/logs/audit_final_b256_drop10_fp32_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
