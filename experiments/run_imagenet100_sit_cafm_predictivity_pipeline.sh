#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

root=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/cafm_tangent_predictivity_v1
mkdir -p "$root/logs"

while true; do
  complete=1
  for seed in 0 1 2 3; do
    marker="$root/critics_b256_drop10_fp32/seed${seed}/complete.json"
    checkpoint="$root/critics_b256_drop10_fp32/seed${seed}/checkpoints/step_001000.pt"
    if [[ ! -f "$marker" || ! -f "$checkpoint" ]]; then
      complete=0
      break
    fi
  done
  if [[ "$complete" == 1 ]]; then
    break
  fi
  sleep 30
done

python -u -m experiments.audit_imagenet100_sit_cafm_predictivity \
  --device cuda:0 \
  --critic-checkpoint "$root/critics_b256_drop10_fp32/seed0/checkpoints/step_001000.pt" \
  --output-dir "$root/audit_smoke_final_b256_drop10_fp32" \
  --num-samples 32 \
  --batch-size 16 \
  --critic-batch-size 8 \
  --workers 0 \
  --seed 91003 \
  >"$root/logs/audit_smoke_final_b256_drop10_fp32.log" 2>&1

bash experiments/launch_imagenet100_sit_cafm_predictivity_audits.sh

python -u -m experiments.summarize_imagenet100_sit_cafm_predictivity \
  --audit-root "$root/audits_final_b256_drop10_fp32" \
  --output-dir "$root/summary_final_b256_drop10_fp32" \
  >"$root/logs/summarize_final_b256_drop10_fp32.log" 2>&1

python - "$root/pipeline_complete.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(
    json.dumps(
        {
            "status": "complete",
            "critic_protocol": "b256_drop10_fp32_fixed_step1000",
            "generator_updated": False,
            "new_fid_computed": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
