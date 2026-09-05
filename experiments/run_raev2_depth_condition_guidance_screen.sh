#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

SAMPLING_SEED=${SAMPLING_SEED:-20260903}
ROOT=${ROOT:-/home/zhoushunyu/data/eqvae/experiments/raev2_depth_condition_guidance_v1/formal_n1000_seed${SAMPLING_SEED}_v1}
if [[ -e "${ROOT}/complete.json" ]]; then
  echo "already complete: ${ROOT}"
  exit 0
fi
mkdir -p "${ROOT}"

names=(
  full_conditional
  conditional_ig_s1p78_t0p5_1p0
  marginal_ig_s1p78_t0p5_1p0
  interaction_ig_s1p78_t0p5_1p0
)
modes=(
  full_conditional
  conditional_depth
  marginal_depth
  interaction
)
scales=(1.0 1.78 1.78 1.78)

pids=()
for gpu in 0 1 2 3; do
  name="${names[$gpu]}"
  mode="${modes[$gpu]}"
  scale="${scales[$gpu]}"
  CUDA_VISIBLE_DEVICES="${gpu}" torchrun --standalone --nproc_per_node=1 \
    experiments/sample_raev2_depth_condition_guidance.py \
    --results-dir "${ROOT}/${name}_run" \
    --sample-count 1000 \
    --per-rank-batch 4 \
    --sampling-seed "${SAMPLING_SEED}" \
    --condition "${name},${mode},${scale},0.5,1.0" \
    >"${ROOT}/${name}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "at least one sampling worker failed" >&2
  exit "${status}"
fi

args=()
for name in "${names[@]}"; do
  args+=(--branch "${name}=${ROOT}/${name}_run/${name}/samples.npz")
done
CUDA_VISIBLE_DEVICES=0 python experiments/evaluate_raev2_official_samples.py \
  "${args[@]}" \
  --output "${ROOT}/official_metrics.csv" \
  --batch-size 64 \
  --device cuda \
  --seed 2020 \
  >"${ROOT}/official_evaluation.log" 2>&1

python - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
(root / "complete.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "metrics": str(root / "official_metrics.csv"),
            "protocol": "raev2_depth_condition_guidance_v1",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

cat "${ROOT}/official_metrics.csv"
