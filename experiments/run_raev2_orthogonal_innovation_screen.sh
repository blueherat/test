#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

ROOT=${ROOT:-/home/zhoushunyu/data/eqvae/experiments/raev2_orthogonal_innovation_v1}
BASE_ROOT=/home/zhoushunyu/data/eqvae/experiments/raev2_depth_condition_guidance_v1
mkdir -p "${ROOT}"

run_sample() {
  local gpu=$1
  local seed=$2
  local name=$3
  local mode=$4
  local seed_root="${ROOT}/seed${seed}"
  local result_root="${seed_root}/${name}_run"
  local sample_path="${result_root}/${name}/samples.npz"
  mkdir -p "${seed_root}"
  if [[ -f "${sample_path}" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" torchrun --standalone --nproc_per_node=1 \
    experiments/sample_raev2_depth_condition_guidance.py \
    --results-dir "${result_root}" \
    --sample-count 1000 \
    --per-rank-batch 4 \
    --sampling-seed "${seed}" \
    --condition "${name},${mode},1.78,0.5,1.0" \
    >"${seed_root}/${name}.log" 2>&1
}

run_wave() {
  local pids=()
  while (( "$#" )); do
    run_sample "$1" "$2" "$3" "$4" &
    pids+=("$!")
    shift 4
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "at least one sampling worker failed" >&2
    exit "${status}"
  fi
}

run_wave \
  0 20260903 positive conditional_marginal_orthogonal_positive \
  1 20260903 negative conditional_marginal_orthogonal_negative \
  2 20260903 donor conditional_marginal_orthogonal_donor \
  3 20260904 positive conditional_marginal_orthogonal_positive

run_wave \
  0 20260904 negative conditional_marginal_orthogonal_negative \
  1 20260904 donor conditional_marginal_orthogonal_donor

for seed in 20260903 20260904; do
  seed_root="${ROOT}/seed${seed}"
  baseline_root="${BASE_ROOT}/formal_n1000_seed${seed}_v1"
  CUDA_VISIBLE_DEVICES=0 python experiments/evaluate_raev2_official_samples.py \
    --branch "full=${baseline_root}/full_conditional_run/full_conditional/samples.npz" \
    --branch "ordinary=${baseline_root}/conditional_ig_s1p78_t0p5_1p0_run/conditional_ig_s1p78_t0p5_1p0/samples.npz" \
    --branch "positive=${seed_root}/positive_run/positive/samples.npz" \
    --branch "negative=${seed_root}/negative_run/negative/samples.npz" \
    --branch "donor=${seed_root}/donor_run/donor/samples.npz" \
    --output "${seed_root}/official_metrics.csv" \
    --batch-size 64 \
    --device cuda \
    --seed 2020 \
    >"${seed_root}/official_evaluation.log" 2>&1
  cat "${seed_root}/official_metrics.csv"
done

python - "${ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
(root / "complete.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "protocol": "raev2_orthogonal_innovation_v1",
            "seeds": [20260903, 20260904],
            "sample_count_per_condition": 1000,
            "guidance_scale": 1.78,
            "guidance_interval": [0.5, 1.0],
            "controls": ["sign_flip", "cross_sample_norm_matched_donor"],
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
