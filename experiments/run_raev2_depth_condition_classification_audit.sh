#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

DATA_ROOT=/home/zhoushunyu/data/eqvae/experiments/raev2_depth_condition_guidance_v1

run_seed() {
  local gpu=$1
  local seed=$2
  local root=${DATA_ROOT}/formal_n1000_seed${seed}_v1
  local output=${root}/classification_audit

  CUDA_VISIBLE_DEVICES=${gpu} python experiments/audit_raev2_generated_classification.py \
    --baseline full_conditional \
    --output-dir "${output}" \
    --batch-size 64 \
    --branch "full_conditional=${root}/full_conditional_run/full_conditional/samples.npz" \
    --branch "conditional_ig=${root}/conditional_ig_s1p78_t0p5_1p0_run/conditional_ig_s1p78_t0p5_1p0/samples.npz" \
    --branch "marginal_ig=${root}/marginal_ig_s1p78_t0p5_1p0_run/marginal_ig_s1p78_t0p5_1p0/samples.npz" \
    --branch "interaction_ig=${root}/interaction_ig_s1p78_t0p5_1p0_run/interaction_ig_s1p78_t0p5_1p0/samples.npz" \
    >"${root}/classification_audit.log" 2>&1
}

run_seed 0 20260903 &
pid0=$!
run_seed 1 20260904 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

cat "${DATA_ROOT}/formal_n1000_seed20260903_v1/classification_audit/classification_summary.csv"
cat "${DATA_ROOT}/formal_n1000_seed20260904_v1/classification_audit/classification_summary.csv"
