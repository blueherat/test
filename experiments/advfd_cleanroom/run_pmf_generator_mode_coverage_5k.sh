#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

gpu="${GPU:-3}"
output_root="${OUTPUT_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_generator_mode_coverage_5k_v1}"

static_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_paired5k_v1/paired_eval/static_5k/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
full_5k_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/official_eval5k/eqvae_advfd_official_eval/step_0005000/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
full_10k_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/official_eval5k/eqvae_advfd_official_eval/step_0009999/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
mean_g_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_paired5k_v1/paired_eval/full_d_mean_g_5k/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="$PWD" \
python experiments/advfd_cleanroom/audit_pmf_generator_mode_coverage.py \
  --image-folder "static=${static_images}" \
  --image-folder "full_5k=${full_5k_images}" \
  --image-folder "full_10k=${full_10k_images}" \
  --image-folder "fullD_meanG_5k=${mean_g_images}" \
  --sample-count 5000 \
  --batch-size 32 \
  --distance-batch 128 \
  --neighborhood 3 \
  --device cuda:0 \
  --output-root "${output_root}"
