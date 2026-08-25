#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

gpu="${GPU:-3}"
output_root="${OUTPUT_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_critic_generator_crossplay_5k_v1}"

static_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_paired5k_v1/paired_eval/static_5k/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
full_5k_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/official_eval5k/eqvae_advfd_official_eval/step_0005000/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
full_10k_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/official_eval5k/eqvae_advfd_official_eval/step_0009999/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
mean_g_images="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_paired5k_v1/paired_eval/full_d_mean_g_5k/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"

full_5k_ckpt="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/eqvae_advfd_reproduction/pmf_b_sim_advinc_officialavg_w3_b24_q50k_10k/checkpoints/step_0005000.pth"
full_10k_ckpt="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/eqvae_advfd_reproduction/pmf_b_sim_advinc_officialavg_w3_b24_q50k_10k/checkpoints/step_0009999.pth"
mean_g_ckpt="/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_w3_b24_5k_v1/eqvae_advfd_reproduction/pmf_b_sim_advinc_full_d_mean_g_officialavg_w3_b24_q50k_5k/checkpoints/step_0004999.pth"

CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="$PWD" \
python experiments/advfd_cleanroom/audit_pmf_critic_generator_crossplay.py \
  --critic pretrained=pretrained \
  --critic "full_5k=${full_5k_ckpt}" \
  --critic "full_10k=${full_10k_ckpt}" \
  --critic "fullD_meanG_5k=${mean_g_ckpt}" \
  --image-folder "static=${static_images}" \
  --image-folder "full_5k=${full_5k_images}" \
  --image-folder "full_10k=${full_10k_images}" \
  --image-folder "fullD_meanG_5k=${mean_g_images}" \
  --real-split train \
  --sample-count 5000 \
  --batch-size 32 \
  --cka-samples 1024 \
  --generator-anchor static \
  --device cuda:0 \
  --output-root "${output_root}"
