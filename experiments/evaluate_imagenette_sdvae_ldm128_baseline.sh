#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

RUN_DIR="${RUN_DIR:-/home/zhoushunyu/data/eqvae/imagenette_sdvae_ldm128/formal_single_head_sdvae_v1}"
GPUS="${GPUS:-0,1,2}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"
STEP="${STEP:-5000}"
STEP_PADDED="$(printf '%07d' "${STEP}")"
METRIC_SCALES="${METRIC_SCALES:-2.0}"
CHECKPOINT="${RUN_DIR}/checkpoint_step${STEP_PADDED}.pt"
SAMPLE_ROOT="${SAMPLE_ROOT:-${RUN_DIR}/sampling_step${STEP}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/zhoushunyu/data/eqvae/torchinductor_cache}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing ${CHECKPOINT}" >&2
  exit 1
fi

# Small, paired visual sweep before committing to the quantitative setting.
for scale in 1.0 1.5 2.0 3.0; do
  scale_tag="${scale/./p}"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
    experiments/imagenette_sdvae_latent_diffusion.py sample \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${SAMPLE_ROOT}/preview_cfg${scale_tag}" \
    --sample-count 300 \
    --batch-size 16 \
    --decode-batch-size 16 \
    --sample-steps 50 \
    --guidance-scale "${scale}" \
    --seed 20260809 \
    --compile
done

# Formal metric points are explicit so a continuation can preserve CFG=2 for
# longitudinal comparison while adding a separately motivated scale.
for scale in ${METRIC_SCALES}; do
  scale_tag="${scale/./p}"
  metric_dir="${SAMPLE_ROOT}/fid5k_cfg${scale_tag}"
  CUDA_VISIBLE_DEVICES="${GPUS}" torchrun --standalone --nproc_per_node="${NPROC}" \
    experiments/imagenette_sdvae_latent_diffusion.py sample \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${metric_dir}" \
    --sample-count 5000 \
    --batch-size 16 \
    --decode-batch-size 16 \
    --sample-steps 50 \
    --guidance-scale "${scale}" \
    --seed 20260809 \
    --compile

  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python \
    experiments/evaluate_imagenette_sdvae_samples.py \
    --samples "${metric_dir}/samples.pt" \
    --output "${metric_dir}/metrics.json" \
    --batch-size 64 \
    --seed 20260809
done
