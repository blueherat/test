#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

MODE="${1:-benchmark}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="$(awk -F, '{print NF}' <<<"${GPUS}")"
RUN_DIR="${RUN_DIR:-/home/zhoushunyu/data/eqvae/imagenette_sdvae_ldm256/formal_single_head_180m_v1}"
CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/home/zhoushunyu/data/eqvae/torchinductor_cache/imagenette_ldm256_180m}"
CHANNELS="${CHANNELS:-224,448,896}"
PER_GPU_BATCH="${PER_GPU_BATCH:-12}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-2}"
STEPS="${STEPS:-40000}"
MAX_USED_MIB="${MAX_USED_MIB:-512}"

IFS=',' read -r -a GPU_LIST <<<"${GPUS}"
for gpu in "${GPU_LIST[@]}"; do
  used_mib="$(
    nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits |
      tr -d ' '
  )"
  if (( used_mib > MAX_USED_MIB )); then
    echo "GPU ${gpu} is using ${used_mib} MiB (limit ${MAX_USED_MIB} MiB); aborting." >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPUS}"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_DIR}"
export PYTHONUNBUFFERED=1
mkdir -p "${RUN_DIR}" "${CACHE_DIR}"

case "${MODE}" in
  benchmark)
    torchrun --standalone --nproc_per_node="${NPROC}" \
      experiments/imagenette_sdvae_latent_diffusion.py benchmark \
      --image-size 256 \
      --channels "${CHANNELS}" \
      --batch-size "${PER_GPU_BATCH}" \
      --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
      --warmup 3 \
      --measured-steps 10 \
      --compile \
      --include-ema \
      2>&1 | tee "${RUN_DIR}/benchmark_${NPROC}gpu.log"
    ;;
  train)
    torchrun --standalone --nproc_per_node="${NPROC}" \
      experiments/imagenette_sdvae_latent_diffusion.py train \
      --data-root /data/shared/imagenette2-320 \
      --output-dir "${RUN_DIR}" \
      --image-size 256 \
      --channels "${CHANNELS}" \
      --batch-size "${PER_GPU_BATCH}" \
      --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
      --steps "${STEPS}" \
      --learning-rate 1e-4 \
      --weight-decay 0.01 \
      --warmup-steps 2000 \
      --class-dropout 0.1 \
      --ema-decay 0.9999 \
      --gradient-clip 1.0 \
      --num-workers 4 \
      --validation-count 2048 \
      --validation-every 1000 \
      --log-every 100 \
      --save-every 10000 \
      --extra-save-steps 1000,2000,5000 \
      --compile \
      --allow-tf32 \
      --resume \
      2>&1 | tee -a "${RUN_DIR}/train.log"
    ;;
  *)
    echo "usage: $0 [benchmark|train]" >&2
    exit 2
    ;;
esac
