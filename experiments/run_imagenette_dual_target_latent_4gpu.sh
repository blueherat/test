#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/imagenette_dual_target_latent/formal_v2}"
CACHE_ROOT="${CACHE_ROOT:-/home/zhoushunyu/data/eqvae/imagenette_sdvae_latents_64}"
INDUCTOR_CACHE="${INDUCTOR_CACHE:-/home/zhoushunyu/data/eqvae/torchinductor_cache/imagenette_dual_reduce_b1536}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1536}"
BASE_CHANNELS="${BASE_CHANNELS:-96}"
LOSS_SPACE="${LOSS_SPACE:-v}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${INDUCTOR_CACHE}"

seeds=(20260809 20260810 20260811 20260812)
pids=()

for gpu in 0 1 2 3; do
  seed="${seeds[$gpu]}"
  run_dir="${OUTPUT_ROOT}/seed${seed}"
  mkdir -p "${run_dir}"
  echo "[launch] gpu=${gpu} seed=${seed} output=${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  CUDA_MODULE_LOADING=LAZY \
  OMP_NUM_THREADS=4 \
  PYTHONUNBUFFERED=1 \
  TORCHINDUCTOR_CACHE_DIR="${INDUCTOR_CACHE}" \
    python experiments/imagenette_dual_target_latent.py train \
      --device cuda:0 \
      --cache-root "${CACHE_ROOT}" \
      --output-dir "${run_dir}" \
      --seed "${seed}" \
      --base-channels "${BASE_CHANNELS}" \
      --batch-size "${BATCH_SIZE}" \
      --steps "${STEPS}" \
      --learning-rate 2e-4 \
      --loss-space "${LOSS_SPACE}" \
      --t-min 0.05 \
      --t-max 0.95 \
      --ema-decay 0.9999 \
      --gradient-clip 10 \
      --log-every 100 \
      --save-every 1000 \
      --extra-save-steps 200,500,1000 \
      --compile \
      --compile-mode reduce-overhead \
      --allow-tf32 \
      --resume \
      >"${run_dir}/train.log" 2>&1 &
  pids+=("$!")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "[failed] gpu=${index} pid=${pids[$index]}" >&2
    status=1
  fi
done

exit "${status}"
