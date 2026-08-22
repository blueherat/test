#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

DATA_ROOT=/home/zhoushunyu/data/eqvae/imagenet_sit_flow
RUN_ROOT=${DATA_ROOT}/runs
OUTPUT_ROOT=${DATA_ROOT}/moment_residual_pilot20k
NATIVE_CKPT=${RUN_ROOT}/sit-s-2_moment-paired-native-pilot20k_seed0/checkpoints/step_00020000.pt
RESIDUAL_CKPT=${RUN_ROOT}/sit-s-2_moment-diagonal-lmmse-pilot20k_seed0/checkpoints/step_00020000.pt
REFERENCE=${DATA_ROOT}/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz
ADM_PYTHON=/data/shared/envs/adm-fid/bin/python

mkdir -p "${OUTPUT_ROOT}/audit" "${OUTPUT_ROOT}/fid5k"

while [[ ! -f "${NATIVE_CKPT}" || ! -f "${RESIDUAL_CKPT}" ]]; do
  sleep 20
done

CUDA_VISIBLE_DEVICES=0 python experiments/analyze_imagenet100_sit_moment_residual_pair.py \
  --native-checkpoint "${NATIVE_CKPT}" \
  --residual-checkpoint "${RESIDUAL_CKPT}" \
  --weights model \
  --samples 5000 \
  --batch-size 128 \
  --output "${OUTPUT_ROOT}/audit/paired_model.json" \
  > /tmp/sit_moment_pair_audit_model.log 2>&1

sample_condition() {
  local gpus=$1
  local checkpoint=$2
  local weights=$3
  local name=$4
  CUDA_VISIBLE_DEVICES="${gpus}" torchrun --standalone --nproc_per_node=2 \
    experiments/sample_imagenet100_sit_fid.py \
    --checkpoint "${checkpoint}" \
    --weights "${weights}" \
    --output-dir "${OUTPUT_ROOT}/fid5k/${name}" \
    --num-samples 5000 \
    --per-rank-batch-size 32 \
    --vae-decode-batch-size 4 \
    --cuda-allocator-limit-gib 5 \
    --global-seed 0 \
    --num-output-points 250 \
    --atol 1e-6 \
    --rtol 1e-3 \
    --precision fp32 \
    > "/tmp/sit_moment_${name}_sample.log" 2>&1
}

sample_condition 0,1 "${NATIVE_CKPT}" model native_model &
native_model_pid=$!
sample_condition 2,3 "${RESIDUAL_CKPT}" model residual_model &
residual_model_pid=$!
wait "${native_model_pid}"
wait "${residual_model_pid}"

for name in native_model residual_model; do
  CUDA_VISIBLE_DEVICES=0 "${ADM_PYTHON}" experiments/compute_adm_fid.py \
    --reference "${REFERENCE}" \
    --samples "${OUTPUT_ROOT}/fid5k/${name}/samples_unguided_n5000.npz" \
    --batch-size 16 \
    --gpu-memory-fraction 0.15 \
    --output "${OUTPUT_ROOT}/fid5k/${name}/adm_metrics.json" \
    > "/tmp/sit_moment_${name}_fid.log" 2>&1
done
