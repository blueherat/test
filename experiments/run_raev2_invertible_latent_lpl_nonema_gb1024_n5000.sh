#!/usr/bin/env bash

set -euo pipefail

ROOT="/home/zhoushunyu/eqvae"
TORCHRUN="/home/zhoushunyu/miniconda3/envs/myenv/bin/torchrun"
RESULT_ROOT="/home/zhoushunyu/data/eqvae/experiments/raev2_invertible_latent_lpl"
CONFIG="${ROOT}/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
SOURCE="/home/zhoushunyu/data/eqvae/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"
DATA="/data/shared/imagenet-1k"
PACKED_DATA="/data/shared/imagenet-1k/random_access_v1"
INDEX_MAP="/home/zhoushunyu/data/eqvae/datasets/raev2_imagenet_train_lexicographic_indices.npy"
FID_REFERENCE="/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz"

# This matches the sample budget of 10,000 updates at global batch 16.
TARGET_UPDATES=160
SAVE_EVERY=32
GLOBAL_BATCH=1024
MICRO_BATCH=1
SAMPLES=5000
SAMPLE_SEED=20260803
GUIDANCE_SCALE=1.78
MIN_FREE_GIB="${MIN_FREE_GIB:-0.5}"

LPL_EXPERIMENT="lpl_model_gb1024_u160_v1"
FLOW_EXPERIMENT="flow_model_gb1024_u160_v1"
ENDPOINTS="${RESULT_ROOT}/nonema_endpoints_n5000_seed${SAMPLE_SEED}_s178_v1"
FID_OUTPUT="${RESULT_ROOT}/endpoint_fid_model_gb1024_u32_160_n5000_s178_v1"

mkdir -p "${RESULT_ROOT}"
cd "${ROOT}"

checkpoint_update() {
    local path="$1"
    local name
    local digits
    name="$(basename "${path}")"
    name="${name#update_}"
    digits="${name%.pt}"
    printf '%d\n' "$((10#${digits}))"
}

latest_checkpoint() {
    local experiment="$1"
    local checkpoint_dir="${RESULT_ROOT}/${experiment}/checkpoints"
    if [[ -d "${checkpoint_dir}" ]]; then
        find "${checkpoint_dir}" -maxdepth 1 -type f -name 'update_*.pt' -print \
            | sort | tail -n 1
    fi
}

train_branch() {
    local objective="$1"
    local experiment="$2"
    local resume
    local update=0
    resume="$(latest_checkpoint "${experiment}")"
    if [[ -n "${resume}" ]]; then
        update="$(checkpoint_update "${resume}")"
    fi
    if (( update >= TARGET_UPDATES )); then
        printf '[%s] %s already reached update %d; skipping.\n' \
            "$(date --iso-8601=seconds)" "${objective}" "${update}"
        return
    fi

    printf '[%s] training %s from update %d to %d with global batch %d.\n' \
        "$(date --iso-8601=seconds)" "${objective}" "${update}" \
        "${TARGET_UPDATES}" "${GLOBAL_BATCH}"
    local command=(
        "${TORCHRUN}" --standalone --nproc_per_node=4
        experiments/train_raev2_invertible_latent_lpl.py
        --config "${CONFIG}"
        --data-path "${DATA}"
        --packed-data-path "${PACKED_DATA}"
        --index-map "${INDEX_MAP}"
        --results-dir "${RESULT_ROOT}"
        --experiment-name "${experiment}"
        --source-checkpoint "${SOURCE}"
        --source-state-key model
        --objective "${objective}"
        --max-updates "${TARGET_UPDATES}"
        --save-every "${SAVE_EVERY}"
        --global-batch-size "${GLOBAL_BATCH}"
        --micro-batch-size "${MICRO_BATCH}"
        --blocks 2
        --hidden-channels 32
        --learning-rate 1e-5
        --weight-decay 0
        --lpl-variant prediction_full
        --lpl-prediction-target full
        --data-identity-weight 0.001
        --noise-identity-weight 0.001
        --precision bf16
        --num-workers 0
        --min-free-gib "${MIN_FREE_GIB}"
    )
    if [[ -n "${resume}" ]]; then
        command+=(--resume "${resume}")
    fi
    if [[ "${objective}" == "lpl" ]]; then
        command+=(--lpl-weight 5.121181994940295e-5)
    fi
    "${command[@]}"
}

sample_shared_endpoints() {
    printf '[%s] sampling %d shared non-EMA endpoints at scale %.2f.\n' \
        "$(date --iso-8601=seconds)" "${SAMPLES}" "${GUIDANCE_SCALE}"
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
        experiments/run_raev2_scale_response_study.py \
        --config "${CONFIG}" \
        --checkpoint "${SOURCE}" \
        --packed-data-path "${PACKED_DATA}" \
        --parquet-data-path "${DATA}" \
        --fid-reference "${FID_REFERENCE}" \
        --output-dir "${ENDPOINTS}" \
        --samples "${SAMPLES}" \
        --scale "${GUIDANCE_SCALE}" \
        --per-rank-batch 2 \
        --log-every-batches 25 \
        --state-key model \
        --precision bf16 \
        --seed "${SAMPLE_SEED}" \
        --endpoints-only
}

evaluate_all_checkpoints() {
    local adapter_args=()
    local update
    local lpl_path
    local flow_path
    for update in 32 64 96 128 160; do
        printf -v lpl_path '%s/%s/checkpoints/update_%07d.pt' \
            "${RESULT_ROOT}" "${LPL_EXPERIMENT}" "${update}"
        printf -v flow_path '%s/%s/checkpoints/update_%07d.pt' \
            "${RESULT_ROOT}" "${FLOW_EXPERIMENT}" "${update}"
        [[ -f "${lpl_path}" ]] || { echo "Missing ${lpl_path}"; exit 1; }
        [[ -f "${flow_path}" ]] || { echo "Missing ${flow_path}"; exit 1; }
        adapter_args+=(--adapter "lpl_gb1024_u${update}=${lpl_path}")
        adapter_args+=(--adapter "flow_gb1024_u${update}=${flow_path}")
    done

    printf '[%s] decoding identity and all checkpoints on the same 5k endpoints.\n' \
        "$(date --iso-8601=seconds)"
    "${TORCHRUN}" --standalone --nproc_per_node=4 \
        experiments/evaluate_raev2_invertible_latent_endpoints.py \
        --config "${CONFIG}" \
        --source-checkpoint "${SOURCE}" \
        --source-state-key model \
        --endpoint-dir "${ENDPOINTS}" \
        --scale "${GUIDANCE_SCALE}" \
        --adapter-state-key adapter \
        --fid-reference "${FID_REFERENCE}" \
        --output-dir "${FID_OUTPUT}" \
        --decode-batch 4 \
        --precision bf16 \
        "${adapter_args[@]}"
}

printf '[%s] starting global-batch-1024 paired-5k non-EMA pipeline.\n' \
    "$(date --iso-8601=seconds)"
train_branch lpl "${LPL_EXPERIMENT}"
train_branch flow "${FLOW_EXPERIMENT}"
sample_shared_endpoints
evaluate_all_checkpoints
printf '[%s] pipeline completed successfully.\n' "$(date --iso-8601=seconds)"
