#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

TRAIN_SESSION="${TRAIN_SESSION:-advfd_mean_g_5k}"
MEAN_ROOT="${MEAN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_w3_b24_5k_v1}"
MEAN_EXP="${MEAN_EXP:-pmf_b_sim_advinc_full_d_mean_g_officialavg_w3_b24_q50k_5k}"
MEAN_CHECKPOINT="${MEAN_ROOT}/eqvae_advfd_reproduction/${MEAN_EXP}/checkpoints/step_0004999.pth"
STATIC_CHECKPOINT="${STATIC_CHECKPOINT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w3_b24_10k_v1/eqvae_advfd_static_reproduction/pmf_b_sim_static_officialavg_w3_b24_q50k_10k/checkpoints/step_0005000.pth}"
FULL_CHECKPOINT="${FULL_CHECKPOINT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/eqvae_advfd_reproduction/pmf_b_sim_advinc_officialavg_w3_b24_q50k_10k/checkpoints/step_0005000.pth}"
ROOT="${ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_decoupled_mean_g_paired5k_v1}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
REF_STATS="${REF_STATS:-/data/users/zhoushunyu/research_deps/advfd_reference_stats/guided_diffusion_stats.npz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"
GPU_IDS="${GPU_IDS:-0,1,2}"
EVAL_BATCH="${EVAL_BATCH:-32}"
FD_BATCH="${FD_BATCH:-16}"
NUM_IMAGES="${NUM_IMAGES:-5000}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
mkdir -p "${ROOT}"

while [[ ! -f "${MEAN_CHECKPOINT}" ]]; do
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    echo "Training ended without ${MEAN_CHECKPOINT}" >&2
    exit 1
  fi
  sleep 60
done
while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${PYTHONPATH:-}"

run_eval() {
  local condition="$1"
  local checkpoint="$2"
  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/run_official_pmf_eval.py \
    --eqvae-official-root "${OFFICIAL_ROOT}" \
    --eqvae-inception-stats "${REF_STATS}" \
    --eqvae-eval-manifest "${ROOT}/${condition}_manifest.json" \
    --eqvae-preserve-generated-images always \
    --project paired_eval --exp_name "${condition}" --output_dir "${ROOT}" \
    --load_from "${BASE_CHECKPOINT}" --resume_from "${checkpoint}" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --models inception --no_prc \
    --num_images "${NUM_IMAGES}" --eval_bsz "${EVAL_BATCH}" \
    --eval_ema_labels online --fid_stats_path "${REF_STATS}" \
    --disable_vis --disable_wandb \
    2>&1 | tee "${ROOT}/${condition}.log"
}

run_eval static_5k "${STATIC_CHECKPOINT}"
run_eval full_advfd_5k "${FULL_CHECKPOINT}"
run_eval full_d_mean_g_5k "${MEAN_CHECKPOINT}"

image_dir() {
  local condition="$1"
  printf '%s/paired_eval/%s/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7' \
    "${ROOT}" "${condition}"
}

for condition in static_5k full_advfd_5k full_d_mean_g_5k; do
  count="$(find "$(image_dir "${condition}")" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if (( count != NUM_IMAGES )); then
    echo "${condition} has ${count}/${NUM_IMAGES} images" >&2
    exit 1
  fi
done

cd "${OFFICIAL_ROOT}"
torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder \
    "$(image_dir static_5k)" \
    "$(image_dir full_advfd_5k)" \
    "$(image_dir full_d_mean_g_5k)" \
  --output_csv \
    "${ROOT}/static_5k_fdr3_raw.csv" \
    "${ROOT}/full_advfd_5k_fdr3_raw.csv" \
    "${ROOT}/full_d_mean_g_5k_fdr3_raw.csv" \
  --models \
    convnext \
    vit_large_patch14_dinov2.lvd142m \
    vit_large_patch14_clip_224.openai \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${ROOT}/heldout_fdr3.log"

cd /home/zhoushunyu/eqvae
python experiments/advfd_cleanroom/summarize_official_fdr3.py \
  --condition-csv "static_5k=${ROOT}/static_5k_fdr3_raw.csv" \
  --condition-csv "full_advfd_5k=${ROOT}/full_advfd_5k_fdr3_raw.csv" \
  --condition-csv "full_d_mean_g_5k=${ROOT}/full_d_mean_g_5k_fdr3_raw.csv" \
  --output-csv "${ROOT}/fdr3_summary.csv" \
  --output-json "${ROOT}/fdr3_summary.json" \
  2>&1 | tee "${ROOT}/fdr3_summary.log"

touch "${ROOT}/_SUCCESS"
