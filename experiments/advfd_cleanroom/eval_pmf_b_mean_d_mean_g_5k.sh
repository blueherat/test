#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

TRAIN_SESSION="${TRAIN_SESSION:-advfd_mean_d_mean_g_5k}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_mean_d_mean_g_w3_b24_5k_v1}"
CANDIDATE_EXP="${CANDIDATE_EXP:-pmf_b_sim_advinc_mean_d_mean_g_officialavg_w3_b24_q50k_5k}"
CANDIDATE_CHECKPOINT="${CANDIDATE_ROOT}/eqvae_advfd_reproduction/${CANDIDATE_EXP}/checkpoints/step_0004999.pth"
ROOT="${ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_mean_d_mean_g_paired5k_v1}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
REF_STATS="${REF_STATS:-/data/users/zhoushunyu/research_deps/advfd_reference_stats/guided_diffusion_stats.npz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"
GPU_IDS="${GPU_IDS:-0,1,2}"
EVAL_BATCH="${EVAL_BATCH:-32}"
FD_BATCH="${FD_BATCH:-16}"
NUM_IMAGES="${NUM_IMAGES:-5000}"
CONDITION="mean_d_mean_g_5k"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
mkdir -p "${ROOT}"

while [[ ! -f "${CANDIDATE_CHECKPOINT}" ]]; do
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    echo "Training ended without ${CANDIDATE_CHECKPOINT}" >&2
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

torchrun --standalone --nproc-per-node="${NPROC}" \
  experiments/advfd_cleanroom/run_official_pmf_eval.py \
  --eqvae-official-root "${OFFICIAL_ROOT}" \
  --eqvae-inception-stats "${REF_STATS}" \
  --eqvae-eval-manifest "${ROOT}/${CONDITION}_manifest.json" \
  --eqvae-preserve-generated-images always \
  --project paired_eval --exp_name "${CONDITION}" --output_dir "${ROOT}" \
  --load_from "${BASE_CHECKPOINT}" --resume_from "${CANDIDATE_CHECKPOINT}" \
  --model pMF_B --rope_2d --learned_pe --disable_v_head \
  --cfg 8.5 --cfg_list 8.5 \
  --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
  --models inception --no_prc \
  --num_images "${NUM_IMAGES}" --eval_bsz "${EVAL_BATCH}" \
  --eval_ema_labels online --fid_stats_path "${REF_STATS}" \
  --disable_vis --disable_wandb \
  2>&1 | tee "${ROOT}/${CONDITION}.log"

IMAGE_DIR="${ROOT}/paired_eval/${CONDITION}/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
count="$(find "${IMAGE_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)"
if (( count != NUM_IMAGES )); then
  echo "${CONDITION} has ${count}/${NUM_IMAGES} images" >&2
  exit 1
fi

cd "${OFFICIAL_ROOT}"
torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${IMAGE_DIR}" \
  --output_csv "${ROOT}/${CONDITION}_fdr3_raw.csv" \
  --models \
    convnext \
    vit_large_patch14_dinov2.lvd142m \
    vit_large_patch14_clip_224.openai \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${ROOT}/${CONDITION}_heldout_fdr3.log"

cd /home/zhoushunyu/eqvae
python experiments/advfd_cleanroom/summarize_official_fdr3.py \
  --condition-csv "${CONDITION}=${ROOT}/${CONDITION}_fdr3_raw.csv" \
  --output-csv "${ROOT}/fdr3_summary.csv" \
  --output-json "${ROOT}/fdr3_summary.json" \
  2>&1 | tee "${ROOT}/fdr3_summary.log"

touch "${ROOT}/_SUCCESS"
