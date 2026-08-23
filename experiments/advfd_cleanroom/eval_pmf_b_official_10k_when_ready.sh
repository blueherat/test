#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

TRAIN_SESSION="${TRAIN_SESSION:-advfd_official_pmf_b_10k_w3}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1}"
TRAIN_EXP="${TRAIN_EXP:-pmf_b_sim_advinc_officialavg_w3_b24_q50k_10k}"
CHECKPOINT_ROOT="${TRAIN_ROOT}/eqvae_advfd_reproduction/${TRAIN_EXP}/checkpoints"
FINAL_CHECKPOINT="${CHECKPOINT_ROOT}/step_0009999.pth"
GPU_IDS="${GPU_IDS:-1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
EVAL_ROOT="${EVAL_ROOT:-${TRAIN_ROOT}/official_eval5k}"
SUCCESS_MARKER="${EVAL_ROOT}/_SUCCESS"
FAILURE_MARKER="${EVAL_ROOT}/_FAILED"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
REF_STATS="${REF_STATS:-/data/users/zhoushunyu/research_deps/advfd_reference_stats/guided_diffusion_stats.npz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"

mkdir -p "${EVAL_ROOT}"
rm -f "${SUCCESS_MARKER}" "${FAILURE_MARKER}"
trap 'status=$?; if (( status != 0 )); then printf "exit_status=%s\n" "${status}" > "${FAILURE_MARKER}"; fi' EXIT

while [[ ! -f "${FINAL_CHECKPOINT}" ]]; do
  if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    echo "Training session ended without final checkpoint: ${FINAL_CHECKPOINT}" >&2
    exit 1
  fi
  sleep 60
done

# The public trainer writes checkpoints asynchronously.  Do not overlap the
# evaluator with the last checkpoint flush/process-group teardown, because the
# training ranks still hold nearly all device memory after the file appears.
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
  local resume_checkpoint="$2"
  local resume_args=()
  if [[ -n "${resume_checkpoint}" ]]; then
    resume_args=(--resume_from "${resume_checkpoint}")
  fi

  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/run_official_pmf_eval.py \
    --eqvae-official-root "${OFFICIAL_ROOT}" \
    --eqvae-inception-stats "${REF_STATS}" \
    --eqvae-eval-manifest "${EVAL_ROOT}/${condition}_manifest.json" \
    --project eqvae_advfd_official_eval \
    --exp_name "${condition}" \
    --output_dir "${EVAL_ROOT}" \
    --load_from "${BASE_CHECKPOINT}" \
    "${resume_args[@]}" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --models inception --no_prc \
    --num_images "${EVAL_SAMPLES}" --eval_bsz "${EVAL_BATCH}" \
    --eval_ema_labels online \
    --fid_stats_path "${REF_STATS}" \
    --disable_vis --disable_wandb \
    2>&1 | tee "${EVAL_ROOT}/${condition}.log"
}

run_eval baseline ""
run_eval step_0005000 "${CHECKPOINT_ROOT}/step_0005000.pth"
run_eval step_0009999 "${FINAL_CHECKPOINT}"
touch "${SUCCESS_MARKER}"
