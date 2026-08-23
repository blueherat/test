#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

BLOCKING_SESSION="${BLOCKING_SESSION:-advfd_official_feature_audit_after_eval}"
ADVFD_TRAIN_ROOT="${ADVFD_TRAIN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1}"
AUDIT_ROOT="${AUDIT_ROOT:-${ADVFD_TRAIN_ROOT}/fresh_feature_audit5k}"
AUDIT_SUCCESS_MARKER="${AUDIT_ROOT}/_SUCCESS"
AUDIT_FAILURE_MARKER="${AUDIT_ROOT}/_FAILED"
GPU_IDS="${GPU_IDS:-1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w3_b24_10k_v1}"
TRAIN_EXP="${TRAIN_EXP:-pmf_b_sim_static_officialavg_w3_b24_q50k_10k}"
CHECKPOINT_ROOT="${TRAIN_ROOT}/eqvae_advfd_static_reproduction/${TRAIN_EXP}/checkpoints"
EVAL_ROOT="${EVAL_ROOT:-${TRAIN_ROOT}/official_eval5k}"
PIPELINE_SUCCESS_MARKER="${TRAIN_ROOT}/_PIPELINE_SUCCESS"
PIPELINE_FAILURE_MARKER="${TRAIN_ROOT}/_PIPELINE_FAILED"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
REF_STATS="${REF_STATS:-/data/users/zhoushunyu/research_deps/advfd_reference_stats/guided_diffusion_stats.npz}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"

mkdir -p "${TRAIN_ROOT}"
rm -f "${PIPELINE_SUCCESS_MARKER}" "${PIPELINE_FAILURE_MARKER}"
trap 'status=$?; if (( status != 0 )); then printf "exit_status=%s\n" "${status}" > "${PIPELINE_FAILURE_MARKER}"; fi' EXIT

while [[ ! -f "${AUDIT_SUCCESS_MARKER}" ]]; do
  if [[ -f "${AUDIT_FAILURE_MARKER}" ]]; then
    echo "Feature audit failed; refusing to launch the static control." >&2
    exit 1
  fi
  if ! tmux has-session -t "${BLOCKING_SESSION}" 2>/dev/null; then
    echo "Feature-audit session ended without success marker: ${AUDIT_SUCCESS_MARKER}" >&2
    exit 1
  fi
  sleep 60
done

if [[ ! -f "${CHECKPOINT_ROOT}/step_0009999.pth" ]]; then
  GPU_IDS="${GPU_IDS}" \
  LOCAL_BATCH=24 \
  QUEUE_SIZE=50000 \
  EPOCHS=8 \
  STEPS_PER_EPOCH=1250 \
  OUTPUT_ROOT="${TRAIN_ROOT}" \
  EXP_NAME="${TRAIN_EXP}" \
  bash experiments/advfd_cleanroom/run_pmf_b_official_static_code.sh
else
  echo "Static 10k checkpoint already exists; skipping training."
fi

for checkpoint in \
  "${CHECKPOINT_ROOT}/step_0005000.pth" \
  "${CHECKPOINT_ROOT}/step_0009999.pth"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing static-control checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${PYTHONPATH:-}"

run_eval() {
  local condition="$1"
  local checkpoint="$2"
  local summary="${EVAL_ROOT}/eqvae_advfd_static_official_eval/${condition}/final_eval_summary.csv"
  if [[ -f "${summary}" ]]; then
    echo "Static evaluation already exists; skipping ${condition}."
    return
  fi
  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/run_official_pmf_eval.py \
    --eqvae-official-root "${OFFICIAL_ROOT}" \
    --eqvae-inception-stats "${REF_STATS}" \
    --eqvae-eval-manifest "${EVAL_ROOT}/${condition}_manifest.json" \
    --project eqvae_advfd_static_official_eval \
    --exp_name "${condition}" \
    --output_dir "${EVAL_ROOT}" \
    --load_from "${BASE_CHECKPOINT}" \
    --resume_from "${checkpoint}" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --models inception --no_prc \
    --num_images 5000 --eval_bsz 32 \
    --eval_ema_labels online \
    --fid_stats_path "${REF_STATS}" \
    --disable_vis --disable_wandb \
    2>&1 | tee "${EVAL_ROOT}/${condition}.log"
}

mkdir -p "${EVAL_ROOT}"
run_eval step_0005000 "${CHECKPOINT_ROOT}/step_0005000.pth"
run_eval step_0009999 "${CHECKPOINT_ROOT}/step_0009999.pth"
touch "${PIPELINE_SUCCESS_MARKER}"
