#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

WAIT_SESSION="${WAIT_SESSION:-advfd_official_static_control_pipeline}"
STATIC10_ROOT="${STATIC10_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w3_b24_10k_v1}"
STATIC10_EVAL_ROOT="${STATIC10_ROOT}/official_eval5k/eqvae_advfd_static_official_eval"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
LOCAL_BATCH="${LOCAL_BATCH:-24}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
AUDIT_BATCH="${AUDIT_BATCH:-48}"
AUDIT_NPROC="${AUDIT_NPROC:-1}"
SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS:-125000}"
LONG_STEPS=(0005000 0010000 0015000 0020000 0024999)

ADV_ROOT="${ADV_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w4_b24_25k_v1}"
ADV_EXP="${ADV_EXP:-pmf_b_sim_advinc_officialavg_w4_b24_q50k_25k}"
ADV_PROJECT="eqvae_advfd_reproduction"
ADV_CHECKPOINT_ROOT="${ADV_ROOT}/${ADV_PROJECT}/${ADV_EXP}/checkpoints"
ADV_EVAL_ROOT="${ADV_ROOT}/official_eval5k"
ADV_EVAL50_ROOT="${ADV_ROOT}/official_eval50k"
ADV_AUDIT_ROOT="${ADV_ROOT}/fresh_feature_audit5k_calibrated"

STATIC_ROOT="${STATIC_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w4_b24_25k_v1}"
STATIC_EXP="${STATIC_EXP:-pmf_b_sim_static_officialavg_w4_b24_q50k_25k}"
STATIC_PROJECT="eqvae_advfd_static_reproduction"
STATIC_CHECKPOINT_ROOT="${STATIC_ROOT}/${STATIC_PROJECT}/${STATIC_EXP}/checkpoints"
STATIC_EVAL_ROOT="${STATIC_ROOT}/official_eval5k"
STATIC_EVAL50_ROOT="${STATIC_ROOT}/official_eval50k"

OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
REF_ROOT="${REF_ROOT:-/data/users/zhoushunyu/research_deps/advfd_reference_stats}"
REF_STATS="${REF_ROOT}/guided_diffusion_stats.npz"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"
SUCCESS_MARKER="${ADV_ROOT}/_FULL_25K_PIPELINE_SUCCESS"
FAILURE_MARKER="${ADV_ROOT}/_FULL_25K_PIPELINE_FAILED"

mkdir -p "${ADV_ROOT}"
rm -f "${SUCCESS_MARKER}" "${FAILURE_MARKER}"
trap 'status=$?; if (( status != 0 )); then printf "exit_status=%s\n" "${status}" > "${FAILURE_MARKER}"; fi' EXIT

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  sleep 60
done

for required in \
  "${STATIC10_EVAL_ROOT}/step_0005000/final_eval_summary.csv" \
  "${STATIC10_EVAL_ROOT}/step_0009999/final_eval_summary.csv"; do
  if [[ ! -f "${required}" ]]; then
    echo "The 10k static control did not finish cleanly: missing ${required}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:/home/zhoushunyu/eqvae:${PYTHONPATH:-}"

run_eval() {
  local project="$1"
  local eval_root="$2"
  local condition="$3"
  local checkpoint="$4"
  local num_images="${5:-${EVAL_SAMPLES}}"
  local preserve_images="${6:-auto}"
  local resume_args=()
  if [[ -n "${checkpoint}" ]]; then
    resume_args=(--resume_from "${checkpoint}")
  fi
  mkdir -p "${eval_root}"
  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/run_official_pmf_eval.py \
    --eqvae-official-root "${OFFICIAL_ROOT}" \
    --eqvae-inception-stats "${REF_STATS}" \
    --eqvae-eval-manifest "${eval_root}/${condition}_manifest.json" \
    --eqvae-preserve-generated-images "${preserve_images}" \
    --project "${project}" \
    --exp_name "${condition}" \
    --output_dir "${eval_root}" \
    --load_from "${BASE_CHECKPOINT}" \
    "${resume_args[@]}" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --models inception --no_prc \
    --num_images "${num_images}" --eval_bsz "${EVAL_BATCH}" \
    --eval_ema_labels online \
    --fid_stats_path "${REF_STATS}" \
    --disable_vis --disable_wandb \
    2>&1 | tee "${eval_root}/${condition}.log"
}

run_audit() {
  local condition="$1"
  local checkpoint="$2"
  local generated_folder
  generated_folder="$(find "${ADV_EVAL_ROOT}" -type d \
    -path "*/${condition}/eval_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7" \
    -print -quit)"
  if [[ -z "${generated_folder}" ]]; then
    echo "Generated folder not found for ${condition}" >&2
    exit 1
  fi
  torchrun --standalone --nproc-per-node="${AUDIT_NPROC}" \
    experiments/advfd_cleanroom/audit_official_advfd_features.py \
    --checkpoint "${checkpoint}" \
    --generated-folder "${generated_folder}" \
    --imagenet-root /data/shared/imagenet-1k \
    --real-split validation \
    --num-images "${EVAL_SAMPLES}" \
    --batch-size "${AUDIT_BATCH}" \
    --num-workers 4 \
    --output-json "${ADV_AUDIT_ROOT}/${condition}.json" \
    2>&1 | tee "${ADV_AUDIT_ROOT}/${condition}.log"
}

GPU_IDS="${GPU_IDS}" \
LOCAL_BATCH="${LOCAL_BATCH}" \
QUEUE_SIZE=50000 \
EPOCHS=20 \
STEPS_PER_EPOCH=1250 \
FD_ADV_START_STEP=1000 \
FD_ADV_WARMUP_STEPS=4000 \
FD_ADV_UPDATE_FREQ=2 \
PRINT_FREQ=20 \
SAVE_FREQ=5000 \
MILESTONE_INTERVAL=4 \
NUM_WORKERS=4 \
OUTPUT_ROOT="${ADV_ROOT}" \
EXP_NAME="${ADV_EXP}" \
SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS}" \
bash experiments/advfd_cleanroom/run_pmf_b_official_code.sh

run_eval eqvae_advfd_official_eval_25k "${ADV_EVAL_ROOT}" baseline ""
mkdir -p "${ADV_AUDIT_ROOT}"
for step in "${LONG_STEPS[@]}"; do
  checkpoint="${ADV_CHECKPOINT_ROOT}/step_${step}.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing AdvFD checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  run_eval eqvae_advfd_official_eval_25k "${ADV_EVAL_ROOT}" "step_${step}" "${checkpoint}"
  run_audit "step_${step}" "${checkpoint}"
done

run_eval eqvae_advfd_official_eval_50k "${ADV_EVAL50_ROOT}" baseline "" 50000 never
run_eval \
  eqvae_advfd_official_eval_50k \
  "${ADV_EVAL50_ROOT}" \
  step_0024999 \
  "${ADV_CHECKPOINT_ROOT}/step_0024999.pth" \
  50000 \
  never

GPU_IDS="${GPU_IDS}" \
LOCAL_BATCH="${LOCAL_BATCH}" \
QUEUE_SIZE=50000 \
EPOCHS=20 \
STEPS_PER_EPOCH=1250 \
OUTPUT_ROOT="${STATIC_ROOT}" \
EXP_NAME="${STATIC_EXP}" \
SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS}" \
bash experiments/advfd_cleanroom/run_pmf_b_official_static_code.sh

run_eval eqvae_advfd_static_official_eval_25k "${STATIC_EVAL_ROOT}" baseline ""
for step in "${LONG_STEPS[@]}"; do
  checkpoint="${STATIC_CHECKPOINT_ROOT}/step_${step}.pth"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing static checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  run_eval eqvae_advfd_static_official_eval_25k "${STATIC_EVAL_ROOT}" "step_${step}" "${checkpoint}"
done


run_eval \
  eqvae_advfd_static_official_eval_50k \
  "${STATIC_EVAL50_ROOT}" \
  step_0024999 \
  "${STATIC_CHECKPOINT_ROOT}/step_0024999.pth" \
  50000 \
  never

touch "${SUCCESS_MARKER}"
