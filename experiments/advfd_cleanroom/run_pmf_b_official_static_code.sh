#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

GPU_IDS="${GPU_IDS:-1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
LOCAL_BATCH="${LOCAL_BATCH:-24}"
QUEUE_SIZE="${QUEUE_SIZE:-50000}"
EPOCHS="${EPOCHS:-8}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1250}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w3_b24_10k_v1}"
EXP_NAME="${EXP_NAME:-pmf_b_sim_static_officialavg_w3_b24_q50k_10k}"
GRADIENT_REDUCTION="${GRADIENT_REDUCTION:-official_avg}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SCHEDULE_TOTAL_STEPS="${SCHEDULE_TOTAL_STEPS:-}"
PRINT_FREQ="${PRINT_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MILESTONE_INTERVAL="${MILESTONE_INTERVAL:-4}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
REF_ROOT="${REF_ROOT:-/data/users/zhoushunyu/research_deps/advfd_reference_stats}"
PACKED_DATA="${PACKED_DATA:-/data/shared/imagenet-1k/random_access_v1}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${PYTHONPATH:-}"

SCHEDULE_ARGS=()
if [[ -n "${SCHEDULE_TOTAL_STEPS}" ]]; then
  SCHEDULE_ARGS=(--eqvae-lr-schedule-total-steps "${SCHEDULE_TOTAL_STEPS}")
fi

mkdir -p "${OUTPUT_ROOT}"

torchrun --standalone --nproc-per-node="${NPROC}" \
  experiments/advfd_cleanroom/run_official_advfd_packed.py \
  --eqvae-official-root "${OFFICIAL_ROOT}" \
  --eqvae-packed-data "${PACKED_DATA}" \
  --eqvae-gradient-reduction "${GRADIENT_REDUCTION}" \
  --eqvae-adapter-manifest "${OUTPUT_ROOT}/${EXP_NAME}_adapter_manifest.json" \
  "${SCHEDULE_ARGS[@]}" \
  --project eqvae_advfd_static_reproduction \
  --exp_name "${EXP_NAME}" \
  --output_dir "${OUTPUT_ROOT}" \
  --batch_size "${LOCAL_BATCH}" \
  --data_path "${PACKED_DATA}" \
  --load_from /data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth \
  --model pMF_B --rope_2d --learned_pe --disable_v_head \
  --cfg 8.5 --interval_min 0.1 --interval_max 0.7 \
  --num_sampling_steps 1 \
  --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}" \
  --warmup_epochs 5 \
  --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
  --grad_checkpointing \
  --fd_repr_grad_checkpoint_models siglip \
  --fd_repr_models \
    vit_so400m_patch16_siglip_256.v2_webli \
    vit_large_patch16_224.mae \
    inception \
  --fd_repr_pool_types cls cls cls \
  --fd_target_sizes 224 224 256 \
  --fd_repr_stats_paths \
    "${REF_ROOT}/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz" \
    "${REF_ROOT}/vit_large_patch16_224_mae_in256_t224_stats.npz" \
    "${REF_ROOT}/guided_diffusion_stats.npz" \
  --fid_stats_path "${REF_ROOT}/guided_diffusion_stats.npz" \
  --fd_eigvalsh --fd_ema_beta 0.999 \
  --queue_size "${QUEUE_SIZE}" --fd_queue_fill_bsz "${LOCAL_BATCH}" \
  --num_workers "${NUM_WORKERS}" --pin_mem \
  --print_freq "${PRINT_FREQ}" --save_freq "${SAVE_FREQ}" --milestone_interval "${MILESTONE_INTERVAL}" \
  --disable_vis --disable_wandb --auto_resume \
  2>&1 | tee -a "${OUTPUT_ROOT}/${EXP_NAME}.log"
