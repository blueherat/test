#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhoushunyu/eqvae"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
TRAIN_ROOT="${TRAIN_ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_velocity_control_10k_v1/pmf_b_local_velocity_w4_b18_10k}"
ROOT="${ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_velocity_control_paired5k_v1}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth}"
STATIC_CHECKPOINT="${STATIC_CHECKPOINT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_static_w3_b24_10k_v1/eqvae_advfd_static_reproduction/pmf_b_sim_static_officialavg_w3_b24_q50k_10k/checkpoints/step_0009999.pth}"
ADVFD_CHECKPOINT="${ADVFD_CHECKPOINT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_code_w3_b24_10k_v1/eqvae_advfd_reproduction/pmf_b_sim_advinc_officialavg_w3_b24_q50k_10k/checkpoints/step_0009999.pth}"
INCEPTION_STATS="${INCEPTION_STATS:-${OFFICIAL_ROOT}/data/fid_stats/guided_diffusion_stats.npz}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
GEN_BATCH="${GEN_BATCH:-32}"
FD_BATCH="${FD_BATCH:-16}"
NUM_IMAGES="${NUM_IMAGES:-5000}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
if (( NPROC < 1 )); then
  echo "GPU_IDS must contain at least one GPU" >&2
  exit 2
fi

VELOCITY_5K="${TRAIN_ROOT}/checkpoints/step_0005000.pth"
VELOCITY_10K="${TRAIN_ROOT}/checkpoints/step_0010000.pth"
for required in \
  "${OFFICIAL_ROOT}/eval_all_fds.py" \
  "${BASE_CHECKPOINT}" \
  "${STATIC_CHECKPOINT}" \
  "${ADVFD_CHECKPOINT}" \
  "${VELOCITY_5K}" \
  "${VELOCITY_10K}" \
  "${INCEPTION_STATS}" \
  "${OFFICIAL_ROOT}/data/fid_stats/convnext_in256_t224_stats.npz" \
  "${OFFICIAL_ROOT}/data/fid_stats/vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz" \
  "${OFFICIAL_ROOT}/data/fid_stats/vit_large_patch14_clip_224_openai_in256_t256_stats.npz"; do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required file: ${required}" >&2
    exit 1
  fi
done

mkdir -p "${ROOT}"
if [[ -f "${ROOT}/_SUCCESS" ]]; then
  echo "Evaluation already complete: ${ROOT}"
  exit 0
fi
for partial in \
  "${ROOT}/base_fdr3_raw.csv" \
  "${ROOT}/static_10k_fdr3_raw.csv" \
  "${ROOT}/advfd_10k_fdr3_raw.csv" \
  "${ROOT}/velocity_5k_fdr3_raw.csv" \
  "${ROOT}/velocity_10k_fdr3_raw.csv" \
  "${ROOT}/base_inception.csv" \
  "${ROOT}/static_10k_inception.csv" \
  "${ROOT}/advfd_10k_inception.csv" \
  "${ROOT}/velocity_5k_inception.csv" \
  "${ROOT}/velocity_10k_inception.csv"; do
  if [[ -f "${partial}" ]]; then
    echo "Refusing to append to partial evaluation output: ${partial}" >&2
    exit 1
  fi
done
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${REPO_ROOT}:${PYTHONPATH:-}"

image_dir() {
  local condition="$1"
  printf '%s/paired_generation/%s/gen_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7' \
    "${ROOT}" "${condition}"
}

generate_condition() {
  local condition="$1"
  local checkpoint="$2"
  local folder count
  folder="$(image_dir "${condition}")"
  count=0
  if [[ -d "${folder}" ]]; then
    count="$(find "${folder}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  fi
  if (( count == NUM_IMAGES )); then
    echo "Reusing ${count} images for ${condition}: ${folder}"
    return
  fi
  if (( count != 0 )); then
    echo "Refusing partial ${condition} folder with ${count}/${NUM_IMAGES} images" >&2
    exit 1
  fi

  local resume_args=()
  if [[ -n "${checkpoint}" ]]; then
    resume_args=(--resume_from "${checkpoint}")
  fi
  cd "${REPO_ROOT}"
  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/run_official_pmf_eval.py \
    --eqvae-official-root "${OFFICIAL_ROOT}" \
    --eqvae-inception-stats "${INCEPTION_STATS}" \
    --eqvae-eval-manifest "${ROOT}/${condition}_generation_manifest.json" \
    --eqvae-preserve-generated-images always \
    --gen_only \
    --project paired_generation --exp_name "${condition}" --output_dir "${ROOT}" \
    --load_from "${BASE_CHECKPOINT}" "${resume_args[@]}" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --num_images "${NUM_IMAGES}" --eval_bsz "${GEN_BATCH}" \
    --eval_ema_labels online --disable_vis --disable_wandb \
    2>&1 | tee "${ROOT}/${condition}_generation.log"
}

generate_condition base ""
generate_condition static_10k "${STATIC_CHECKPOINT}"
generate_condition advfd_10k "${ADVFD_CHECKPOINT}"
generate_condition velocity_5k "${VELOCITY_5K}"
generate_condition velocity_10k "${VELOCITY_10K}"

cd "${REPO_ROOT}"
python experiments/advfd_cleanroom/eval_pmf_velocity_mse.py \
  --official-root "${OFFICIAL_ROOT}" \
  --packed-data /data/shared/imagenet-1k/random_access_v1 \
  --condition "base=${BASE_CHECKPOINT}" \
  --condition "static_10k=${STATIC_CHECKPOINT}" \
  --condition "advfd_10k=${ADVFD_CHECKPOINT}" \
  --condition "velocity_5k=${VELOCITY_5K}" \
  --condition "velocity_10k=${VELOCITY_10K}" \
  --output-json "${ROOT}/continuation_unseen_velocity_mse.json" \
  --output-csv "${ROOT}/continuation_unseen_velocity_mse.csv" \
  --samples 1024 --batch-size 16 --num-workers 4 --seed 260823 --device cuda:0 \
  --continuation-samples-seen 720000 \
  --continuation-sampler-seed 0 --continuation-world-size 4 \
  2>&1 | tee "${ROOT}/continuation_unseen_velocity_mse.log"

python - "${ROOT}" "${NPROC}" "${NUM_IMAGES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
world_size = int(sys.argv[2])
num_images = int(sys.argv[3])
names = ("base", "static_10k", "advfd_10k", "velocity_5k", "velocity_10k")
manifests = {
    name: json.loads((root / f"{name}_generation_manifest.json").read_text())
    for name in names
}
fields = (
    "seed", "eval_bsz", "num_images", "num_classes", "cfg_list",
    "interval_min", "interval_max", "num_sampling_steps", "eval_ema_labels",
    "model", "rope_2d", "learned_pe", "disable_v_head",
)
for name, manifest in manifests.items():
    assert manifest["world_size"] == world_size, name
    assert manifest["official_arguments"]["num_images"] == num_images, name
for field in fields:
    values = {
        json.dumps(manifest["official_arguments"][field], sort_keys=True)
        for manifest in manifests.values()
    }
    if len(values) != 1:
        raise SystemExit(f"paired generation mismatch for {field}: {values}")
(root / "paired_generation_audit.json").write_text(
    json.dumps(
        {
            "conditions": list(names),
            "compared_fields": list(fields),
            "num_images": num_images,
            "world_size": world_size,
            "pairing_basis": (
                "fresh process per condition with identical seed, world size, "
                "class order, batch size, and sampler arguments"
            ),
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

BASE_IMAGES="$(image_dir base)"
STATIC_IMAGES="$(image_dir static_10k)"
ADVFD_IMAGES="$(image_dir advfd_10k)"
V5_IMAGES="$(image_dir velocity_5k)"
V10_IMAGES="$(image_dir velocity_10k)"

cd "${OFFICIAL_ROOT}"
torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${BASE_IMAGES}" "${STATIC_IMAGES}" "${ADVFD_IMAGES}" "${V5_IMAGES}" "${V10_IMAGES}" \
  --output_csv \
    "${ROOT}/base_fdr3_raw.csv" \
    "${ROOT}/static_10k_fdr3_raw.csv" \
    "${ROOT}/advfd_10k_fdr3_raw.csv" \
    "${ROOT}/velocity_5k_fdr3_raw.csv" \
    "${ROOT}/velocity_10k_fdr3_raw.csv" \
  --models \
    convnext \
    vit_large_patch14_dinov2.lvd142m \
    vit_large_patch14_clip_224.openai \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${ROOT}/heldout_fdr3.log"

torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${BASE_IMAGES}" "${STATIC_IMAGES}" "${ADVFD_IMAGES}" "${V5_IMAGES}" "${V10_IMAGES}" \
  --output_csv \
    "${ROOT}/base_inception.csv" \
    "${ROOT}/static_10k_inception.csv" \
    "${ROOT}/advfd_10k_inception.csv" \
    "${ROOT}/velocity_5k_inception.csv" \
    "${ROOT}/velocity_10k_inception.csv" \
  --models inception \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${ROOT}/paired_inception.log"

cd "${REPO_ROOT}"
python experiments/advfd_cleanroom/summarize_official_fdr3.py \
  --condition-csv "base=${ROOT}/base_fdr3_raw.csv" \
  --condition-csv "static_10k=${ROOT}/static_10k_fdr3_raw.csv" \
  --condition-csv "advfd_10k=${ROOT}/advfd_10k_fdr3_raw.csv" \
  --condition-csv "velocity_5k=${ROOT}/velocity_5k_fdr3_raw.csv" \
  --condition-csv "velocity_10k=${ROOT}/velocity_10k_fdr3_raw.csv" \
  --output-csv "${ROOT}/fdr3_summary.csv" \
  --output-json "${ROOT}/fdr3_summary.json" \
  2>&1 | tee "${ROOT}/fdr3_summary.log"

touch "${ROOT}/_SUCCESS"
