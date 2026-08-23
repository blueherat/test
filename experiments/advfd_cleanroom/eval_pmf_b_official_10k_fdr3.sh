#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhoushunyu/eqvae"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
ROOT="${ROOT:-/data/users/zhoushunyu/eqvae/experiments/advfd_pmf_b_official_10k_fdr3_paired5k_v1}"
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
  exit 1
fi

for required in \
  "${OFFICIAL_ROOT}/eval_all_fds.py" \
  "${BASE_CHECKPOINT}" \
  "${STATIC_CHECKPOINT}" \
  "${ADVFD_CHECKPOINT}" \
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
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/home/zhoushunyu/.cache/huggingface/token}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${REPO_ROOT}:${PYTHONPATH:-}"

image_dir() {
  local condition="$1"
  printf '%s/paired_generation/%s/gen_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7' \
    "${ROOT}" "${condition}"
}

generate_condition() {
  local condition="$1"
  local checkpoint="$2"
  local folder
  folder="$(image_dir "${condition}")"
  local count=0
  if [[ -d "${folder}" ]]; then
    count="$(find "${folder}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  fi
  if (( count == NUM_IMAGES )); then
    echo "Reusing ${count} paired images for ${condition}: ${folder}"
    return
  fi
  if (( count != 0 )); then
    echo "Refusing partial ${condition} folder with ${count}/${NUM_IMAGES} images: ${folder}" >&2
    exit 1
  fi

  cd "${REPO_ROOT}"
  torchrun --standalone --nproc-per-node="${NPROC}" \
    experiments/advfd_cleanroom/run_official_pmf_eval.py \
    --eqvae-official-root "${OFFICIAL_ROOT}" \
    --eqvae-inception-stats "${INCEPTION_STATS}" \
    --eqvae-eval-manifest "${ROOT}/${condition}_generation_manifest.json" \
    --gen_only \
    --project paired_generation --exp_name "${condition}" --output_dir "${ROOT}" \
    --load_from "${BASE_CHECKPOINT}" --resume_from "${checkpoint}" \
    --model pMF_B --rope_2d --learned_pe --disable_v_head \
    --cfg 8.5 --cfg_list 8.5 \
    --interval_min 0.1 --interval_max 0.7 --num_sampling_steps 1 \
    --num_images "${NUM_IMAGES}" --eval_bsz "${GEN_BATCH}" \
    --eval_ema_labels online --disable_vis --disable_wandb \
    2>&1 | tee "${ROOT}/${condition}_generation.log"
}

generate_condition static "${STATIC_CHECKPOINT}"
generate_condition advfd "${ADVFD_CHECKPOINT}"

python - "${ROOT}" "${NPROC}" "${NUM_IMAGES}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_world_size = int(sys.argv[2])
expected_images = int(sys.argv[3])
manifests = {
    name: json.loads((root / f"{name}_generation_manifest.json").read_text())
    for name in ("static", "advfd")
}
fields = (
    "seed",
    "eval_bsz",
    "num_images",
    "num_classes",
    "cfg_list",
    "interval_min",
    "interval_max",
    "num_sampling_steps",
    "eval_ema_labels",
    "model",
    "rope_2d",
    "learned_pe",
    "disable_v_head",
)
for name, manifest in manifests.items():
    if manifest["world_size"] != expected_world_size:
        raise SystemExit(f"{name}: world_size mismatch")
    if manifest["official_arguments"]["num_images"] != expected_images:
        raise SystemExit(f"{name}: num_images mismatch")
for field in fields:
    values = {json.dumps(m["official_arguments"][field], sort_keys=True) for m in manifests.values()}
    if len(values) != 1:
        raise SystemExit(f"paired generation mismatch for {field}: {values}")
audit = {
    "pairing_basis": "fresh process per condition with identical seed, world size, class order, batch size, and sampler arguments",
    "compared_fields": list(fields),
    "world_size": expected_world_size,
    "num_images": expected_images,
    "static_checkpoint": manifests["static"]["official_arguments"]["resume_from"],
    "advfd_checkpoint": manifests["advfd"]["official_arguments"]["resume_from"],
}
(root / "paired_generation_audit.json").write_text(
    json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(audit, indent=2, sort_keys=True))
PY

STATIC_IMAGES="$(image_dir static)"
ADVFD_IMAGES="$(image_dir advfd)"
STATIC_CSV="${ROOT}/static_fdr3_raw.csv"
ADVFD_CSV="${ROOT}/advfd_fdr3_raw.csv"
if [[ -f "${STATIC_CSV}" || -f "${ADVFD_CSV}" ]]; then
  echo "Refusing to append to an existing partial FDr3 CSV under ${ROOT}" >&2
  exit 1
fi

cd "${OFFICIAL_ROOT}"
torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${STATIC_IMAGES}" "${ADVFD_IMAGES}" \
  --output_csv "${STATIC_CSV}" "${ADVFD_CSV}" \
  --models \
    convnext \
    vit_large_patch14_dinov2.lvd142m \
    vit_large_patch14_clip_224.openai \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${ROOT}/heldout_fdr3.log"

# Re-evaluate the same paired PNGs in both Inception reference conventions.
# This keeps a tiny ADM-FID delta from being compared against FDr3 on a
# different random sample set.
STATIC_INCEPTION_CSV="${ROOT}/static_inception_paired.csv"
ADVFD_INCEPTION_CSV="${ROOT}/advfd_inception_paired.csv"
if [[ -f "${STATIC_INCEPTION_CSV}" || -f "${ADVFD_INCEPTION_CSV}" ]]; then
  echo "Refusing to append to an existing paired Inception CSV under ${ROOT}" >&2
  exit 1
fi

torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${STATIC_IMAGES}" "${ADVFD_IMAGES}" \
  --output_csv "${STATIC_INCEPTION_CSV}" "${ADVFD_INCEPTION_CSV}" \
  --models inception \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${ROOT}/paired_inception.log"

cd "${REPO_ROOT}"
python experiments/advfd_cleanroom/summarize_official_fdr3.py \
  --condition-csv "static=${STATIC_CSV}" \
  --condition-csv "advfd=${ADVFD_CSV}" \
  --output-csv "${ROOT}/fdr3_summary.csv" \
  --output-json "${ROOT}/fdr3_summary.json" \
  2>&1 | tee "${ROOT}/fdr3_summary.log"

sha256sum \
  "${OFFICIAL_ROOT}/data/fid_stats/convnext_in256_t224_stats.npz" \
  "${OFFICIAL_ROOT}/data/fid_stats/vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz" \
  "${OFFICIAL_ROOT}/data/fid_stats/vit_large_patch14_clip_224_openai_in256_t256_stats.npz" \
  "${OFFICIAL_ROOT}/data/fid_stats/jit_in256_stats.npz" \
  "${INCEPTION_STATS}" \
  > "${ROOT}/reference_stats_sha256.txt"
touch "${ROOT}/_SUCCESS"
