#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/zhoushunyu/eqvae"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-/data/users/zhoushunyu/research_repos/AdvFD}"
SOURCE_ROOT="${SOURCE_ROOT:?Set SOURCE_ROOT to the paired-generation root}"
RESULT_ROOT="${RESULT_ROOT:-${SOURCE_ROOT}/exact_evaluation}"
GPU_IDS="${GPU_IDS:-1,2}"
FD_BATCH="${FD_BATCH:-16}"
NUM_IMAGES="${NUM_IMAGES:-5000}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${#GPU_ARRAY[@]}"
if (( NPROC < 1 || NUM_IMAGES % NPROC != 0 )); then
  echo "NUM_IMAGES=${NUM_IMAGES} must be divisible by evaluation NPROC=${NPROC}" >&2
  exit 2
fi

image_dir() {
  local condition="$1"
  printf '%s/paired_generation/%s/gen_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7' \
    "${SOURCE_ROOT}" "${condition}"
}

CONDITIONS=(base static_10k advfd_10k velocity_5k velocity_10k)
IMAGE_DIRS=()
for condition in "${CONDITIONS[@]}"; do
  folder="$(image_dir "${condition}")"
  count="$(find "${folder}" -maxdepth 1 -type f -name '*.png' | wc -l)"
  if (( count != NUM_IMAGES )); then
    echo "Expected ${NUM_IMAGES} images for ${condition}, found ${count}: ${folder}" >&2
    exit 1
  fi
  IMAGE_DIRS+=("${folder}")
done

mkdir -p "${RESULT_ROOT}"
if [[ -f "${RESULT_ROOT}/_SUCCESS" ]]; then
  echo "Exact evaluation already complete: ${RESULT_ROOT}"
  exit 0
fi

FDR3_CSVS=()
INCEPTION_CSVS=()
for condition in "${CONDITIONS[@]}"; do
  FDR3_CSVS+=("${RESULT_ROOT}/${condition}_fdr3_raw.csv")
  INCEPTION_CSVS+=("${RESULT_ROOT}/${condition}_inception.csv")
done
for output in "${FDR3_CSVS[@]}" "${INCEPTION_CSVS[@]}"; do
  if [[ -f "${output}" ]]; then
    echo "Refusing to append to partial output: ${output}" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export HF_HOME="${HF_HOME:-/data/users/zhoushunyu/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/data/users/zhoushunyu/torch_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="/data/users/zhoushunyu/research_deps/muon_optimizer_0_1_0:/data/users/zhoushunyu/research_deps/timm_1_0_28:${REPO_ROOT}:${PYTHONPATH:-}"

python - "${RESULT_ROOT}/evaluation_protocol.json" "${SOURCE_ROOT}" \
  "${NUM_IMAGES}" "${NPROC}" "${GPU_IDS}" <<'PY'
import json
import sys
from pathlib import Path

path, source_root, num_images, world_size, gpu_ids = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "source_root": source_root,
            "num_images_per_condition": int(num_images),
            "evaluation_world_size": int(world_size),
            "evaluation_gpu_ids": gpu_ids,
            "distributed_padding_images": 0,
            "conditions": [
                "base", "static_10k", "advfd_10k", "velocity_5k", "velocity_10k"
            ],
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)
PY

cd "${OFFICIAL_ROOT}"
torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${IMAGE_DIRS[@]}" \
  --output_csv "${FDR3_CSVS[@]}" \
  --models \
    convnext \
    vit_large_patch14_dinov2.lvd142m \
    vit_large_patch14_clip_224.openai \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${RESULT_ROOT}/heldout_fdr3.log"

torchrun --standalone --nproc-per-node="${NPROC}" eval_all_fds.py \
  --image_folder "${IMAGE_DIRS[@]}" \
  --output_csv "${INCEPTION_CSVS[@]}" \
  --models inception \
  --img_size 256 --batch_size "${FD_BATCH}" --num_workers 4 --no_prc \
  2>&1 | tee "${RESULT_ROOT}/paired_inception.log"

cd "${REPO_ROOT}"
SUMMARY_ARGS=()
for condition in "${CONDITIONS[@]}"; do
  SUMMARY_ARGS+=(--condition-csv "${condition}=${RESULT_ROOT}/${condition}_fdr3_raw.csv")
done
python experiments/advfd_cleanroom/summarize_official_fdr3.py \
  "${SUMMARY_ARGS[@]}" \
  --output-csv "${RESULT_ROOT}/fdr3_summary.csv" \
  --output-json "${RESULT_ROOT}/fdr3_summary.json" \
  2>&1 | tee "${RESULT_ROOT}/fdr3_summary.log"

touch "${RESULT_ROOT}/_SUCCESS"
