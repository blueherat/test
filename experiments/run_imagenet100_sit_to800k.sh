#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

RUN_DIR=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/runs/sit-s-2_seed0
FID_ROOT=/home/zhoushunyu/data/eqvae/imagenet_sit_flow/fid5k
PIPELINE_LOG="$RUN_DIR/train_300k_to_800k_pipeline.log"
INITIAL_STEPS=60000,120000,180000,240000,300000

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[$(date --iso-8601=seconds)] validating the 60k-to-300k unguided FID curve" \
  | tee -a "$PIPELINE_LOG"
if ! python experiments/run_imagenet100_sit_fid_curve.py \
  --steps "$INITIAL_STEPS" \
  --run-dir "$RUN_DIR" \
  --output-root "$FID_ROOT" \
  --require-improving-tail \
  2>&1 | tee -a "$PIPELINE_LOG"; then
  echo "[$(date --iso-8601=seconds)] FID tail is not improving; stop before continuation." \
    | tee -a "$PIPELINE_LOG"
  exit 4
fi

for TARGET_STEP in 400000 500000 600000 700000 800000; do
  echo "[$(date --iso-8601=seconds)] continuing SiT-S/2 to step $TARGET_STEP" \
    | tee -a "$PIPELINE_LOG"
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  SIT_MODEL=SiT-S/2 \
  GLOBAL_BATCH_SIZE=256 \
  MAX_STEPS="$TARGET_STEP" \
  SAVE_EVERY=100000 \
  SEED=0 \
  OUTPUT_DIR="$RUN_DIR" \
  bash experiments/run_imagenet100_sit_4gpu.sh \
    2>&1 | tee -a "$RUN_DIR/train_to_${TARGET_STEP}.log"

  python experiments/run_imagenet100_sit_fid_curve.py \
    --steps "$TARGET_STEP" \
    --run-dir "$RUN_DIR" \
    --output-root "$FID_ROOT" \
    2>&1 | tee -a "$PIPELINE_LOG"
done

python experiments/run_imagenet100_sit_fid_curve.py \
  --steps 60000,120000,180000,240000,300000,400000,500000,600000,700000,800000 \
  --run-dir "$RUN_DIR" \
  --output-root "$FID_ROOT" \
  2>&1 | tee -a "$PIPELINE_LOG"

touch "$RUN_DIR/COMPLETE_800K_WITH_FID"
echo "[$(date --iso-8601=seconds)] complete through 800k" | tee -a "$PIPELINE_LOG"
