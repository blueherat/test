#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-ditdh_s_dinov2_adapter_imagenet32768_s20000_4gpu}
CONFIG=${CONFIG:-$EQVAE_ROOT/experiments/configs/rae_stage2_ditdh_s_dinov2_adapter_imagenet32768_parquet.yaml}
DATA_PATH=${DATA_PATH:-$EQVAE_SHARED_DATA_ROOT/imagenet-1k}
RESULTS_DIR=${RESULTS_DIR:-$EQVAE_STAGE2_TRAINING}
SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$SAMPLE_ROOT/logs}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-20000}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
SAMPLE_BATCH=${SAMPLE_BATCH:-16}
SAMPLE_FOLDER=${SAMPLE_FOLDER:-${RUN_NAME}_n50000_adm}
REF=${REF:-$EQVAE_ADM_REF}
INCEPTION=${INCEPTION:-$EQVAE_ADM_INCEPTION}
ADM_PY=${ADM_PY:-$EQVAE_ADM_PY}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
NUM_STEPS=${NUM_STEPS:-50}
WAIT_FOR_FREE_GPUS=${WAIT_FOR_FREE_GPUS:-0}
GPU_MEM_MAX_MB=${GPU_MEM_MAX_MB:-1200}
GPU_UTIL_MAX=${GPU_UTIL_MAX:-10}
GPU_WAIT_SECONDS=${GPU_WAIT_SECONDS:-300}

RUN_DIR="$RESULTS_DIR/$RUN_NAME"
CHAIN_LOG="$LOG_DIR/${RUN_NAME}_chain.log"
TRAIN_LOG="$LOG_DIR/${RUN_NAME}_train.log"
SAMPLE_LOG="$LOG_DIR/${SAMPLE_FOLDER}.log"
ADM_LOG="$LOG_DIR/${SAMPLE_FOLDER}_adm_fid.log"
SAMPLE_CFG="$RUN_DIR/sampling_step${MAX_TRAIN_STEPS}.yaml"
CHECKPOINT="$RUN_DIR/checkpoints/step-$(printf '%07d' "$MAX_TRAIN_STEPS").pt"
SAMPLE_NPZ="$SAMPLE_ROOT/${SAMPLE_FOLDER}.npz"
OUT_JSON="$SAMPLE_ROOT/${SAMPLE_FOLDER}_adm_fid.json"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$SAMPLE_ROOT"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "ADAPTER_CHAIN_EXIT $? $(date)"' EXIT

echo "ADAPTER_CHAIN_START $(date)"
echo "RUN_NAME $RUN_NAME"
echo "CONFIG $CONFIG"
echo "DATA_PATH $DATA_PATH"
echo "RESULTS_DIR $RESULTS_DIR"
echo "TRAIN_GPUS $TRAIN_GPUS"
echo "NPROC $NPROC"
echo "MAX_TRAIN_STEPS $MAX_TRAIN_STEPS"
echo "WAIT_FOR_FREE_GPUS $WAIT_FOR_FREE_GPUS"

wait_for_free_gpus() {
  if [ "$WAIT_FOR_FREE_GPUS" != "1" ]; then
    return
  fi
  while true; do
    if python - "$TRAIN_GPUS" "$GPU_MEM_MAX_MB" "$GPU_UTIL_MAX" <<'PY'
import subprocess
import sys

selected = {int(x) for x in sys.argv[1].split(",") if x.strip()}
mem_limit = int(sys.argv[2])
util_limit = int(sys.argv[3])
out = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
busy = []
for line in out.strip().splitlines():
    idx_s, mem_s, util_s = [part.strip() for part in line.split(",")]
    idx, mem, util = int(idx_s), int(mem_s), int(util_s)
    if idx in selected and (mem > mem_limit or util > util_limit):
        busy.append((idx, mem, util))
if busy:
    print("busy_gpus", busy)
    raise SystemExit(1)
print("selected_gpus_free", sorted(selected))
PY
    then
      echo "GPU_WAIT_DONE $(date)"
      return
    fi
    echo "GPU_WAIT_SLEEP ${GPU_WAIT_SECONDS}s $(date)"
    sleep "$GPU_WAIT_SECONDS"
  done
}

if [ ! -s "$CHECKPOINT" ]; then
  wait_for_free_gpus
  echo "ADAPTER_TRAIN_START $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  EXPERIMENT_NAME="$RUN_NAME" CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$NPROC" src/train.py \
      --config "$CONFIG" \
      --data-path "$DATA_PATH" \
      --results-dir "$RESULTS_DIR" \
      --image-size 256 \
      --precision fp32 \
      --max-train-steps "$MAX_TRAIN_STEPS" > "$TRAIN_LOG" 2>&1
  echo "ADAPTER_TRAIN_DONE $(date)"
else
  echo "ADAPTER_CHECKPOINT_EXISTS $CHECKPOINT"
fi

if [ ! -s "$CHECKPOINT" ]; then
  echo "ADAPTER_CHECKPOINT_MISSING $CHECKPOINT"
  exit 2
fi

if [ ! -s "$SAMPLE_CFG" ]; then
  echo "ADAPTER_SAMPLING_CONFIG_START $(date)"
  cd "$EQVAE_ROOT"
  python experiments/make_rae_sampling_config.py \
    --base-config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$SAMPLE_CFG" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --num-steps "$NUM_STEPS"
  echo "ADAPTER_SAMPLING_CONFIG_DONE $SAMPLE_CFG"
else
  echo "ADAPTER_SAMPLING_CONFIG_EXISTS $SAMPLE_CFG"
fi

while [ ! -s "$SAMPLE_NPZ" ]; do
  wait_for_free_gpus
  echo "ADAPTER_SAMPLE_START batch=$SAMPLE_BATCH $(date)"
  cd "$EQVAE_ROOT/external/RAE"
  set +e
  CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" SAVE_FOLDER="$SAMPLE_FOLDER" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node="$NPROC" src/sample_ddp.py \
      --config "$SAMPLE_CFG" \
      --sample-dir "$SAMPLE_ROOT" \
      --per-proc-batch-size "$SAMPLE_BATCH" \
      --num-fid-samples 50000 \
      --global-seed 0 \
      --precision fp32 \
      --label-sampling equal >> "$SAMPLE_LOG" 2>&1
  sample_code=$?
  set -e
  echo "ADAPTER_SAMPLE_EXIT $sample_code $(date)"
  if [ "$sample_code" -eq 0 ]; then
    break
  fi
  echo "ADAPTER_SAMPLE_RETRY_AFTER_FAILURE $(date)"
  sleep 300
done
echo "ADAPTER_SAMPLE_DONE $(date)"

if [ ! -s "$OUT_JSON" ]; then
  echo "ADAPTER_ADM_START $(date)"
  cd "$EQVAE_ROOT"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$ADM_PY" experiments/compute_adm_fid.py \
    --reference "$REF" \
    --samples "$SAMPLE_NPZ" \
    --batch-size 64 \
    --inception-path "$INCEPTION" \
    --output "$OUT_JSON" > "$ADM_LOG" 2>&1
  echo "ADAPTER_ADM_DONE $OUT_JSON $(date)"
else
  echo "ADAPTER_ADM_JSON_EXISTS $OUT_JSON"
fi
