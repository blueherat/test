#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

OFF_JSON=${OFF_JSON:-$EQVAE_STAGE2_SAMPLES/official_ditdh_xl_ag_n50000_adm_adm_fid.json}
OUT_ROOT=${OUT_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$EQVAE_STAGE2_SAMPLES/logs}
SAVE_FOLDER=${SAVE_FOLDER:-ditdh_s_step20000_n50000_adm}
CFG=${CFG:-$EQVAE_ARTIFACTS_DIR/rae_stage2_training/ditdh_s_dinov2_imagenet256_parquet_2gpu_s20000/sampling_step20000.yaml}
REF=${REF:-$EQVAE_ADM_REF}
INCEPTION=${INCEPTION:-$EQVAE_ADM_INCEPTION}
ADM_PY=${ADM_PY:-$EQVAE_ADM_PY}
MAX_OFFICIAL_FID=${MAX_OFFICIAL_FID:-5.0}
PER_PROC_BATCH_SIZE=${PER_PROC_BATCH_SIZE:-16}

CHAIN_LOG="$LOG_DIR/${SAVE_FOLDER}_chain.log"
SAMPLE_LOG="$LOG_DIR/${SAVE_FOLDER}.log"
EVAL_LOG="$LOG_DIR/${SAVE_FOLDER}_adm_fid.log"
OUT_JSON="$OUT_ROOT/${SAVE_FOLDER}_adm_fid.json"

mkdir -p "$LOG_DIR"
exec >> "$CHAIN_LOG" 2>&1
trap 'echo "CHAIN_EXIT $? $(date)"' EXIT

echo "CHAIN_START $(date)"
echo "WAIT_OFFICIAL_JSON $OFF_JSON"
while [ ! -s "$OFF_JSON" ]; do
  sleep 60
done

FID=$("$ADM_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["fid"])' "$OFF_JSON")
echo "OFFICIAL_ADM_FID $FID $(date)"
"$ADM_PY" -c 'import sys
fid = float(sys.argv[1])
limit = float(sys.argv[2])
if not fid <= limit:
    raise SystemExit(f"Official ADM FID {fid:.6f} is above {limit:.6f}; aborting downstream run")
' "$FID" "$MAX_OFFICIAL_FID"

if [ -s "$OUT_JSON" ]; then
  echo "OUR_ADM_JSON_EXISTS $OUT_JSON"
  exit 0
fi

while [ ! -s "$OUT_ROOT/${SAVE_FOLDER}.npz" ]; do
  cd "$EQVAE_ROOT/external/RAE"
  echo "OUR_SAMPLE_START $(date)"
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 SAVE_FOLDER="$SAVE_FOLDER" PYTHONPATH=src PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node=4 src/sample_ddp.py \
      --config "$CFG" \
      --sample-dir "$OUT_ROOT" \
      --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
      --num-fid-samples 50000 \
      --global-seed 0 \
      --precision fp32 \
      --label-sampling equal >> "$SAMPLE_LOG" 2>&1
  sample_code=$?
  set -e
  echo "OUR_SAMPLE_EXIT $sample_code $(date)"
  if [ "$sample_code" -eq 0 ]; then
    break
  fi
  echo "OUR_SAMPLE_RETRY_AFTER_FAILURE $(date)"
  sleep 300
done
echo "OUR_SAMPLE_DONE $(date)"

cd "$EQVAE_ROOT"
echo "OUR_ADM_START $(date)"
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$ADM_PY" experiments/compute_adm_fid.py \
  --reference "$REF" \
  --samples "$OUT_ROOT/${SAVE_FOLDER}.npz" \
  --batch-size 64 \
  --inception-path "$INCEPTION" \
  --output "$OUT_JSON" > "$EVAL_LOG" 2>&1
echo "OUR_ADM_DONE $OUT_JSON $(date)"
