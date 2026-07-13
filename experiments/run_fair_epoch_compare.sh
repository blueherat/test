#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

EPOCH_INDEX=${EPOCH_INDEX:-1}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-1251}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-$((EPOCH_INDEX * STEPS_PER_EPOCH))}
NUM_FID_SAMPLES=${NUM_FID_SAMPLES:-10000}
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
NPROC=${NPROC:-4}
SAMPLE_BATCH=${SAMPLE_BATCH:-8}
WAIT_FOR_FREE_GPUS=${WAIT_FOR_FREE_GPUS:-0}

ORIG_RUN=${ORIG_RUN:-fair_ditdh_s_dinov2_original_gbs1024_epochwise_4gpu}
ADAPTER_RUN=${ADAPTER_RUN:-fair_ditdh_s_dinov2_adapter_gbs1024_epochwise_4gpu}
ORIG_CONFIG=${ORIG_CONFIG:-$EQVAE_ROOT/experiments/configs/fair_ditdh_s_dinov2_original_gbs1024_ep80.yaml}
ADAPTER_CONFIG=${ADAPTER_CONFIG:-$EQVAE_ROOT/experiments/configs/fair_ditdh_s_dinov2_adapter_gbs1024_ep80.yaml}

suffix="step$(printf '%07d' "$MAX_TRAIN_STEPS")_n${NUM_FID_SAMPLES}_adm"

echo "EPOCH_INDEX $EPOCH_INDEX"
echo "MAX_TRAIN_STEPS $MAX_TRAIN_STEPS"
echo "NUM_FID_SAMPLES $NUM_FID_SAMPLES"

RUN_NAME="$ORIG_RUN" \
CONFIG="$ORIG_CONFIG" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
TRAIN_GPUS="$TRAIN_GPUS" \
NPROC="$NPROC" \
SAMPLE_BATCH="$SAMPLE_BATCH" \
NUM_FID_SAMPLES="$NUM_FID_SAMPLES" \
SAMPLE_FOLDER="${ORIG_RUN}_${suffix}" \
WAIT_FOR_FREE_GPUS="$WAIT_FOR_FREE_GPUS" \
bash "$EQVAE_ROOT/experiments/run_fair_ditdh_s_gfid.sh"

RUN_NAME="$ADAPTER_RUN" \
CONFIG="$ADAPTER_CONFIG" \
MAX_TRAIN_STEPS="$MAX_TRAIN_STEPS" \
TRAIN_GPUS="$TRAIN_GPUS" \
NPROC="$NPROC" \
SAMPLE_BATCH="$SAMPLE_BATCH" \
NUM_FID_SAMPLES="$NUM_FID_SAMPLES" \
SAMPLE_FOLDER="${ADAPTER_RUN}_${suffix}" \
WAIT_FOR_FREE_GPUS="$WAIT_FOR_FREE_GPUS" \
bash "$EQVAE_ROOT/experiments/run_fair_ditdh_s_gfid.sh"

python - "$MAX_TRAIN_STEPS" "$NUM_FID_SAMPLES" "$ORIG_RUN" "$ADAPTER_RUN" "$EQVAE_STAGE2_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

step = int(sys.argv[1])
num = int(sys.argv[2])
orig = sys.argv[3]
adapter = sys.argv[4]
root = Path(sys.argv[5])
suffix = f"step{step:07d}_n{num}_adm_adm_fid.json"
rows = []
for name, label in [(orig, "original"), (adapter, "adapter")]:
    path = root / f"{name}_{suffix}"
    if not path.exists():
        rows.append((label, "missing", None, None, None, str(path)))
        continue
    data = json.loads(path.read_text())
    rows.append((label, "ok", data["fid"], data["sfid"], data["inception_score"], str(path)))
print("| latent | status | ADM gFID | sFID | IS | json |")
print("|---|---|---:|---:|---:|---|")
for label, status, fid, sfid, iscore, path in rows:
    if status == "ok":
        print(f"| {label} | {status} | {fid:.4f} | {sfid:.4f} | {iscore:.4f} | `{path}` |")
    else:
        print(f"| {label} | {status} |  |  |  | `{path}` |")
PY
