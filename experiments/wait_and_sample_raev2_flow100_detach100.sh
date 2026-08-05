#!/usr/bin/env bash
set -euo pipefail

# Launch this in a separate tmux while matched Flow100/Detach100 are training.
# It waits for both final checkpoints to be complete and loadable, waits for
# the GPUs to be released, then runs the IG-scale sampling/evaluation sweep.
#
# Defaults:
#   sample count: 5000
#   scales: 1.0,1.2,1.4,1.47,1.54,1.6,1.78,1.82,1.86
#
# Override example for a cheap pilot:
#   SAMPLE_COUNT=1000 bash experiments/wait_and_sample_raev2_flow100_detach100.sh

REPO="${REPO:-/home/zhoushunyu/eqvae}"
DATA_ROOT="${DATA_ROOT:-/data/users/zhoushunyu/eqvae}"
PYTHON="${PYTHON:-$DATA_ROOT/envs/raev2/bin/python}"
CONFIG="${CONFIG:-$REPO/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml}"
SWEEP_PROGRAM="${SWEEP_PROGRAM:-$REPO/experiments/run_raev2_ig_scale_sweep.py}"

TRAIN_ROOT="${TRAIN_ROOT:-$DATA_ROOT/experiments/raev2_flow_detach100_matched}"
FLOW100="${FLOW100:-$TRAIN_ROOT/flow100_matched_detach_protocol/checkpoints/branch-0000100-global-0100180.pt}"
DETACH100="${DETACH100:-$TRAIN_ROOT/detach100_prediction_detach/checkpoints/branch-0000100-global-0100180.pt}"

SAMPLE_COUNT="${SAMPLE_COUNT:-5000}"
PER_RANK_BATCH="${PER_RANK_BATCH:-8}"
SAMPLING_SEED="${SAMPLING_SEED:-20260805}"
METRIC_SEED="${METRIC_SEED:-20260805}"
SCALES="${SCALES:-1.0,1.2,1.4,1.47,1.54,1.6,1.78,1.82,1.86}"
DEVICES="${DEVICES:-0,1,2,3}"
PRECISION="${PRECISION:-bf16}"
STATE_KEY="${STATE_KEY:-model}"
DINO_REPO="${DINO_REPO:-$DATA_ROOT/models/RAEv2/dinov3_repo}"

POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"

OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments/raev2_ig_scale_sweep/flow100_vs_detach100_n${SAMPLE_COUNT}_seed${SAMPLING_SEED}}"
LOG="$OUT_ROOT/run.log"

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "$path" ]]; then
    echo "missing $description: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "$path" ]]; then
    echo "missing $description: $path" >&2
    exit 1
  fi
}

require_file "$PYTHON" "RAEv2 Python"
require_file "$CONFIG" "RAEv2 config"
require_file "$SWEEP_PROGRAM" "IG scale sweep program"
require_dir "$DINO_REPO" "DINOv3 repository"

checkpoint_ready() {
  "$PYTHON" - "$FLOW100" "$DETACH100" <<'PY' >/dev/null 2>&1
from pathlib import Path
import sys
import torch

flow_path, detach_path = map(Path, sys.argv[1:3])

for path in (flow_path, detach_path):
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(1)

flow = torch.load(flow_path, map_location="cpu", weights_only=False, mmap=True)
detach = torch.load(detach_path, map_location="cpu", weights_only=False, mmap=True)

def check_common(name, checkpoint):
    metadata = checkpoint.get("raev2_lpl")
    if metadata is None:
        raise RuntimeError(f"{name}: missing raev2_lpl metadata")
    if int(checkpoint.get("step", -1)) != 100180:
        raise RuntimeError(
            f"{name}: expected global step 100180, got {checkpoint.get('step')}"
        )
    if int(metadata.get("branch_update", -1)) != 100:
        raise RuntimeError(
            f"{name}: expected branch_update 100, got "
            f"{metadata.get('branch_update')}"
        )
    if checkpoint.get("model") is None or checkpoint.get("ema") is None:
        raise RuntimeError(f"{name}: missing model or EMA state")
    return metadata

flow_meta = check_common("flow100", flow)
detach_meta = check_common("detach100", detach)

if flow_meta.get("objective") != "flow":
    raise RuntimeError(
        f"flow100: expected objective=flow, got {flow_meta.get('objective')}"
    )

if detach_meta.get("objective") != "lpl":
    raise RuntimeError(
        f"detach100: expected objective=lpl, got {detach_meta.get('objective')}"
    )
if detach_meta.get("lpl_target") != "full_base":
    raise RuntimeError(
        f"detach100: expected target=full_base, got "
        f"{detach_meta.get('lpl_target')}"
    )
if detach_meta.get("lpl_variant") != "prediction_detach":
    raise RuntimeError(
        f"detach100: expected prediction_detach, got "
        f"{detach_meta.get('lpl_variant')}"
    )
if detach_meta.get("lpl_gradient_mode") != "direct":
    raise RuntimeError(
        f"detach100: expected direct gradient, got "
        f"{detach_meta.get('lpl_gradient_mode')}"
    )

for field in (
    "source_sha256",
    "source_step",
    "source_epoch",
    "source_steps_per_epoch",
    "config_sha256",
    "data_indices_sha256",
):
    if flow_meta.get(field) != detach_meta.get(field):
        raise RuntimeError(
            f"final pair differs in {field}: "
            f"flow={flow_meta.get(field)!r}, "
            f"detach={detach_meta.get(field)!r}"
        )
PY
}

selected_gpu_count="$(
  awk -F',' '{print NF}' <<<"$DEVICES"
)"
if [[ "$selected_gpu_count" -ne 4 ]]; then
  echo "this protocol expects exactly four GPUs; got DEVICES=$DEVICES" >&2
  exit 1
fi

gpu_memory_ready() {
  local threshold_mib
  threshold_mib="$("$PYTHON" - "$MIN_FREE_GIB" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024))
PY
)"
  CUDA_VISIBLE_DEVICES="$DEVICES" nvidia-smi \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits |
  awk -v threshold="$threshold_mib" '
    BEGIN { ok=1; count=0 }
    {
      gsub(/^[ \t]+|[ \t]+$/, "", $0)
      count += 1
      if (($0 + 0) < threshold) ok=0
    }
    END { exit !(ok && count == 4) }
  '
}

echo "Waiting for matched Flow100 and Detach100..."
echo "  Flow100:   $FLOW100"
echo "  Detach100: $DETACH100"
echo "  Scales:    $SCALES"
echo "  Samples:   $SAMPLE_COUNT per branch per scale"

while true; do
  if checkpoint_ready; then
    if gpu_memory_ready; then
      echo "[$(date --iso-8601=seconds)] checkpoints verified and GPUs ready."
      break
    fi
    echo "[$(date --iso-8601=seconds)] checkpoints are ready; waiting for GPU memory..."
  else
    echo "[$(date --iso-8601=seconds)] training not complete yet; waiting..."
  fi

  CUDA_VISIBLE_DEVICES="$DEVICES" nvidia-smi \
    --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader || true
  sleep "$POLL_SECONDS"
done

mkdir -p "$OUT_ROOT"
cd "$REPO"

{
  echo "[$(date --iso-8601=seconds)] starting Flow100 vs Detach100 IG sweep"
  echo "output: $OUT_ROOT"
  echo "scales: $SCALES"
  echo "sample_count: $SAMPLE_COUNT"
  echo "sampling_seed: $SAMPLING_SEED"
  echo "metric_seed: $METRIC_SEED"
} | tee -a "$LOG"

"$PYTHON" "$SWEEP_PROGRAM" \
  --config "$CONFIG" \
  --branch "flow100=$FLOW100" \
  --branch "detach100=$DETACH100" \
  --baseline-branch flow100 \
  --output-root "$OUT_ROOT" \
  --scales "$SCALES" \
  --sample-count "$SAMPLE_COUNT" \
  --per-rank-batch "$PER_RANK_BATCH" \
  --sampling-seed "$SAMPLING_SEED" \
  --metric-seed "$METRIC_SEED" \
  --precision "$PRECISION" \
  --state-key "$STATE_KEY" \
  --devices "$DEVICES" \
  --dino-repo-dir "$DINO_REPO" \
  2>&1 | tee -a "$LOG"

echo "[$(date --iso-8601=seconds)] sweep completed: $OUT_ROOT" | tee -a "$LOG"