#!/usr/bin/env bash
set -euo pipefail

# Strictly continue the matched RAEv2 Flow50 and prediction-detach50 branches to
# cumulative branch update 100.  Usage:
#
#   bash experiments/run_raev2_matched_100step_training.sh both
#   bash experiments/run_raev2_matched_100step_training.sh detach
#   bash experiments/run_raev2_matched_100step_training.sh flow
#
# --max-updates 100 is cumulative.  When resuming branch_update=50, only
# another 50 optimizer updates are executed.

MODE="${1:-both}"
case "$MODE" in
  check|both|detach|flow) ;;
  *)
    echo "usage: $0 [check|both|detach|flow]" >&2
    exit 2
    ;;
esac

REPO="${REPO:-/home/zhoushunyu/eqvae}"
DATA_ROOT="${DATA_ROOT:-/data/users/zhoushunyu/eqvae}"
PYTHON="${PYTHON:-$DATA_ROOT/envs/raev2/bin/python}"
CONFIG="${CONFIG:-$REPO/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml}"
TRAINER="${TRAINER:-$REPO/experiments/train_raev2_strict_lpl.py}"

SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$DATA_ROOT/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt}"
DINO_CKPT_DIR="${DINO_CKPT_DIR:-$DATA_ROOT/models/RAEv2/encoders/dinov3}"
DINO_REPO_DIR="${DINO_REPO_DIR:-$DATA_ROOT/models/RAEv2/dinov3_repo}"
INDEX_MAP="${INDEX_MAP:-$DATA_ROOT/datasets/raev2_imagenet_train_lexicographic_indices.npy}"

# The newly trained matched Flow50 branch.
FLOW50="${FLOW50:-$DATA_ROOT/experiments/raev2_flow50_matched_detach_protocol/flow50_matched_detach_protocol/checkpoints/branch-0000050-global-0100130.pt}"

# The retained prediction-detach branch at cumulative update 50.
DETACH50="${DETACH50:-$DATA_ROOT/retained/cleanup_20260731/checkpoints/raev2_dinov3_l/detach_branch50.pt}"

OUT_ROOT="${OUT_ROOT:-$DATA_ROOT/experiments/raev2_flow_detach100_matched}"
FLOW_EXPERIMENT="${FLOW_EXPERIMENT:-flow100_matched_detach_protocol}"
DETACH_EXPERIMENT="${DETACH_EXPERIMENT:-detach100_prediction_detach}"

LPL_WEIGHT="${LPL_WEIGHT:-2.9384045033942286e-5}"
GPUS="${GPUS:-0,1,2,3}"
MIN_FREE_GIB="${MIN_FREE_GIB:-18}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WAIT_FOR_ACTIVE_JOBS="${WAIT_FOR_ACTIVE_JOBS:-1}"

mkdir -p "$OUT_ROOT/logs"
cd "$REPO"

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
require_file "$CONFIG" "training config"
require_file "$TRAINER" "training program"
require_file "$SOURCE_CHECKPOINT" "official source checkpoint"
require_file "$FLOW50" "matched Flow50 checkpoint"
require_file "$DETACH50" "prediction-detach50 checkpoint"
require_file "$INDEX_MAP" "ImageNet index map"
require_dir "$DINO_CKPT_DIR" "DINOv3 checkpoint directory"
require_dir "$DINO_REPO_DIR" "DINOv3 repository"
require_dir "/data/shared/imagenet-1k/data" "ImageNet parquet data"
require_dir "/data/shared/imagenet-1k/random_access_v1" "packed ImageNet data"

# Check that both update-50 checkpoints are full resumable checkpoints and
# represent the intended matched protocol.
"$PYTHON" - "$FLOW50" "$DETACH50" "$SOURCE_CHECKPOINT" <<'PY'
from pathlib import Path
import hashlib
import sys
import torch

flow_path, detach_path, source_path = map(Path, sys.argv[1:4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def tree_equal(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right)
        )
    try:
        import numpy as np
        if isinstance(left, np.ndarray):
            return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    except ImportError:
        pass
    return left == right


def inspect(path: Path, expected_objective: str, expected_variant: str | None):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    for key in ("model", "ema", "optimizer", "scheduler", "raev2_lpl"):
        if key not in checkpoint:
            raise RuntimeError(f"{path}: missing checkpoint key {key!r}")
    if checkpoint["optimizer"] is None:
        raise RuntimeError(f"{path}: optimizer state is missing")
    if checkpoint["scheduler"] is None:
        raise RuntimeError(f"{path}: scheduler state is missing")

    metadata = checkpoint["raev2_lpl"]
    if int(checkpoint["step"]) != 100130:
        raise RuntimeError(
            f"{path}: expected global checkpoint step=100130, got "
            f"{checkpoint['step']}"
        )
    if int(metadata["branch_update"]) != 50:
        raise RuntimeError(
            f"{path}: expected branch_update=50, got "
            f"{metadata['branch_update']}"
        )
    if int(metadata["source_step"]) != 100080:
        raise RuntimeError(
            f"{path}: expected source_step=100080, got "
            f"{metadata['source_step']}"
        )
    if metadata["objective"] != expected_objective:
        raise RuntimeError(
            f"{path}: expected objective={expected_objective}, got "
            f"{metadata['objective']}"
        )
    if expected_variant is not None:
        if metadata.get("lpl_target") != "full_base":
            raise RuntimeError(
                f"{path}: expected lpl_target=full_base, got "
                f"{metadata.get('lpl_target')}"
            )
        if metadata.get("lpl_variant") != expected_variant:
            raise RuntimeError(
                f"{path}: expected lpl_variant={expected_variant}, got "
                f"{metadata.get('lpl_variant')}"
            )
        if metadata.get("lpl_gradient_mode") != "direct":
            raise RuntimeError(
                f"{path}: expected direct LPL gradient, got "
                f"{metadata.get('lpl_gradient_mode')}"
            )

    result = {
        "path": str(path),
        "step": int(checkpoint["step"]),
        "branch_update": int(metadata["branch_update"]),
        "objective": metadata["objective"],
        "variant": metadata.get("lpl_variant"),
        "data_indices_sha256": metadata.get("data_indices_sha256"),
        "source_sha256": metadata["source_sha256"],
        "source_epoch": int(metadata["source_epoch"]),
        "source_steps_per_epoch": int(metadata["source_steps_per_epoch"]),
        "config_sha256": metadata["config_sha256"],
        "rank_rng_states": metadata["rank_rng_states"],
        "scheduler": checkpoint["scheduler"],
    }
    return result

flow = inspect(flow_path, "flow", None)
detach = inspect(detach_path, "lpl", "prediction_detach")
source_hash = sha256(source_path)

for name, item in (("flow50", flow), ("detach50", detach)):
    if item["source_sha256"] != source_hash:
        raise RuntimeError(
            f"{name}: source hash disagrees with official checkpoint"
        )

common_metadata_fields = (
    "source_epoch",
    "source_steps_per_epoch",
    "config_sha256",
    "data_indices_sha256",
)
for field in common_metadata_fields:
    if flow[field] != detach[field]:
        raise RuntimeError(
            f"Flow50 and Detach50 differ in {field}: "
            f"flow={flow[field]!r}, detach={detach[field]!r}"
        )

if not tree_equal(flow["scheduler"], detach["scheduler"]):
    raise RuntimeError(
        "Flow50 and Detach50 scheduler states differ. "
        "They are not at an identical learning-rate/scheduler position."
    )

if not tree_equal(flow["rank_rng_states"], detach["rank_rng_states"]):
    raise RuntimeError(
        "Flow50 and Detach50 saved rank RNG states differ. "
        "Continuing them would not guarantee identical future noise, time, "
        "CFG masks, and stochastic model operations."
    )

def printable(item):
    return {
        key: value
        for key, value in item.items()
        if key not in ("rank_rng_states", "scheduler")
    }

print("Checkpoint audit passed:")
print(printable(flow))
print(printable(detach))
print("The update-50 data stream, scheduler position, and rank RNG states match exactly.")
PY

if [[ "$MODE" == "check" ]]; then
  echo "Preflight-only check completed; no GPU training was launched."
  exit 0
fi

selected_gpu_count="$(
  awk -F',' '{print NF}' <<<"$GPUS"
)"
if [[ "$selected_gpu_count" -ne 4 ]]; then
  echo "this protocol requires exactly four GPUs; got GPUS=$GPUS" >&2
  exit 1
fi

active_jobs() {
  pgrep -u "$USER" -af \
    'sample_raev2_threeway\.py|run_raev2_ig_scale_sweep\.py|evaluate_raev2_samples\.py|audit_raev2_detach_gradient_geometry\.py' \
    || true
}

gpu_memory_ready() {
  local threshold_mib
  threshold_mib="$("$PYTHON" - "$MIN_FREE_GIB" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024))
PY
)"
  CUDA_VISIBLE_DEVICES="$GPUS" nvidia-smi \
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

wait_for_resources() {
  if [[ "$WAIT_FOR_ACTIVE_JOBS" == "0" ]]; then
    return
  fi

  while true; do
    jobs="$(active_jobs)"
    memory_ok=0
    if gpu_memory_ready; then
      memory_ok=1
    fi

    if [[ -z "$jobs" && "$memory_ok" -eq 1 ]]; then
      echo "[$(date --iso-8601=seconds)] GPUs ready; starting training."
      return
    fi

    echo "[$(date --iso-8601=seconds)] waiting for resources..."
    if [[ -n "$jobs" ]]; then
      echo "active matching jobs:"
      echo "$jobs"
    fi
    echo "free GPU memory:"
    CUDA_VISIBLE_DEVICES="$GPUS" nvidia-smi \
      --query-gpu=index,memory.free,memory.total \
      --format=csv,noheader
    sleep "$POLL_SECONDS"
  done
}

latest_checkpoint_or_initial() {
  local checkpoint_dir="$1"
  local initial="$2"
  local latest=""

  if [[ -d "$checkpoint_dir" ]]; then
    latest="$(
      find "$checkpoint_dir" -maxdepth 1 -type f \
        -name 'branch-*-global-*.pt' -print |
      sort |
      tail -n 1
    )"
  fi

  if [[ -n "$latest" ]]; then
    printf '%s\n' "$latest"
  else
    printf '%s\n' "$initial"
  fi
}

run_flow100() {
  local experiment_dir="$OUT_ROOT/$FLOW_EXPERIMENT"
  local checkpoint_dir="$experiment_dir/checkpoints"
  local final="$checkpoint_dir/branch-0000100-global-0100180.pt"

  if [[ -f "$final" ]]; then
    echo "[$(date --iso-8601=seconds)] Flow100 already complete: $final"
    return
  fi

  local resume
  resume="$(latest_checkpoint_or_initial "$checkpoint_dir" "$FLOW50")"
  require_file "$resume" "Flow resume checkpoint"

  echo "[$(date --iso-8601=seconds)] Flow: resume $(basename "$resume") -> update 100"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    experiments/train_raev2_strict_lpl.py \
    --config "$CONFIG" \
    --data-path /data/shared/imagenet-1k/data \
    --packed-data-path /data/shared/imagenet-1k/random_access_v1 \
    --index-map "$INDEX_MAP" \
    --results-dir "$OUT_ROOT" \
    --experiment-name "$FLOW_EXPERIMENT" \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --resume "$resume" \
    --objective flow \
    --max-updates 100 \
    --save-every 10 \
    --precision bf16 \
    --ema-device cpu \
    --global-seed 42 \
    --num-workers 4 \
    --min-free-gib 0.5 \
    --dino-ckpt-dir "$DINO_CKPT_DIR" \
    --dino-repo-dir "$DINO_REPO_DIR" \
    2>&1 | tee -a "$OUT_ROOT/logs/flow100.log"

  require_file "$final" "final matched Flow100 checkpoint"
  echo "[$(date --iso-8601=seconds)] completed Flow100: $final"
}

run_detach100() {
  local experiment_dir="$OUT_ROOT/$DETACH_EXPERIMENT"
  local checkpoint_dir="$experiment_dir/checkpoints"
  local final="$checkpoint_dir/branch-0000100-global-0100180.pt"

  if [[ -f "$final" ]]; then
    echo "[$(date --iso-8601=seconds)] Detach100 already complete: $final"
    return
  fi

  local resume
  resume="$(latest_checkpoint_or_initial "$checkpoint_dir" "$DETACH50")"
  require_file "$resume" "Detach resume checkpoint"

  echo "[$(date --iso-8601=seconds)] Detach: resume $(basename "$resume") -> update 100"
  CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    experiments/train_raev2_strict_lpl.py \
    --config "$CONFIG" \
    --data-path /data/shared/imagenet-1k/data \
    --packed-data-path /data/shared/imagenet-1k/random_access_v1 \
    --index-map "$INDEX_MAP" \
    --results-dir "$OUT_ROOT" \
    --experiment-name "$DETACH_EXPERIMENT" \
    --source-checkpoint "$SOURCE_CHECKPOINT" \
    --resume "$resume" \
    --objective lpl \
    --lpl-target full_base \
    --lpl-variant prediction_detach \
    --lpl-gradient-mode direct \
    --lpl-guidance-scale 1.78 \
    --lpl-multiscale-scales 1.0,1.39,1.78 \
    --lpl-weight "$LPL_WEIGHT" \
    --lpl-noise-threshold 3.0 \
    --lpl-max-samples-per-rank 1 \
    --max-updates 100 \
    --save-every 10 \
    --precision bf16 \
    --ema-device cpu \
    --global-seed 42 \
    --num-workers 4 \
    --min-free-gib 0.5 \
    --dino-ckpt-dir "$DINO_CKPT_DIR" \
    --dino-repo-dir "$DINO_REPO_DIR" \
    2>&1 | tee -a "$OUT_ROOT/logs/detach100.log"

  require_file "$final" "final prediction-detach100 checkpoint"
  echo "[$(date --iso-8601=seconds)] completed Detach100: $final"
}

verify_final_pair() {
  local detach_final="$OUT_ROOT/$DETACH_EXPERIMENT/checkpoints/branch-0000100-global-0100180.pt"
  local flow_final="$OUT_ROOT/$FLOW_EXPERIMENT/checkpoints/branch-0000100-global-0100180.pt"
  local detach_first="$OUT_ROOT/$DETACH_EXPERIMENT/first_batch_audit.json"
  local flow_first="$OUT_ROOT/$FLOW_EXPERIMENT/first_batch_audit.json"

  require_file "$detach_final" "final Detach100 checkpoint"
  require_file "$flow_final" "final Flow100 checkpoint"
  require_file "$detach_first" "Detach100 first-batch audit"
  require_file "$flow_first" "Flow100 first-batch audit"

  "$PYTHON" - "$flow_final" "$detach_final" "$flow_first" "$detach_first" <<'PY'
from pathlib import Path
import json
import sys
import torch

flow_path, detach_path, flow_first_path, detach_first_path = map(Path, sys.argv[1:5])

def tree_equal(left, right):
    if type(left) is not type(right):
        return False
    if torch.is_tensor(left):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            tree_equal(a, b) for a, b in zip(left, right)
        )
    try:
        import numpy as np
        if isinstance(left, np.ndarray):
            return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    except ImportError:
        pass
    return left == right

flow = torch.load(flow_path, map_location="cpu", weights_only=False, mmap=True)
detach = torch.load(detach_path, map_location="cpu", weights_only=False, mmap=True)

for name, checkpoint in (("flow100", flow), ("detach100", detach)):
    metadata = checkpoint["raev2_lpl"]
    if int(checkpoint["step"]) != 100180:
        raise RuntimeError(f"{name}: expected global step 100180, got {checkpoint['step']}")
    if int(metadata["branch_update"]) != 100:
        raise RuntimeError(
            f"{name}: expected branch_update 100, got {metadata['branch_update']}"
        )

flow_meta = flow["raev2_lpl"]
detach_meta = detach["raev2_lpl"]

for field in (
    "source_sha256",
    "source_step",
    "source_epoch",
    "source_steps_per_epoch",
    "config_sha256",
    "data_indices_sha256",
):
    if flow_meta[field] != detach_meta[field]:
        raise RuntimeError(
            f"final pair differs in {field}: "
            f"flow={flow_meta[field]!r}, detach={detach_meta[field]!r}"
        )

if not tree_equal(flow["scheduler"], detach["scheduler"]):
    raise RuntimeError("final scheduler states differ")
if not tree_equal(flow_meta["rank_rng_states"], detach_meta["rank_rng_states"]):
    raise RuntimeError("final rank RNG states differ")

flow_first = json.loads(flow_first_path.read_text(encoding="utf-8"))
detach_first = json.loads(detach_first_path.read_text(encoding="utf-8"))
if flow_first != detach_first:
    raise RuntimeError(
        "update-51 first-batch audits differ: images/labels/latents/noise/time/CFG "
        "were not exactly paired"
    )

print("Final paired verification passed:")
print("  update-51 first-batch audit: exact match")
print("  updates 51-100 data-index hash: exact match")
print("  final scheduler state: exact match")
print("  final rank RNG state: exact match")
PY
}

wait_for_resources

# Run the requested branch.  For "both", Detach is run first because it is the
# experimental branch of immediate interest, followed by its matched Flow control.
case "$MODE" in
  detach)
    run_detach100
    ;;
  flow)
    run_flow100
    ;;
  both)
    run_detach100
    run_flow100
    verify_final_pair
    ;;
esac

echo
echo "Final expected checkpoints:"
echo "  Detach100: $OUT_ROOT/$DETACH_EXPERIMENT/checkpoints/branch-0000100-global-0100180.pt"
echo "  Flow100:   $OUT_ROOT/$FLOW_EXPERIMENT/checkpoints/branch-0000100-global-0100180.pt"