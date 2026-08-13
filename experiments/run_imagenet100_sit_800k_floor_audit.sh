#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
GPU="${GPU:-0}"
GROUP="${GROUP:-mechanism}"
WEAK_STEPS="${WEAK_STEPS:-400 500 600 700}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETE_SERIES="${SKIP_COMPLETE_SERIES:-1}"
ROOT="${ROOT:-$BASE/fid5k_step800k_floor_audit_seed0}"
LOG_DIR="$ROOT/logs"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"

mkdir -p "$LOG_DIR"

for path in "$V800" "$X800" "$REFERENCE"; do
  [[ -f "$path" ]] || { echo "Missing required file: $path" >&2; exit 2; }
done

case "$GROUP" in
  mechanism|inference_floor|autoguidance) ;;
  *)
    echo "Unsupported GROUP=$GROUP (expected mechanism, inference_floor, or autoguidance)" >&2
    exit 2
    ;;
esac

case "$DRY_RUN" in
  0|1) ;;
  *) echo "Unsupported DRY_RUN=$DRY_RUN (expected 0 or 1)" >&2; exit 2 ;;
esac

case "$SKIP_COMPLETE_SERIES" in
  0|1) ;;
  *)
    echo "Unsupported SKIP_COMPLETE_SERIES=$SKIP_COMPLETE_SERIES (expected 0 or 1)" >&2
    exit 2
    ;;
esac

COMMON=(
  --anchor-checkpoint "$V800"
  --anchor-field auto
  --reference "$REFERENCE"
  --num-samples 5000
  --per-rank-batch-size 8
  --vae-decode-batch-size 2
  --cuda-allocator-limit-gib 4
  --sampling-cuda-visible-devices "$GPU"
  --fid-batch-size 8
  --fid-gpu-memory-fraction 0.10
  --fid-cuda-visible-devices "$GPU"
  --gpu-memory-ceiling-mib 8192
  --memory-poll-interval 0.25
  --global-seed 0
)

run_case() {
  local name="$1"
  shift
  if [[ "$SKIP_COMPLETE_SERIES" == "1" && -f "$ROOT/$name/field_control_fid5k.json" ]]; then
    echo "[$(date --iso-8601=seconds)] skip complete series $name"
    return
  fi
  local command=(
    python experiments/run_imagenet100_sit_static_pair_fid5k.py
    "${COMMON[@]}"
    --other-checkpoint "$X800"
    --other-field auto
    --output-root "$ROOT/$name"
    "$@"
  )
  echo "[$(date --iso-8601=seconds)] start $name"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${command[@]}"
    printf '\n'
  else
    "${command[@]}" 2>&1 | tee -a "$LOG_DIR/$name.log"
  fi
  echo "[$(date --iso-8601=seconds)] complete $name"
}

if [[ "$GROUP" == "mechanism" ]]; then
  run_case floor_only \
    --control-mode floor_only \
    --scales -0.2 -0.5 -0.75 -1
  run_case floor_residual \
    --control-mode floor_residual \
    --scales -1
  run_case pre_floor_pair \
    --control-mode pre_floor_pair \
    --window-transition-width 0.01 \
    --scales -1
  run_case post_floor_pair \
    --control-mode post_floor_pair \
    --window-transition-width 0.01 \
    --scales -1
  run_case parallel_pair \
    --control-mode parallel_pair \
    --scales -1
  run_case orthogonal_pair \
    --control-mode orthogonal_pair \
    --scales -1
fi

if [[ "$GROUP" == "inference_floor" ]]; then
  for floor in 0.02 0.01 0.005 0.001; do
    tag="${floor//./p}"
    run_case "x_inference_floor_$tag" \
      --control-mode full_pair \
      --other-inference-denominator-floor "$floor" \
      --scales 1
  done
fi

if [[ "$GROUP" == "autoguidance" ]]; then
  for weak_step in $WEAK_STEPS; do
    weak_checkpoint="$BASE/runs/sit-s-2_seed0/checkpoints/step_$(printf '%08d' "${weak_step}000").pt"
    [[ -f "$weak_checkpoint" ]] || {
      echo "Missing weak checkpoint: $weak_checkpoint" >&2
      exit 2
    }
    name="same_target_v${weak_step}"
    if [[ "$SKIP_COMPLETE_SERIES" == "1" && -f "$ROOT/$name/field_control_fid5k.json" ]]; then
      echo "[$(date --iso-8601=seconds)] skip complete series $name"
      continue
    fi
    command=(
      python experiments/run_imagenet100_sit_static_pair_fid5k.py
      "${COMMON[@]}"
      --other-checkpoint "$weak_checkpoint"
      --other-field auto
      --allow-step-mismatch
      --control-mode full_pair
      --scales 1 -0.2 -0.5 -0.75 -1
      --output-root "$ROOT/$name"
    )
    echo "[$(date --iso-8601=seconds)] start $name"
    if [[ "$DRY_RUN" == "1" ]]; then
      printf 'DRY_RUN:'
      printf ' %q' "${command[@]}"
      printf '\n'
    else
      "${command[@]}" 2>&1 | tee -a "$LOG_DIR/$name.log"
    fi
    echo "[$(date --iso-8601=seconds)] complete $name"
  done
fi

if [[ "$DRY_RUN" == "0" ]]; then
  touch "$ROOT/COMPLETE_$GROUP"
fi
echo "[$(date --iso-8601=seconds)] 800K audit group $GROUP complete"
