#!/usr/bin/env bash
set -euo pipefail

# Official RAEv2: paired-error / predicted-clean distribution / free gFID study.
#
# Usage:
#   bash experiments/run_official_error_distribution_study.sh paired
#   bash experiments/run_official_error_distribution_study.sh reference
#   bash experiments/run_official_error_distribution_study.sh predicted
#   bash experiments/run_official_error_distribution_study.sh free
#   bash experiments/run_official_error_distribution_study.sh evaluate
#   bash experiments/run_official_error_distribution_study.sh all
#
# Optional overrides:
#   PAIR_SAMPLES=64 DIST_SAMPLES=5000 FREE_SAMPLES=5000 \
#   SCALES="1.0 1.2 1.4 1.6 1.78 2.0" \
#   bash experiments/run_official_error_distribution_study.sh all

MODE=${1:-}

REPO=/home/zhoushunyu/eqvae
DATA=/data/users/zhoushunyu/eqvae
PY="$DATA/envs/raev2/bin/python"

CONFIG="$REPO/experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
CKPT="$DATA/models/RAEv2/stage2/imagenet/dinov3l-k7/checkpoint.pt"
DINO_CKPT="$DATA/models/RAEv2/encoders/dinov3"
DINO_REPO="$DATA/models/RAEv2/dinov3_repo"
INDEX_MAP="$DATA/datasets/raev2_imagenet_train_lexicographic_indices.npy"

PARQUET=/data/shared/imagenet-1k/data
PACKED=/data/shared/imagenet-1k/random_access_v1
FID_REF=/data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz

SEED=${SEED:-20260805}
PAIR_SAMPLES=${PAIR_SAMPLES:-32}
DIST_SAMPLES=${DIST_SAMPLES:-1000}
FREE_SAMPLES=${FREE_SAMPLES:-5000}
SCALES=${SCALES:-"1.0 1.2 1.4 1.6 1.78 2.0"}

OUT="$DATA/experiments/official_error_distribution_seed${SEED}"
PAIR_OUT="$OUT/paired_error_n${PAIR_SAMPLES}"
REFERENCE_OUT="$OUT/decoded_reference_n${DIST_SAMPLES}"
PRED_OUT="$OUT/predicted_clean_n${DIST_SAMPLES}"
FREE_OUT="$OUT/free_generation_n${FREE_SAMPLES}"

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

require_file() {
  [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}
require_dir() {
  [[ -d "$1" ]] || { echo "Missing directory: $1" >&2; exit 1; }
}

preflight() {
  require_file "$PY"
  require_file "$CONFIG"
  require_file "$CKPT"
  require_file "$INDEX_MAP"
  require_dir "$PARQUET"
  require_dir "$PACKED"
  require_file "$FID_REF"
  require_file "$REPO/experiments/run_raev2_lpl_component_audit.py"
  require_file "$REPO/experiments/run_raev2_decoded_distribution_audit.py"
  require_file "$REPO/experiments/run_raev2_predicted_clean_audit.py"
  require_file "$REPO/experiments/sample_raev2_threeway.py"
  require_file "$REPO/experiments/evaluate_raev2_samples.py"

  "$PY" - <<'PY'
import torch
import torch_fidelity
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
}

tag_for_scale() {
  local scale=$1
  echo "${scale/./p}"
}

run_paired() {
  mkdir -p "$PAIR_OUT"
  for scale in $SCALES; do
    tag=$(tag_for_scale "$scale")
    out="$PAIR_OUT/s_${tag}"
    if [[ -f "$out/component_audit_summary.csv" ]]; then
      echo "[paired] skip existing scale=$scale"
      continue
    fi
    echo "[paired] scale=$scale samples=$PAIR_SAMPLES"
    CUDA_VISIBLE_DEVICES=3 "$PY" \
      experiments/run_raev2_lpl_component_audit.py \
      --config "$CONFIG" \
      --data-path "$PARQUET" \
      --index-map "$INDEX_MAP" \
      --checkpoint "official=$CKPT" \
      --output-dir "$out" \
      --samples "$PAIR_SAMPLES" \
      --noise-ratio 0.3333333333333333 \
      --noise-ratio 1.0 \
      --noise-ratio 3.0 \
      --state-key model \
      --precision bf16 \
      --seed "$SEED" \
      --prediction-target guided \
      --guidance-scale "$scale" \
      --dino-ckpt-dir "$DINO_CKPT" \
      --dino-repo-dir "$DINO_REPO"
  done

  "$PY" - "$PAIR_OUT" $SCALES <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
scales = [float(x) for x in sys.argv[2:]]
frames = []
for scale in scales:
    tag = str(scale).replace(".", "p")
    path = root / f"s_{tag}" / "component_audit_summary.csv"
    frame = pd.read_csv(path)
    frame["requested_guidance_scale"] = scale
    frames.append(frame)
out = pd.concat(frames, ignore_index=True)
keep = [
    c for c in [
        "requested_guidance_scale",
        "noise_to_signal_ratio",
        "time",
        "latent_relative_error_rms",
        "prediction_over_target_variance",
        "flow_loss",
        "raw_loss",
        "prediction_detach_loss",
        "prediction_full_loss",
    ]
    if c in out.columns
]
out[keep].to_csv(root / "paired_error_curve.csv", index=False)
print(out[keep].to_string(index=False))
PY
}

run_reference() {
  if [[ -f "$REFERENCE_OUT/manifest.json" ]]; then
    echo "[reference] skip existing: $REFERENCE_OUT"
    return
  fi
  mkdir -p "$REFERENCE_OUT"
  echo "[reference] samples=$DIST_SAMPLES"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    experiments/run_raev2_decoded_distribution_audit.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --packed-data-path "$PACKED" \
    --parquet-data-path /data/shared/imagenet-1k \
    --fid-reference "$FID_REF" \
    --output-dir "$REFERENCE_OUT" \
    --samples "$DIST_SAMPLES" \
    --per-rank-batch 2 \
    --decode-batch 4 \
    --time 0.0 \
    --time 0.25 \
    --time 0.50 \
    --time 0.75 \
    --time 1.0 \
    --ig-scale 1.78 \
    --state-key model \
    --precision bf16 \
    --inception-feature 2048 \
    --seed "$SEED" \
    --dino-ckpt-dir "$DINO_CKPT" \
    --dino-repo-dir "$DINO_REPO"
}

run_predicted() {
  require_file "$REFERENCE_OUT/manifest.json"
  mkdir -p "$PRED_OUT"
  for scale in $SCALES; do
    tag=$(tag_for_scale "$scale")
    out="$PRED_OUT/s_${tag}"
    if [[ -f "$out/predicted_clean_summary.csv" ]]; then
      echo "[predicted] skip existing scale=$scale"
      continue
    fi
    echo "[predicted] scale=$scale samples=$DIST_SAMPLES"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      experiments/run_raev2_predicted_clean_audit.py \
      --config "$CONFIG" \
      --checkpoint "$CKPT" \
      --decoded-reference-run "$REFERENCE_OUT" \
      --fid-reference "$FID_REF" \
      --output-dir "$out" \
      --samples "$DIST_SAMPLES" \
      --per-rank-batch 2 \
      --time 0.25 \
      --time 0.50 \
      --time 0.75 \
      --time 1.0 \
      --ig-scale "$scale" \
      --state-key model \
      --precision bf16 \
      --inception-feature 2048 \
      --seed "$SEED" \
      --dino-ckpt-dir "$DINO_CKPT" \
      --dino-repo-dir "$DINO_REPO"
  done

  "$PY" - "$PRED_OUT" $SCALES <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
scales = [float(x) for x in sys.argv[2:]]
frames = []
for scale in scales:
    tag = str(scale).replace(".", "p")
    path = root / f"s_{tag}" / "predicted_clean_summary.csv"
    frame = pd.read_csv(path)
    # The on-policy IG condition is the actual guided prediction on its own trajectory.
    frame = frame[frame["condition"] == "ig_on_ig"].copy()
    frame["requested_guidance_scale"] = scale
    frames.append(frame)
out = pd.concat(frames, ignore_index=True)
keep = [
    c for c in [
        "requested_guidance_scale",
        "requested_time",
        "actual_time",
        "condition",
        "auc",
        "auc_separability",
        "fid_real",
        "fid_reconstruction",
    ]
    if c in out.columns
]
out[keep].to_csv(root / "predicted_clean_distribution_curve.csv", index=False)
print(out[keep].to_string(index=False))
PY
}

run_free() {
  mkdir -p "$FREE_OUT"
  for scale in $SCALES; do
    tag=$(tag_for_scale "$scale")
    run_dir="$FREE_OUT/s_${tag}"
    archive="$run_dir/official/samples.npz"
    if [[ -f "$archive" ]]; then
      echo "[free] skip existing scale=$scale"
      continue
    fi
    echo "[free] scale=$scale samples=$FREE_SAMPLES"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run \
      --standalone \
      --nproc_per_node=4 \
      experiments/sample_raev2_threeway.py \
      --config "$CONFIG" \
      --branch "official=$CKPT" \
      --results-dir "$run_dir" \
      --sample-count "$FREE_SAMPLES" \
      --per-rank-batch 8 \
      --sampling-seed "$SEED" \
      --precision bf16 \
      --state-key model \
      --ig-scale "$scale" \
      --dino-ckpt-dir "$DINO_CKPT" \
      --dino-repo-dir "$DINO_REPO"
  done
}

run_evaluate() {
  args=()
  for scale in $SCALES; do
    tag=$(tag_for_scale "$scale")
    archive="$FREE_OUT/s_${tag}/official/samples.npz"
    require_file "$archive"
    args+=(--branch "official_s${tag}=$archive")
  done

  "$PY" experiments/evaluate_raev2_samples.py \
    "${args[@]}" \
    --reference "$FID_REF" \
    --output "$FREE_OUT/metrics.csv" \
    --batch-size 64 \
    --seed "$SEED"
}

preflight

case "$MODE" in
  paired)
    run_paired
    ;;
  reference)
    run_reference
    ;;
  predicted)
    run_predicted
    ;;
  free)
    run_free
    ;;
  evaluate)
    run_evaluate
    ;;
  all)
    run_paired
    run_reference
    run_predicted
    run_free
    run_evaluate
    ;;
  *)
    echo "Usage: $0 {paired|reference|predicted|free|evaluate|all}" >&2
    exit 2
    ;;
esac

echo "Done. Output root: $OUT"