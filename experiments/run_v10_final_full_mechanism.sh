#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae
V10=experiments/run_prediction_target_extrapolation_toy_v10_final.py

if [[ -d /data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_main ]]; then
  DATA=/data/users/zhoushunyu/eqvae/experiments
elif [[ -d /home/zhoushunyu/data/eqvae/experiments/prediction_target_toy_v4_main ]]; then
  DATA=/home/zhoushunyu/data/eqvae/experiments
else
  echo "Could not find prediction_target_toy_v4_main." >&2
  exit 2
fi

SRC="$DATA/prediction_target_toy_v4_main"
OUT="$DATA/prediction_target_toy_v10_final_full_mechanism"
EXPECTED_CONDITIONS=187
mkdir -p "$OUT"

# This is evaluation-only. Every worker reuses the corresponding archived v4
# checkpoint and executes all default v10 mechanism conditions. The fourth GPU
# repeats seed 20260807 outside worker_seed*/ so it tests determinism without
# entering the three-independent-seed aggregate.
pids=()
for item in \
  "0 20260807 worker_seed20260807" \
  "1 20260808 worker_seed20260808" \
  "2 20260809 worker_seed20260809" \
  "3 20260807 replay_seed20260807"; do
  set -- $item
  GPU=$1
  SEED=$2
  WORKER=$3
  SETTING="$SRC/seed${SEED}/D512/curv1/scale_unit_rms/loss_v/H1024"
  if [[ ! -f "$SETTING/models.pt" ]]; then
    echo "Missing checkpoint: $SETTING/models.pt" >&2
    exit 3
  fi

  CUDA_VISIBLE_DEVICES=$GPU python "$V10" \
    --output-root "$OUT/$WORKER" \
    --dims 512 \
    --curvatures 1.0 \
    --hidden-dims 1024 \
    --seeds "$SEED" \
    --depth 5 \
    --time-dim 32 \
    --frequency-scale 6.0 \
    --scale-mode unit_rms \
    --data-jitter 0.015 \
    --conversion-clip 0.02 \
    --training-mode fixed \
    --loss-space v \
    --reuse-v4-setting "$SETTING" \
    --condition-suite mechanism \
    --sample-count 10000 \
    --reference-count 20000 \
    --sample-batch-size 1024 \
    --batch-conditions \
    --condition-batch-size 16 \
    --sample-steps 200 \
    --sample-t-max 0.98 \
    --sample-t-min 0.02 \
    --diag-stride 5 \
    --ridge-grid-points 8192 \
    --coverage-bins 100 \
    --conditional-ridge-bins 20 \
    --conditional-ridge-min-count 20 \
    --swd-projections 256 \
    --swd-max-points 10000 \
    --full-swd-projections 256 \
    --full-swd-max-points 10000 \
    --mmd-max-points 2048 \
    --bootstrap-reps 200 \
    --bootstrap-max-points 2048 \
    --bootstrap-projections 64 \
    --geometry-metric-points 10000 \
    --curveD-coarse-points 512 \
    --curveD-refine-points 9 \
    --curveD-refine-rounds 2 \
    --surface-audit-samples 512 \
    --surface-audit-iterations 6 \
    --surface-audit-damping 1e-5 \
    --observability-samples 8192 \
    --plot-points 4000 \
    --visual-atlas \
    --device cuda \
    > "/tmp/v10_final_full_mechanism_${WORKER}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  echo "At least one v10 worker failed. Check /tmp/v10_final_full_mechanism_*.log" >&2
  exit $status
fi

python - "$OUT" "$EXPECTED_CONDITIONS" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
for path in sorted(root.glob("*/generation_metrics_v10_all.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} conditions, found {count}")
PY

python "$V10" --output-root "$OUT" --aggregate-only
touch "$OUT/COMPLETE"
echo "Finished full v10 mechanism study: $OUT"
