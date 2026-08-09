#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

GROUP=${1:?usage: $0 GROUP SEEDS}
SEEDS=${2:?usage: $0 GROUP SEEDS}
DATA_ROOT=/home/zhoushunyu/data/eqvae/experiments

COMMON=(
  --dims 512
  --frequency-scale 6.0
  --depth 5
  --time-dim 32
  --train-steps 30000
  --oracle-hidden-dim 2048
  --oracle-depth 6
  --oracle-train-steps 60000
  --batch-size 1024
  --lr 0.0002
  --weight-decay 0
  --grad-clip 10
  --t-min 0.02
  --t-max 0.98
  --conversion-clip 0.02
  --data-jitter 0.015
  --log-every 1000
  --eval-times 0.1,0.3,0.5,0.7,0.9
  --eval-samples 8192
  --eval-batch-size 1024
  --sample-count 10000
  --sample-batch-size 1000
  --sample-condition-batch-size 16
  --sample-steps 200
  --sample-t-max 0.98
  --sample-t-min 0.02
  --gammas=-0.1,-0.03,-0.01,0.003,0.01,0.03,0.1,0.3
  --normalized-etas=-0.03,-0.01,0.01,0.03
  --swd-projections 256
  --metric-max-points 4096
  --bootstrap-reps 100
  --bootstrap-max-points 1024
  --bootstrap-projections 64
  --plot-points 4000
  --seeds "$SEEDS"
  --device cuda
  --generation-profile full
  --sampling-execution batched
  --save-checkpoints
  --resume
)

case "$GROUP" in
  main)
    ROOT="$DATA_ROOT/prediction_target_toy_v4_main"
    EXTRA=(
      --curvatures 0,0.25,0.5,1.0
      --hidden-dims 64,128,256,512,1024
      --loss-spaces v
      --scale-mode unit_rms
      --skip-root-summary
    )
    ;;
  constant_norm)
    ROOT="$DATA_ROOT/prediction_target_toy_v4_constant_norm"
    EXTRA=(
      --curvatures 0,0.5,1.0
      --hidden-dims 128,256,512
      --loss-spaces v
      --scale-mode constant_norm
    )
    ;;
  direct_loss)
    ROOT="$DATA_ROOT/prediction_target_toy_v4_direct_loss"
    EXTRA=(
      --curvatures 0.5
      --hidden-dims 128,256,512
      --loss-spaces direct
      --scale-mode unit_rms
    )
    ;;
  *)
    echo "unknown group: $GROUP" >&2
    exit 2
    ;;
esac

mkdir -p "$ROOT"
LOG="$ROOT/worker_${GROUP}_seeds_${SEEDS//,/_}.log"

python -u experiments/run_prediction_target_extrapolation_toy_v4.py \
  --output-root "$ROOT" \
  "${COMMON[@]}" \
  "${EXTRA[@]}" \
  2>&1 | tee -a "$LOG"
