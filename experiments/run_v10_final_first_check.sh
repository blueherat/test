#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae
V10=experiments/run_prediction_target_extrapolation_toy_v10_final.py

# Reuse the archived v4 H=1024 models, but v10 uses its own evaluation
# randomness/metric protocol. Detect the local archive root rather than
# silently assuming one storage layout.
if [[ -d /data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_main ]]; then
  DATA=/data/users/zhoushunyu/eqvae/experiments
elif [[ -d /home/zhoushunyu/data/eqvae/experiments/prediction_target_toy_v4_main ]]; then
  DATA=/home/zhoushunyu/data/eqvae/experiments
else
  echo "Could not find prediction_target_toy_v4_main under the known experiment roots." >&2
  exit 2
fi

SRC="$DATA/prediction_target_toy_v4_main"
OUT="$DATA/prediction_target_toy_v10_final_first_check"
mkdir -p "$OUT"

# Focused adjudication:
#   - x-eps raw positive extrapolation where v4 showed the interesting effect;
#   - x-v raw controls;
#   - absolute RMS-normalized +/- directions including the known xv eta=+0.01
#     visual counterexample;
#   - primary endpoint metrics use all 10k generated samples whenever the
#     metric is O(n) / O(n log n).  Only MMD/bootstrap/local-surface audit use
#     smaller subsets because their cost scales quadratically/iteratively.

pids=()
# The archive contains three independent training seeds.  The fourth GPU runs
# an identical seed-20260807 replay in a non-aggregate directory: it uses all
# four cards while also giving us a deterministic end-to-end reproducibility
# check without pretending that the replay is a fourth independent seed.
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
    --gamma-xeps=0.01,0.03,0.1,0.3 \
    --gamma-xv=0.03,0.1 \
    --path-alphas=0,1 \
    --absolute-actions=-0.03,-0.01,0.01,0.03 \
    --relative-actions= \
    --control-gammas= \
    --geometry-gammas= \
    --geometry-relative-actions= \
    --geometry-absolute-actions= \
    --stage-gammas= \
    --sample-count 10000 \
    --reference-count 20000 \
    --sample-batch-size 1024 \
    --batch-conditions \
    --condition-batch-size 17 \
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
    --save-samples \
    --device cuda \
    > "/tmp/v10_final_first_check_${WORKER}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  echo "At least one v10 worker failed. Check /tmp/v10_final_first_check_*.log" >&2
  exit $status
fi

python "$V10" --output-root "$OUT" --aggregate-only
touch "$OUT/COMPLETE"

echo "Finished. Results: $OUT"
