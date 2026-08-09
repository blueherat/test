#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# This is the full follow-up grid, not the first screen. Run
# run_prediction_target_toy_v4_curved_screen.sh first. Checkpoints and
# condition-level resume are mandatory because this grid is intentionally large.

# ---------------------------------------------------------------------------
# Group A: main experiment.
# Curved manifold + capacity sweep + large x-oracle, common v-space loss.
# ---------------------------------------------------------------------------
python experiments/run_prediction_target_extrapolation_toy_v4.py \
  --output-root /data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_main \
  --dims 512 \
  --curvatures 0,0.25,0.5,1.0 \
  --hidden-dims 64,128,256,512,1024 \
  --loss-spaces v \
  --scale-mode unit_rms \
  --frequency-scale 6.0 \
  --train-steps 30000 \
  --oracle-hidden-dim 2048 \
  --oracle-depth 6 \
  --oracle-train-steps 60000 \
  --batch-size 1024 \
  --eval-samples 8192 \
  --sample-count 10000 \
  --sample-steps 200 \
  --gammas=-0.1,-0.03,-0.01,0.003,0.01,0.03,0.1,0.3 \
  --normalized-etas=-0.03,-0.01,0.01,0.03 \
  --bootstrap-reps 100 \
  --seeds 20260807,20260808,20260809 \
  --device cuda \
  --generation-profile core \
  --save-checkpoints \
  --resume \
  2>&1 | tee /tmp/prediction_target_toy_v4_main.log

# ---------------------------------------------------------------------------
# Group B: reproduce old constant-total-norm scaling as a control.
# Fewer settings are enough because this is a scaling control.
# ---------------------------------------------------------------------------
python experiments/run_prediction_target_extrapolation_toy_v4.py \
  --output-root /data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_constant_norm \
  --dims 512 \
  --curvatures 0,0.5,1.0 \
  --hidden-dims 128,256,512 \
  --loss-spaces v \
  --scale-mode constant_norm \
  --frequency-scale 6.0 \
  --train-steps 30000 \
  --oracle-hidden-dim 2048 \
  --oracle-train-steps 60000 \
  --batch-size 1024 \
  --eval-samples 8192 \
  --sample-count 10000 \
  --sample-steps 200 \
  --gammas=-0.1,-0.03,-0.01,0.003,0.01,0.03,0.1,0.3 \
  --normalized-etas=-0.03,-0.01,0.01,0.03 \
  --bootstrap-reps 100 \
  --seeds 20260807 \
  --device cuda \
  --generation-profile core \
  --save-checkpoints \
  --resume \
  2>&1 | tee /tmp/prediction_target_toy_v4_constant_norm.log

# ---------------------------------------------------------------------------
# Group C: direct-loss control.
# Separates target geometry from v-loss parameterization conditioning.
# ---------------------------------------------------------------------------
python experiments/run_prediction_target_extrapolation_toy_v4.py \
  --output-root /data/users/zhoushunyu/eqvae/experiments/prediction_target_toy_v4_direct_loss \
  --dims 512 \
  --curvatures 0.5 \
  --hidden-dims 128,256,512 \
  --loss-spaces direct \
  --scale-mode unit_rms \
  --frequency-scale 6.0 \
  --train-steps 30000 \
  --oracle-hidden-dim 2048 \
  --oracle-train-steps 60000 \
  --batch-size 1024 \
  --eval-samples 8192 \
  --sample-count 10000 \
  --sample-steps 200 \
  --gammas=-0.1,-0.03,-0.01,0.003,0.01,0.03,0.1,0.3 \
  --normalized-etas=-0.03,-0.01,0.01,0.03 \
  --bootstrap-reps 100 \
  --seeds 20260807,20260808,20260809 \
  --device cuda \
  --generation-profile core \
  --save-checkpoints \
  --resume \
  2>&1 | tee /tmp/prediction_target_toy_v4_direct_loss.log
