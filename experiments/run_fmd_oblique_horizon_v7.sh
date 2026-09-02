#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

while tmux has-session -t fmd_weak_forecast_v6 2>/dev/null; do
  sleep 30
done

python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v7/oblique_horizon_fid1k \
  --fmd-decomposition-components weak_drift_oblique \
  --fmd-oblique-specs 0.0234375:1.8:0.25,0.0234375:2.4:0.25,0.0234375:3:0.25,0.03125:1.35:0.25,0.03125:1.8:0.25,0.03125:2.25:0.25,0.0390625:1.08:0.25,0.0390625:1.44:0.25,0.0390625:1.8:0.25,0.0546875:0.75:0.25,0.0546875:1.025:0.25,0.0546875:1.275:0.25,0.0625:0.675:0.25,0.0625:0.9:0.25,0.0625:1.125:0.25,0.078125:0.54:0.25,0.078125:0.72:0.25,0.078125:0.9:0.25,0.09375:0.45:0.25,0.09375:0.6:0.25,0.09375:0.75:0.25 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_oblique_horizon_v7.log
