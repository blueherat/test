#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

while tmux has-session -t fmd_oblique_v5 2>/dev/null; do
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
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v6/weak_forecast_fid1k \
  --fmd-decomposition-components weak_reference_forecast \
  --fmd-decomposition-horizons 0.046875 \
  --fmd-forecast-factors 1,1.5,1.75,2,2.25,2.5 \
  --fmd-oblique-alphas 0,0.125,0.25,0.375,0.5 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_weak_forecast_v6.log
