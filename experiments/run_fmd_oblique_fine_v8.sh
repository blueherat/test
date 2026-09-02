#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

while tmux has-session -t fmd_oblique_horizon_v7 2>/dev/null; do
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
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v8/oblique_fine_fid1k \
  --fmd-decomposition-components weak_drift_oblique \
  --fmd-decomposition-horizons 0.02734375,0.03125,0.03515625 \
  --fmd-decomposition-strengths 1.65,1.8,1.95 \
  --fmd-oblique-alphas 0.1875,0.25,0.3125 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_oblique_fine_v8.log
