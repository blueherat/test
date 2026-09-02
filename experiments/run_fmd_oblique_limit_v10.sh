#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

while tmux has-session -t fmd_oblique_fid5k_v9 2>/dev/null; do
  sleep 30
done

# Keep c=eta*h near 0.045 while shrinking the finite-difference horizon.
# This distinguishes the directional-derivative limit from a beneficial
# finite-horizon curvature effect without changing the oblique direction.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v10/oblique_limit_fid1k \
  --fmd-decomposition-components weak_drift_oblique \
  --fmd-oblique-specs \
0.015625:2.7:0.25,0.015625:2.9:0.25,0.015625:3.1:0.25,\
0.01953125:2.2:0.25,0.01953125:2.3:0.25,0.01953125:2.4:0.25,\
0.0234375:1.95:0.25,0.0234375:2.1:0.25 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_oblique_limit_v10.log
