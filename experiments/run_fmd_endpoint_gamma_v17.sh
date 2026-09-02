#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# Jointly calibrate only the original static IG gap after fixing the invariant
# posterior-revision operator.  Multipliers act on gamma=(.6,.7) in the first
# and second IG intervals, respectively.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --seed 0 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v17/endpoint_gamma_fid1k \
  --fmd-decomposition-components weak_endpoint_contrast_segmented \
  --fmd-decomposition-horizons 0.02734375 \
  --fmd-oblique-alphas 0.25 \
  --fmd-strength-segment-pairs 1.65:1.65 \
  --fmd-gamma-segment-pairs 0.9:0.9,0.9:1,0.9:1.1,1:0.9,1:1.1,1.1:0.9,1.1:1,1.1:1.1 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_endpoint_gamma_v17.log
