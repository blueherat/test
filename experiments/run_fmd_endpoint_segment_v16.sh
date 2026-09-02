#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# The invariant endpoint-posterior contrast is fixed.  Only its gain is split
# across the two established IG intervals to measure and tune time-local
# posterior revision without introducing another spatial direction.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --seed 0 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v16/endpoint_segment_fid1k \
  --fmd-decomposition-components weak_endpoint_contrast_segmented \
  --fmd-decomposition-horizons 0.02734375 \
  --fmd-oblique-alphas 0.25 \
  --fmd-strength-segment-pairs 0:1.65,1.65:0,1.35:1.35,1.35:1.65,1.35:1.95,1.65:1.35,1.65:1.65,1.65:1.95,1.95:1.35,1.95:1.65,1.95:1.95 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_endpoint_segment_v16.log
