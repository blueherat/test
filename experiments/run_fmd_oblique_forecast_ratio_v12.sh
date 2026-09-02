#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# The forecast factor r=eta/gamma(t) is dimensionless. Unlike a fixed eta, a
# fixed r preserves the same effective weak-reference extrapolation across the
# two historical IG gamma segments.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --seed 0 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v12/oblique_forecast_ratio_fid1k \
  --fmd-decomposition-components weak_reference_forecast \
  --fmd-decomposition-horizons 0.02734375 \
  --fmd-forecast-factors 2.25,2.4,2.55,2.7,2.85,3.0 \
  --fmd-oblique-alphas 0.25 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_oblique_forecast_ratio_v12.log
