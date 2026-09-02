#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# lambda=0 keeps the Richardson first-directional term; lambda=1 is exactly
# the established finite-horizon operator. Values above one extrapolate the
# measured finite-horizon curvature instead of adding an unrelated direction.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --seed 0 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v13/oblique_curvature_fid1k \
  --fmd-decomposition-components weak_drift_oblique_curvature \
  --fmd-decomposition-horizons 0.02734375 \
  --fmd-decomposition-strengths 1.65 \
  --fmd-oblique-alphas 0.25 \
  --fmd-curvature-weights 0,0.5,0.75,1,1.25,1.5,2 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_oblique_curvature_v13.log
