#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# C_lambda=C_x+lambda*C_negative_eps.  Lambda=1 is the unique linear
# combination that cancels the common displacement of the oblique query and
# is bitwise identical to the established weak-velocity contrast.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --seed 0 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v15/endpoint_contrast_fid1k \
  --fmd-decomposition-components weak_endpoint_contrast \
  --fmd-decomposition-horizons 0.02734375 \
  --fmd-decomposition-strengths 1.65 \
  --fmd-oblique-alphas 0.25 \
  --fmd-endpoint-noise-weights 0.5,0.75,0.875,1,1.125,1.25,1.5 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_endpoint_contrast_v15.log
