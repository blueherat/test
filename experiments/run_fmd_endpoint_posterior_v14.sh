#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

# Causal split of the established finite oblique weak-velocity change.
# For the linear bridge, clean_change + negative_noise_change is exactly
# W(z,t) - W(z + alpha*h*G, t+h).  The matched controls remove component
# scale as a confound while preserving each component's direction.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0,1,2,3 \
  --num-samples 1000 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --fid-batch-size 16 \
  --fid-gpu-memory-fraction 0.25 \
  --seed 0 \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v14/endpoint_posterior_fid1k \
  --fmd-decomposition-components weak_drift_oblique,weak_clean_endpoint_drift,weak_clean_endpoint_drift_matched,weak_noise_endpoint_drift,weak_noise_endpoint_drift_matched \
  --fmd-decomposition-horizons 0.02734375 \
  --fmd-decomposition-strengths 1.65 \
  --fmd-oblique-alphas 0.25 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_endpoint_posterior_v14.log
