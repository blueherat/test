#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

COMMON=(
  --gpus 0,1,2,3
  --num-samples 1000
  --batch-size 8
  --vae-decode-batch-size 2
  --fid-batch-size 16
  --fid-gpu-memory-fraction 0.25
  --no-include-global-anchor
  --fmd-decomposition-horizons 0.046875
  --condition-regex '^fmd_decomposition_'
)

python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  "${COMMON[@]}" \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v4/extended_fid1k \
  --fmd-decomposition-components weak_drift_extended \
  --fmd-extended-specs 0.6:0:0.625,0.6:0.1:0.625,0.6:0.2:0.625,0.6:0.3:0.625,0.6:0.4:0.625,0.6:0.6:0.625,0.6:0:0.75,0.6:0.1:0.75,0.6:0.2:0.75,0.6:0.3:0.75,0.6:0.4:0.75,0.6:0.6:0.75,0.6:0:0.875,0.6:0.1:0.875,0.6:0.2:0.875,0.6:0.3:0.875,0.6:0.4:0.875,0.6:0.6:0.875,0.6:0:1,0.6:0.1:1,0.6:0.2:1,0.6:0.3:1,0.6:0.4:1,0.6:0.6:1 \
  2>&1 | tee /tmp/fmd_extended_v4.log

python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  "${COMMON[@]}" \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v4/path_mix_fid1k \
  --fmd-decomposition-components weak_drift_path_mix \
  --fmd-decomposition-strengths 0.45,0.525,0.6,0.675 \
  --fmd-characteristic-rhos 0,0.3,0.6,0.8,1,1.2,1.5 \
  2>&1 | tee /tmp/fmd_path_mix_v4.log

python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  "${COMMON[@]}" \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v4/material_fid1k \
  --fmd-decomposition-components weak_material_anisotropic \
  --fmd-material-pairs 0:0,0:0.3,0:0.6,0:0.9,0:1.2,0.3:0,0.3:0.3,0.3:0.6,0.3:0.9,0.3:1.2,0.6:0,0.6:0.3,0.6:0.6,0.6:0.9,0.6:1.2,0.9:0,0.9:0.3,0.9:0.6,0.9:0.9,0.9:1.2,1.2:0,1.2:0.3,1.2:0.6,1.2:0.9,1.2:1.2 \
  2>&1 | tee /tmp/fmd_material_v4.log

python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  "${COMMON[@]}" \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v4/segmented_fid1k \
  --fmd-decomposition-components weak_drift_segmented \
  --fmd-strength-segment-pairs 0.3:0.3,0.3:0.45,0.3:0.6,0.3:0.75,0.3:0.9,0.45:0.3,0.45:0.45,0.45:0.6,0.45:0.75,0.45:0.9,0.6:0.3,0.6:0.45,0.6:0.6,0.6:0.75,0.6:0.9,0.75:0.3,0.75:0.45,0.75:0.6,0.75:0.75,0.75:0.9,0.9:0.3,0.9:0.45,0.9:0.6,0.9:0.75,0.9:0.9 \
  2>&1 | tee /tmp/fmd_segmented_v4.log
