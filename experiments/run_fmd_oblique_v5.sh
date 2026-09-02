#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

while tmux has-session -t fmd_characteristic_v4 2>/dev/null; do
  sleep 30
done

COMMON=(
  --gpus 0,1,2,3
  --num-samples 1000
  --batch-size 8
  --vae-decode-batch-size 2
  --fid-batch-size 16
  --fid-gpu-memory-fraction 0.25
  --no-include-global-anchor
  --condition-regex '^fmd_decomposition_'
)

# Alpha=1 must be exactly identical to the historical guided-characteristic
# query before the new family is allowed into a metric sweep.
python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  --gpus 0 \
  --num-samples 16 \
  --batch-size 8 \
  --vae-decode-batch-size 2 \
  --skip-fid \
  --no-include-global-anchor \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v5/oblique_anchor_smoke \
  --fmd-decomposition-components weak_drift_path_mix,weak_drift_oblique \
  --fmd-decomposition-horizons 0.046875 \
  --fmd-decomposition-strengths 0.6 \
  --fmd-characteristic-rhos 1 \
  --fmd-oblique-alphas 1 \
  --condition-regex '^fmd_decomposition_' \
  2>&1 | tee /tmp/fmd_oblique_anchor_v5.log

cmp \
  /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v5/oblique_anchor_smoke/fmd_decomposition_weak_drift_path_mix_h0p046875_eta0p6_rho1/preview.png \
  /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v5/oblique_anchor_smoke/fmd_decomposition_weak_drift_oblique_h0p046875_eta0p6_alpha1/preview.png

python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
  "${COMMON[@]}" \
  --output-root /data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v5/oblique_fid1k \
  --fmd-decomposition-components weak_drift_oblique \
  --fmd-decomposition-horizons 0.046875 \
  --fmd-decomposition-strengths 0.9,1.05,1.2,1.35,1.5 \
  --fmd-oblique-alphas 0,0.125,0.25,0.375,0.5 \
  2>&1 | tee /tmp/fmd_oblique_v5.log
