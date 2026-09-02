#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

run_eval() {
  local gpu="$1"
  local tag="$2"
  local horizon="$3"
  local strength="$4"
  local iterations="$5"

  python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
    --gpus "${gpu}" \
    --num-samples 1000 \
    --batch-size 8 \
    --vae-decode-batch-size 2 \
    --fid-batch-size 16 \
    --fid-gpu-memory-fraction 0.25 \
    --seed 0 \
    --no-include-global-anchor \
    --output-root "/data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v11/${tag}_fid1k" \
    --fmd-decomposition-components weak_drift_oblique_picard \
    --fmd-decomposition-horizons "${horizon}" \
    --fmd-decomposition-strengths "${strength}" \
    --fmd-oblique-alphas 0.25 \
    --fmd-picard-iterations "${iterations}" \
    --condition-regex '^fmd_decomposition_' \
    > "/tmp/fmd_oblique_picard_v11_${tag}.log" 2>&1
}

# K=1 is exactly the established oblique method. These conditions only test
# whether making its future weak-head query self-consistent improves it.
run_eval 0 fine_k2 0.02734375 1.65 2 &
pid0=$!
run_eval 1 fine_k3 0.02734375 1.65 3 &
pid1=$!
run_eval 2 short_k2 0.03125 1.8 2 &
pid2=$!
run_eval 3 short_k3 0.03125 1.8 3 &
pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

for log in /tmp/fmd_oblique_picard_v11_{fine,short}_k{2,3}.log; do
  tail -n 2 "${log}"
done
