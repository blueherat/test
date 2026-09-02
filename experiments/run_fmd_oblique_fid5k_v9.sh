#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

while tmux has-session -t fmd_oblique_fine_v8 2>/dev/null; do
  sleep 30
done

run_eval() {
  local gpu="$1"
  local seed="$2"
  local tag="$3"
  local horizon="$4"
  local strength="$5"

  python experiments/run_imagenet100_sit_path_extrapolated_ig.py sweep \
    --gpus "${gpu}" \
    --num-samples 5000 \
    --batch-size 8 \
    --vae-decode-batch-size 2 \
    --fid-batch-size 16 \
    --fid-gpu-memory-fraction 0.25 \
    --seed "${seed}" \
    --no-include-global-anchor \
    --output-root "/data/users/zhoushunyu/eqvae/imagenet_sit_flow/fmd_geometric_v9/${tag}_seed${seed}_fid5k" \
    --fmd-decomposition-components weak_drift_oblique \
    --fmd-decomposition-horizons "${horizon}" \
    --fmd-decomposition-strengths "${strength}" \
    --fmd-oblique-alphas 0.25 \
    --condition-regex '^fmd_decomposition_' \
    > "/tmp/fmd_${tag}_seed${seed}_fid5k.log" 2>&1
}

# The short-horizon candidate is theory-selected by the constant first-order
# gain eta*h=0.05625. The fine candidate is its stable local refinement: the
# neighboring alpha values are worse and alpha=0.25 was predicted independently
# by the temporal/state material split before this sweep.
run_eval 0 0 fine 0.02734375 1.65 &
pid0=$!
run_eval 1 1 fine 0.02734375 1.65 &
pid1=$!
run_eval 2 0 short 0.03125 1.8 &
pid2=$!
run_eval 3 1 short 0.03125 1.8 &
pid3=$!

wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

for log in /tmp/fmd_fine_seed{0,1}_fid5k.log /tmp/fmd_short_seed{0,1}_fid5k.log; do
  tail -n 2 "${log}"
done
