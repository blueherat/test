#!/usr/bin/env bash
set -euo pipefail

while tmux has-session -t imagenette_dual_target_v2 2>/dev/null; do
  sleep 30
done

cd /home/zhoushunyu/eqvae

OUTPUT_ROOT=/home/zhoushunyu/data/eqvae/imagenette_dual_target_latent/formal_direct_v1 \
LOSS_SPACE=direct \
STEPS=20000 \
  bash experiments/run_imagenette_dual_target_latent_4gpu.sh
