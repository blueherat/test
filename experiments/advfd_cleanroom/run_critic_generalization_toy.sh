#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

python -m experiments.advfd_cleanroom.run_critic_generalization_toy \
  --output-root /data/users/zhoushunyu/eqvae/experiments/advfd_cleanroom_critic_toy_v2 \
  --seeds 8101,8102,8103 \
  --regimes matched,shift \
  --modes none,real,pooled \
  --steps 1000 \
  --eval-every 25 \
  --train-samples 256 \
  --heldout-samples 4096 \
  --device cuda:3
