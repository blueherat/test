#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

python experiments/run_imagenet100_sit_checkpoint_reference_long_study_v1.py \
  --gpus 0,2
