#!/usr/bin/env bash
set -euo pipefail
cd /home/zhoushunyu/eqvae
exec > >(tee -a /home/zhoushunyu/data/eqvae/experiments/raev2_shallow_ig_20260914/tmux_supervisor.log) 2>&1
exec /home/zhoushunyu/miniconda3/envs/myenv/bin/python -u -m experiments.raev2_shallow_ig_20260914.tmux_sweep
