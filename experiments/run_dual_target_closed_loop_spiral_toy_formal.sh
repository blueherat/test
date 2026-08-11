#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zhoushunyu/data/eqvae/experiments/dual_target_closed_loop_spiral_toy_v1}"
DEVICE="${DEVICE:-cuda}"

python experiments/run_dual_target_closed_loop_spiral_toy.py \
  --output-root "$OUTPUT_ROOT" \
  --dims 2,512 \
  --seeds 20260831,20260901,20260902 \
  --hidden-dim 128 \
  --depth 4 \
  --data-jitter 0.015 \
  --curvature 0 \
  --scale-mode unit_rms \
  --quadrature-points 1024 \
  --locator-points 4096 \
  --train-steps 15000 \
  --batch-size 2048 \
  --teacher-samples 4096 \
  --gradient-samples 2048 \
  --sample-count 4096 \
  --reference-count 8192 \
  --sample-steps 200 \
  --swd-projections 256 \
  --swd-max-points 4096 \
  --full-swd-projections 64 \
  --full-swd-max-points 4096 \
  --mmd-max-points 2048 \
  --device "$DEVICE"

python experiments/summarize_dual_target_closed_loop_toy.py \
  --root "$OUTPUT_ROOT"

python experiments/summarize_dual_target_closed_loop_spiral_toy.py \
  --root "$OUTPUT_ROOT"

python experiments/audit_dual_target_spiral_solver_convergence.py \
  --output-root "$OUTPUT_ROOT" \
  --device "$DEVICE"

python experiments/validate_dual_target_closed_loop_spiral_results.py \
  --root "$OUTPUT_ROOT"
