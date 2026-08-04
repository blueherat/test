#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhoushunyu/eqvae"
PYTHON="/home/zhoushunyu/data/eqvae/envs/raev2/bin/python"
OUTPUT_ROOT="/home/zhoushunyu/data/eqvae/experiments/sit_ig_endpoint_dynamics"

is_complete() {
    local manifest="$1/manifest.json"
    [[ -f "$manifest" ]] && "$PYTHON" -c \
        'import json,sys; raise SystemExit(json.load(open(sys.argv[1]))["status"] != "complete")' \
        "$manifest"
}

is_running() {
    pgrep -f "run_sit_ig_endpoint_dynamics.py --output-dir $1" >/dev/null
}

run_or_resume_seed() {
    local seed="$1"
    local output_dir="$OUTPUT_ROOT/n128_seed${seed}_steps50_v1"
    local log="$output_dir"_console.log

    mkdir -p "$output_dir"
    while ! is_complete "$output_dir"; do
        if is_running "$output_dir"; then
            sleep 60
            continue
        fi
        cd "$ROOT"
        CUDA_VISIBLE_DEVICES=1 \
        MPLCONFIGDIR="/tmp/mpl-sit-ig-endpoint-${seed}" \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
            experiments/run_sit_ig_endpoint_dynamics.py \
            --output-dir "$output_dir" \
            --samples 128 \
            --num-steps 50 \
            --per-rank-batch 2 \
            --condition-group-size 8 \
            --pulse-steps 5,15,25,35,45,49 \
            --pulse-gammas 0.01,0.05 \
            --window-count 5 \
            --window-gamma 0.01 \
            --interaction-windows 0,2,4 \
            --seed "$seed" \
            --label-mode random_without_replacement \
            --label-seed "$seed" \
            --bootstrap-repeats 5000 \
            --log-every-samples 4 2>&1 | tee -a "$log"
    done
}

run_or_resume_seed 20260812
run_or_resume_seed 20260813
