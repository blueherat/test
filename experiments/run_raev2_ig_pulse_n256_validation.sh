#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhoushunyu/eqvae"
PYTHON="/home/zhoushunyu/data/eqvae/envs/raev2/bin/python"
OUTPUT_ROOT="/home/zhoushunyu/data/eqvae/experiments/raev2_ig_impulse_response"

is_complete() {
    local manifest="$1/manifest.json"
    [[ -f "$manifest" ]] && "$PYTHON" -c \
        'import json,sys; raise SystemExit(json.load(open(sys.argv[1]))["status"] != "complete")' \
        "$manifest"
}

is_running() {
    pgrep -f "run_raev2_ig_impulse_response.py --output-dir $1" >/dev/null
}

run_or_resume_seed() {
    local seed="$1"
    local label_mode="$2"
    local output_dir="$OUTPUT_ROOT/n256_seed${seed}_pulse_g001_g005_v1"
    local log="$output_dir"_console.log

    mkdir -p "$output_dir"
    while ! is_complete "$output_dir"; do
        if is_running "$output_dir"; then
            sleep 60
            continue
        fi

        cd "$ROOT"
        CUDA_VISIBLE_DEVICES=2,3 \
        MPLCONFIGDIR="/tmp/mpl-raev2-ig-pulse-n256-${seed}" \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
            experiments/run_raev2_ig_impulse_response.py \
            --output-dir "$output_dir" \
            --samples 256 \
            --per-rank-batch 1 \
            --condition-group-size 4 \
            --pulse-steps 10,30,50,70,90,98 \
            --pulse-gammas 0.01,0.05 \
            --skip-windows \
            --window-count 5 \
            --gamma 0.05 \
            --precision fp32 \
            --seed "$seed" \
            --label-mode "$label_mode" \
            --label-seed "$seed" \
            --bootstrap-repeats 5000 \
            --log-every-samples 4 2>&1 | tee -a "$log"
    done
}

run_or_resume_seed 20260812 sequential
run_or_resume_seed 20260813 random_without_replacement
