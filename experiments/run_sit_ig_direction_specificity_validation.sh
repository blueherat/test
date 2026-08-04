#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhoushunyu/eqvae"
PYTHON="/home/zhoushunyu/data/eqvae/envs/raev2/bin/python"
OUTPUT_ROOT="/home/zhoushunyu/data/eqvae/experiments/sit_ig_direction_specificity"
ENDPOINT_ROOT="/home/zhoushunyu/data/eqvae/experiments/sit_ig_endpoint_dynamics"

is_complete() {
    local manifest="$1/manifest.json"
    [[ -f "$manifest" ]] && "$PYTHON" -c \
        'import json,sys; raise SystemExit(json.load(open(sys.argv[1]))["status"] != "complete")' \
        "$manifest"
}

while ! is_complete "$ENDPOINT_ROOT/n128_seed20260812_steps50_v1" || \
      ! is_complete "$ENDPOINT_ROOT/n128_seed20260813_steps50_v1"; do
    sleep 60
done

"$PYTHON" experiments/summarize_sit_ig_endpoint_validations.py \
    --run-dirs \
        "$ENDPOINT_ROOT/n128_seed20260812_steps50_v1" \
        "$ENDPOINT_ROOT/n128_seed20260813_steps50_v1" \
    --output-dir "$ENDPOINT_ROOT/n128_seeds12_13_pooled_v1" \
    --bootstrap-repeats 5000 \
    --seed 20260814

run_smoke() {
    local output_dir="$OUTPUT_ROOT/smoke_n2_seed20260814_probes2_steps50_v1"
    local log="$output_dir"_console.log
    mkdir -p "$output_dir"
    while ! is_complete "$output_dir"; do
        if pgrep -f "run_sit_ig_direction_specificity.py --output-dir $output_dir" >/dev/null; then
            sleep 30
            continue
        fi
        cd "$ROOT"
        CUDA_VISIBLE_DEVICES=0 \
        MPLCONFIGDIR="/tmp/mpl-sit-ig-specificity-smoke" \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
            experiments/run_sit_ig_direction_specificity.py \
            --output-dir "$output_dir" \
            --samples 2 \
            --num-steps 50 \
            --pulse-steps 5,15,25,35,45,49 \
            --gamma 0.01 \
            --probe-count 2 \
            --per-rank-batch 2 \
            --condition-group-size 8 \
            --seed 20260814 \
            --label-seed 20260814 \
            --bootstrap-repeats 100 \
            --log-every-samples 2 2>&1 | tee -a "$log"
    done
}

run_or_resume_seed() {
    local seed="$1"
    local gpu="$2"
    local output_dir="$OUTPUT_ROOT/n64_seed${seed}_probes4_steps50_v1"
    local log="$output_dir"_console.log
    mkdir -p "$output_dir"
    while ! is_complete "$output_dir"; do
        if pgrep -f "run_sit_ig_direction_specificity.py --output-dir $output_dir" >/dev/null; then
            sleep 60
            continue
        fi
        cd "$ROOT"
        CUDA_VISIBLE_DEVICES="$gpu" \
        MPLCONFIGDIR="/tmp/mpl-sit-ig-specificity-${seed}" \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
            experiments/run_sit_ig_direction_specificity.py \
            --output-dir "$output_dir" \
            --samples 64 \
            --num-steps 50 \
            --pulse-steps 5,15,25,35,45,49 \
            --gamma 0.01 \
            --probe-count 4 \
            --per-rank-batch 2 \
            --condition-group-size 8 \
            --seed "$seed" \
            --label-seed "$seed" \
            --bootstrap-repeats 5000 \
            --log-every-samples 4 2>&1 | tee -a "$log"
    done
}

run_smoke
run_or_resume_seed 20260814 0 &
pid0=$!
run_or_resume_seed 20260815 1 &
pid1=$!
wait "$pid0" "$pid1"

"$PYTHON" experiments/summarize_sit_ig_direction_specificity.py \
    --run-dirs \
        "$OUTPUT_ROOT/n64_seed20260814_probes4_steps50_v1" \
        "$OUTPUT_ROOT/n64_seed20260815_probes4_steps50_v1" \
    --output-dir "$OUTPUT_ROOT/n64_seeds14_15_pooled_v1" \
    --bootstrap-repeats 5000 \
    --seed 20260816
