#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zhoushunyu/eqvae"
PYTHON="/home/zhoushunyu/data/eqvae/envs/raev2/bin/python"
ADM_PYTHON="/data/shared/envs/adm-fid/bin/python"
OUTPUT_ROOT="/home/zhoushunyu/data/eqvae/experiments/sit_ig_interval_ablation"
DIRECTION_ROOT="/home/zhoushunyu/data/eqvae/experiments/sit_ig_direction_specificity"
RAE_ROOT="/home/zhoushunyu/data/eqvae/experiments/raev2_ig_impulse_response"
PIPELINE_MANIFEST="$OUTPUT_ROOT/pipeline_manifest.json"
LOCK_FILE="$OUTPUT_ROOT/pipeline.lock"
CURRENT_STAGE="bootstrap"

mkdir -p "$OUTPUT_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[$(date -Is)] another interval-ablation pipeline owns $LOCK_FILE"
    exit 1
fi

log() {
    echo "[$(date -Is)] [$CURRENT_STAGE] $*"
}

write_pipeline_status() {
    local status="$1"
    local exit_code="$2"
    "$PYTHON" -c \
        'import json,os,pathlib,sys,time; p=pathlib.Path(sys.argv[1]); payload={"protocol":"sit_ig_interval_pipeline_v1","status":sys.argv[2],"stage":sys.argv[3],"exit_code":int(sys.argv[4]),"updated_unix":time.time(),"formal_run":sys.argv[5],"decoded_run":sys.argv[6],"feature_run":sys.argv[7],"metric_run":sys.argv[8]}; t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(payload,indent=2,ensure_ascii=False)); os.replace(t,p)' \
        "$PIPELINE_MANIFEST" "$status" "$CURRENT_STAGE" "$exit_code" \
        "$OUTPUT_ROOT/n5000_steps250_seed20260816_v1" \
        "$OUTPUT_ROOT/n5000_steps250_seed20260816_v1_decoded" \
        "$OUTPUT_ROOT/n5000_steps250_seed20260816_v1_adm_features" \
        "$OUTPUT_ROOT/n5000_steps250_seed20260816_v1_adm_fid"
}

enter_stage() {
    CURRENT_STAGE="$1"
    write_pipeline_status "$2" 0
    log "entered stage"
}

on_error() {
    local exit_code=$?
    log "pipeline failed with exit code $exit_code"
    write_pipeline_status "failed" "$exit_code" || true
    exit "$exit_code"
}
trap on_error ERR

is_complete() {
    local manifest="$1/manifest.json"
    [[ -f "$manifest" ]] && "$PYTHON" -c \
        'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("status") != "complete")' \
        "$manifest"
}

retry() {
    local maximum="$1"
    local delay="$2"
    shift 2
    local attempt=1
    until "$@"; do
        if (( attempt >= maximum )); then
            log "command failed after $attempt attempts: $*"
            return 1
        fi
        log "attempt $attempt failed; retrying in ${delay}s: $*"
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

wait_for_gpu_memory() {
    local minimum_mib="$1"
    while ! nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | \
        awk -v minimum="$minimum_mib" '{gsub(/ /, "", $1); if ($1 < minimum) exit 1}'; do
        log "waiting until every GPU has at least ${minimum_mib} MiB free"
        sleep 60
    done
}

cd "$ROOT"

enter_stage "wait_prerequisites" "waiting"
log "waiting for direction summary and both RAEv2 N=256 runs"
while ! is_complete "$DIRECTION_ROOT/n64_seeds14_15_pooled_v1" || \
      ! is_complete "$RAE_ROOT/n256_seed20260812_pulse_g001_g005_v1" || \
      ! is_complete "$RAE_ROOT/n256_seed20260813_pulse_g001_g005_v1"; do
    sleep 60
done
log "all prerequisites complete"

enter_stage "generation_smoke" "running"
SMOKE="$OUTPUT_ROOT/smoke_n4_steps20_seed20260816_v1"
if ! is_complete "$SMOKE"; then
    wait_for_gpu_memory 8000
    retry 3 60 env \
        CUDA_VISIBLE_DEVICES=0 \
        MPLCONFIGDIR=/tmp/mpl-sit-ig-interval-smoke \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
        experiments/run_sit_ig_interval_ablation.py \
        --output-dir "$SMOKE" \
        --samples 4 \
        --num-steps 20 \
        --per-rank-batch 2 \
        --seed 20260816 \
        --label-mode sequential \
        --log-every-samples 2
fi

enter_stage "decode_smoke" "running"
SMOKE_DECODE="$OUTPUT_ROOT/smoke_n4_steps20_seed20260816_v1_decoded"
if ! is_complete "$SMOKE_DECODE"; then
    retry 3 60 env \
        CUDA_VISIBLE_DEVICES=0 \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
        experiments/decode_sit_ig_interval_ablation.py \
        --run-dir "$SMOKE" \
        --output-dir "$SMOKE_DECODE" \
        --batch-size 2 \
        --log-every 2
fi

enter_stage "feature_smoke" "running"
REFERENCE_STATS="$OUTPUT_ROOT/imagenet256_adm_reference_stats.npz"
SMOKE_FEATURES="$OUTPUT_ROOT/smoke_n4_steps20_seed20260816_v1_adm_features"
if ! is_complete "$SMOKE_FEATURES"; then
    retry 3 60 env \
        CUDA_VISIBLE_DEVICES= \
        "$ADM_PYTHON" experiments/evaluate_sit_ig_interval_ablation.py \
        --decode-dir "$SMOKE_DECODE" \
        --reference /data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz \
        --reference-stats-cache "$REFERENCE_STATS" \
        --output-dir "$SMOKE_FEATURES" \
        --batch-size 4
fi

enter_stage "fid_smoke" "running"
SMOKE_METRICS="$OUTPUT_ROOT/smoke_n4_steps20_seed20260816_v1_adm_fid"
if ! is_complete "$SMOKE_METRICS"; then
    retry 3 60 env \
        OMP_NUM_THREADS=16 \
        MKL_NUM_THREADS=16 \
        "$PYTHON" experiments/compute_sit_ig_interval_fid.py \
        --feature-dir "$SMOKE_FEATURES" \
        --output-dir "$SMOKE_METRICS" \
        --device cpu \
        --num-threads 16
fi
log "all smoke stages passed"

enter_stage "formal_generation" "running"
FORMAL="$OUTPUT_ROOT/n5000_steps250_seed20260816_v1"
if ! is_complete "$FORMAL"; then
    wait_for_gpu_memory 12000
    retry 3 300 env \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        MPLCONFIGDIR=/tmp/mpl-sit-ig-interval-5k \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
        experiments/run_sit_ig_interval_ablation.py \
        --output-dir "$FORMAL" \
        --samples 5000 \
        --num-steps 250 \
        --per-rank-batch 2 \
        --seed 20260816 \
        --label-mode sequential \
        --log-every-samples 20
fi

enter_stage "formal_decode" "running"
DECODE="$OUTPUT_ROOT/n5000_steps250_seed20260816_v1_decoded"
if ! is_complete "$DECODE"; then
    wait_for_gpu_memory 4000
    retry 3 300 env \
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
        experiments/decode_sit_ig_interval_ablation.py \
        --run-dir "$FORMAL" \
        --output-dir "$DECODE" \
        --batch-size 8 \
        --log-every 64
fi

enter_stage "formal_features" "running"
FEATURES="$OUTPUT_ROOT/n5000_steps250_seed20260816_v1_adm_features"
if ! is_complete "$FEATURES"; then
    retry 3 300 env \
        CUDA_VISIBLE_DEVICES= \
        "$ADM_PYTHON" experiments/evaluate_sit_ig_interval_ablation.py \
        --decode-dir "$DECODE" \
        --reference /data/shared/adm_refs/VIRTUAL_imagenet256_labeled.npz \
        --reference-stats-cache "$REFERENCE_STATS" \
        --output-dir "$FEATURES" \
        --batch-size 64
fi

enter_stage "formal_fid" "running"
METRICS="$OUTPUT_ROOT/n5000_steps250_seed20260816_v1_adm_fid"
if ! is_complete "$METRICS"; then
    retry 3 300 env \
        OMP_NUM_THREADS=16 \
        MKL_NUM_THREADS=16 \
        "$PYTHON" experiments/compute_sit_ig_interval_fid.py \
        --feature-dir "$FEATURES" \
        --output-dir "$METRICS" \
        --device cpu \
        --num-threads 16
fi

enter_stage "plot" "running"
retry 3 60 "$PYTHON" experiments/plot_sit_ig_interval_ablation.py \
    --metrics "$METRICS/interval_metrics.csv" \
    --output "$METRICS/interval_fid.png"

CURRENT_STAGE="complete"
trap - ERR
write_pipeline_status "complete" 0
log "pipeline complete"
