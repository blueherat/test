#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
ADM_PYTHON="${ADM_PYTHON:-/data/shared/envs/adm-fid/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/terminal_distribution_audit_800k_v1}"
REFERENCE="${REFERENCE:-$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
V500="$BASE/runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
SAMPLES="${SAMPLES:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
BATCH_PAUSE_SECONDS="${BATCH_PAUSE_SECONDS:-3}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-2}"
ALLOCATOR_GIB="${ALLOCATOR_GIB:-8}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
ATOL="${ATOL:-1e-7}"
RTOL="${RTOL:-1e-4}"
EQUIVALENCE_RMS_TOLERANCE="${EQUIVALENCE_RMS_TOLERANCE:-5e-3}"
CLOSED_ATOL="${CLOSED_ATOL:-1e-8}"
CLOSED_RTOL="${CLOSED_RTOL:-1e-5}"
BOOTSTRAP_REPS="${BOOTSTRAP_REPS:-8}"
C2ST_REPS="${C2ST_REPS:-3}"
SEEDS=(${SEEDS:-0 1})
GPUS=(${GPUS:-1 3})

BRANCHES=(
  baseline
  factorized_g1_r1p5
  factorized_g1p5_r1p35
  factorized_g2_r1p35
  factorized_g2p5_r1p35
  factorized_g3_r1
  closed_g3
)

if (( ${#SEEDS[@]} != ${#GPUS[@]} )); then
  echo "SEEDS and GPUS must contain the same number of entries" >&2
  exit 2
fi
for path in "$PYTHON_BIN" "$ADM_PYTHON" "$REFERENCE" "$V800" "$V500"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 3; }
done

mkdir -p "$ROOT/logs"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local seed_root="$ROOT/seed${seed}"
  local free_mib
  free_mib="$(
    nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
      | tr -d '[:space:]'
  )"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU $gpu has only ${free_mib} MiB free; require at least ${MIN_FREE_MIB} MiB" >&2
    return 4
  fi
  mkdir -p "$seed_root/adm_results" "$seed_root/adm_activations"
  echo "[$(date --iso-8601=seconds)] seed=$seed gpu=$gpu sampling"
  if [[ ! -f "$seed_root/SAMPLING_COMPLETE" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      experiments/run_imagenet100_sit_terminal_distribution_audit.py \
      --anchor-checkpoint "$V800" \
      --other-checkpoint "$V500" \
      --allow-step-mismatch \
      --num-samples "$SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --batch-pause-seconds "$BATCH_PAUSE_SECONDS" \
      --vae-decode-batch-size "$DECODE_BATCH_SIZE" \
      --global-seed "$seed" \
      --atol "$ATOL" \
      --rtol "$RTOL" \
      --closed-atol "$CLOSED_ATOL" \
      --closed-rtol "$CLOSED_RTOL" \
      --verify-first-batch-individual \
      --equivalence-rms-tolerance "$EQUIVALENCE_RMS_TOLERANCE" \
      --cuda-allocator-limit-gib "$ALLOCATOR_GIB" \
      --device cuda:0 \
      --output-dir "$seed_root"
  fi

  local branch
  for branch in "${BRANCHES[@]}"; do
    local result="$seed_root/adm_results/${branch}.json"
    local activations="$seed_root/adm_activations/${branch}.npz"
    local samples="$seed_root/samples_${branch}_n${SAMPLES}.npz"
    if [[ ! -f "$result" || ! -f "$activations" ]]; then
      echo "[$(date --iso-8601=seconds)] seed=$seed gpu=$gpu ADM $branch"
      CUDA_VISIBLE_DEVICES="$gpu" "$ADM_PYTHON" experiments/compute_adm_fid.py \
        --reference "$REFERENCE" \
        --samples "$samples" \
        --batch-size 8 \
        --gpu-memory-fraction 0.10 \
        --activations-output "$activations" \
        --output "$result"
    fi
  done

  echo "[$(date --iso-8601=seconds)] seed=$seed gpu=$gpu analysis"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
    experiments/analyze_imagenet100_sit_terminal_distribution_audit.py \
    --root "$seed_root" \
    --bootstrap-reps "$BOOTSTRAP_REPS" \
    --c2st-reps "$C2ST_REPS" \
    --seed "$((20260815 + seed))" \
    --device cuda:0
  touch "$seed_root/PIPELINE_COMPLETE"
  echo "[$(date --iso-8601=seconds)] seed=$seed complete"
}

pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$index]}"
  log="$ROOT/logs/seed${seed}.log"
  (run_seed "$seed" "$gpu") >"$log" 2>&1 &
  pids+=("$!")
  echo "Started seed=$seed on GPU=$gpu pid=${pids[-1]} log=$log"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one terminal-distribution audit job failed" >&2
  exit "$status"
fi
touch "$ROOT/PIPELINE_COMPLETE"
echo "[$(date --iso-8601=seconds)] terminal-distribution audit complete"
