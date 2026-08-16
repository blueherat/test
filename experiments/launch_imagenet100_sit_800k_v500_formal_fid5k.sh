#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/v500_formal_n5000}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
V500="$BASE/runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
LOG_DIR="$ROOT/logs"
GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 3}"
GPU_MEMORY_CEILING_MIB="${GPU_MEMORY_CEILING_MIB:-22528}"
MIN_FREE_MIB="${MIN_FREE_MIB:-7000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$V500" "$REFERENCE"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

wait_for_any_gpu() {
  local gpu free_mib
  while true; do
    for gpu in $GPU_CANDIDATES; do
      free_mib="$(
        nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
          | tr -d '[:space:]'
      )"
      if (( free_mib >= MIN_FREE_MIB )); then
        printf '%s\n' "$gpu"
        return
      fi
    done
    echo "[$(date --iso-8601=seconds)] waiting for a free GPU in: $GPU_CANDIDATES" >&2
    sleep 30
  done
}

run_condition() {
  local seed="$1" condition="$2"
  local tag output gpu attempt log
  local -a args

  case "$condition" in
    closed)
      tag="closed_g3_n5000_seed${seed}"
      args=(--mode closed --gamma 3)
      ;;
    response)
      tag="response_g1p5_r1p35_n5000_seed${seed}"
      args=(
        --mode factorized
        --gamma 1.5
        --nominal-scale 1
        --orthogonal-scale 0
        --response-scale 1.35
      )
      ;;
    *)
      echo "Unknown condition: $condition" >&2
      return 2
      ;;
  esac

  output="$ROOT/v500/$tag"
  log="$LOG_DIR/${tag}.log"
  if [[ -f "$output/nominal_intervention_fid5k.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reusing completed $tag"
    return
  fi

  for attempt in $(seq 1 20); do
    gpu="$(wait_for_any_gpu)"
    echo "[$(date --iso-8601=seconds)] starting $tag on GPU $gpu (attempt $attempt)"
    if "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
      --anchor-checkpoint "$V800" \
      --other-checkpoint "$V500" \
      --allow-step-mismatch \
      "${args[@]}" \
      --reference "$REFERENCE" \
      --num-samples 5000 \
      --batch-size "$BATCH_SIZE" \
      --vae-decode-batch-size 2 \
      --global-seed "$seed" \
      --cuda-visible-devices "$gpu" \
      --cuda-allocator-limit-gib 4 \
      --fid-batch-size 8 \
      --fid-gpu-memory-fraction 0.10 \
      --gpu-memory-ceiling-mib "$GPU_MEMORY_CEILING_MIB" \
      --output-dir "$output" \
      >"$log" 2>&1; then
      echo "[$(date --iso-8601=seconds)] completed $tag on GPU $gpu"
      return
    fi
    if ! grep -q "gpu_memory_ceiling_reached" "$log"; then
      echo "[$(date --iso-8601=seconds)] non-resource failure in $tag" >&2
      tail -50 "$log" >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] GPU contention interrupted $tag; retrying" >&2
    sleep 30
  done
  echo "[$(date --iso-8601=seconds)] exhausted retries for $tag" >&2
  return 1
}

run_condition 0 closed
run_condition 0 response
run_condition 1 closed
run_condition 1 response

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" \
  >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/FORMAL_COMPLETE"
echo "[$(date --iso-8601=seconds)] v800-v500 formal FID-5K comparison complete"
