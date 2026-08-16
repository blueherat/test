#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/response_direction_dense_n1000_seed0}"
SAMPLES="${SAMPLES:-1000}"
SEED="${SEED:-0}"
GPU_LIST="${GPU_LIST:-3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
V500_GAMMAS="${V500_GAMMAS:-1.25 1.5 1.75}"
V500_RHOS="${V500_RHOS:-1.20 1.25 1.30 1.35 1.40 1.45 1.50}"
X800_GAMMAS="${X800_GAMMAS:-0.75 1.0 1.25}"
X800_RHOS="${X800_RHOS:-1.15 1.20 1.25 1.30 1.35 1.40 1.45}"
V500_LAMBDAS="${V500_LAMBDAS:-0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0 1.125 1.25}"
X800_LAMBDAS="${X800_LAMBDAS:-0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0}"
RUN_X800="${RUN_X800:-0}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
V500="$BASE/runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
X800="$BASE/runs/sit-s-2_x-velocity-loss-floor0p05_seed0/checkpoints/step_00800000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$V500" "$X800" "$REFERENCE"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

float_tag() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "$value"
}

wait_for_gpu() {
  local gpu="$1" free_mib
  while true; do
    free_mib="$(
      nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
        | tr -d '[:space:]'
    )"
    if (( free_mib >= MIN_FREE_MIB )); then
      return
    fi
    echo "[$(date --iso-8601=seconds)] waiting for GPU $gpu" >&2
    sleep 30
  done
}

run_job() {
  local gpu="$1" family="$2" gamma="$3" rho="$4" lambda="$5"
  local other tag output log attempt
  local -a mismatch_args=()
  if [[ "$family" == "v500" ]]; then
    other="$V500"
    mismatch_args=(--allow-step-mismatch)
  else
    other="$X800"
  fi

  tag="g$(float_tag "$gamma")_r$(float_tag "$rho")_l$(float_tag "$lambda")_n${SAMPLES}_seed${SEED}"
  output="$ROOT/$family/$tag"
  log="$LOG_DIR/${family}_${tag}.log"
  if [[ -f "$output/nominal_intervention_fid5k.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reusing $family/$tag"
    return
  fi

  for attempt in $(seq 1 20); do
    wait_for_gpu "$gpu"
    echo "[$(date --iso-8601=seconds)] starting $family/$tag GPU=$gpu attempt=$attempt"
    if "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
      --anchor-checkpoint "$V800" \
      --other-checkpoint "$other" \
      "${mismatch_args[@]}" \
      --mode factorized \
      --gamma "$gamma" \
      --nominal-scale 1 \
      --orthogonal-scale "$lambda" \
      --response-scale "$rho" \
      --reference "$REFERENCE" \
      --num-samples "$SAMPLES" \
      --batch-size 8 \
      --vae-decode-batch-size 2 \
      --global-seed "$SEED" \
      --cuda-visible-devices "$gpu" \
      --cuda-allocator-limit-gib 4 \
      --fid-batch-size 8 \
      --fid-gpu-memory-fraction 0.10 \
      --gpu-memory-ceiling-mib 22528 \
      --output-dir "$output" \
      >"$log" 2>&1; then
      echo "[$(date --iso-8601=seconds)] completed $family/$tag"
      return
    fi
    if ! grep -q "gpu_memory_ceiling_reached" "$log"; then
      echo "[$(date --iso-8601=seconds)] non-resource failure in $family/$tag" >&2
      tail -50 "$log" >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] resource retry for $family/$tag" >&2
    sleep 30
  done
  return 1
}

worker() {
  local family="$1" gpu="$2" stride="$3" offset="$4"
  local gammas rhos lambdas gamma rho lambda index=0
  if [[ "$family" == "v500" ]]; then
    gammas="$V500_GAMMAS"
    rhos="$V500_RHOS"
    lambdas="$V500_LAMBDAS"
  else
    gammas="$X800_GAMMAS"
    rhos="$X800_RHOS"
    lambdas="$X800_LAMBDAS"
  fi
  for gamma in $gammas; do
    for rho in $rhos; do
      for lambda in $lambdas; do
        if (( index % stride == offset )); then
          run_job "$gpu" "$family" "$gamma" "$rho" "$lambda"
        fi
        index=$((index + 1))
      done
    done
  done
}

run_family() {
  local family="$1" status=0 gpu pid
  local -a gpus=() pids=()
  read -r -a gpus <<<"$GPU_LIST"
  ((${#gpus[@]} > 0)) || { echo "GPU_LIST is empty" >&2; return 2; }

  for offset in "${!gpus[@]}"; do
    gpu="${gpus[$offset]}"
    worker "$family" "$gpu" "${#gpus[@]}" "$offset" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  return "$status"
}

grid_size() {
  local gammas="$1" rhos="$2" lambdas="$3"
  local ng nr nl
  ng="$(wc -w <<<"$gammas")"
  nr="$(wc -w <<<"$rhos")"
  nl="$(wc -w <<<"$lambdas")"
  printf '%d' "$((ng * nr * nl))"
}

echo "v500 conditions: $(grid_size "$V500_GAMMAS" "$V500_RHOS" "$V500_LAMBDAS")"
echo "x800 conditions: $(grid_size "$X800_GAMMAS" "$X800_RHOS" "$X800_LAMBDAS")"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

# Finish the primary v500 grid before spending compute on x800.
run_family v500
touch "$ROOT/V500_COMPLETE"

if [[ "$RUN_X800" == "1" ]]; then
  run_family x800
  touch "$ROOT/X800_COMPLETE"
else
  echo "[$(date --iso-8601=seconds)] skipping x800 grid (RUN_X800=$RUN_X800)"
fi

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" \
  >"$LOG_DIR/summary.log" 2>&1
touch "$ROOT/DENSE_SWEEP_COMPLETE"
echo "[$(date --iso-8601=seconds)] response-direction dense sweep complete"
