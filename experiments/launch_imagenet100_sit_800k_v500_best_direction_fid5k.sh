#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"
BASE="${BASE:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow}"
ROOT="${ROOT:-$BASE/factorized_guidance_800k_v1/v500_best_direction_formal_n5000}"
CONTROL_ROOT="${CONTROL_ROOT:-$BASE/factorized_guidance_800k_v1/v500_formal_n5000}"
GAMMA="${GAMMA:-1.75}"
RHO="${RHO:-1.25}"
LAMBDA="${LAMBDA:-1.125}"
CONDITION_TAG="${CONDITION_TAG:-best_g1p75_r1p25_l1p125_n5000}"
V800="$BASE/runs/sit-s-2_seed0/checkpoints/step_00800000.pt"
V500="$BASE/runs/sit-s-2_seed0/checkpoints/step_00500000.pt"
REFERENCE="$BASE/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz"
LOG_DIR="$ROOT/logs"

mkdir -p "$LOG_DIR"
for path in "$PYTHON_BIN" "$V800" "$V500" "$REFERENCE"; do
  [[ -e "$path" ]] || { echo "Missing required asset: $path" >&2; exit 2; }
done

run_seed() {
  local seed="$1" gpu="$2" batch_size="$3"
  local tag="${CONDITION_TAG}_seed${seed}"
  local output="$ROOT/v500/$tag"
  local log="$LOG_DIR/${tag}.log"

  if [[ -f "$output/nominal_intervention_fid5k.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reusing $tag"
    return
  fi

  echo "[$(date --iso-8601=seconds)] starting $tag on GPU $gpu (batch=$batch_size)"
  "$PYTHON_BIN" experiments/run_imagenet100_sit_nominal_intervention_fid5k.py \
    --anchor-checkpoint "$V800" \
    --other-checkpoint "$V500" \
    --allow-step-mismatch \
    --mode factorized \
    --gamma "$GAMMA" \
    --nominal-scale 1 \
    --orthogonal-scale "$LAMBDA" \
    --response-scale "$RHO" \
    --reference "$REFERENCE" \
    --num-samples 5000 \
    --batch-size "$batch_size" \
    --vae-decode-batch-size 2 \
    --global-seed "$seed" \
    --cuda-visible-devices "$gpu" \
    --cuda-allocator-limit-gib 4 \
    --fid-batch-size 8 \
    --fid-gpu-memory-fraction 0.10 \
    --gpu-memory-ceiling-mib 22528 \
    --output-dir "$output" \
    >"$log" 2>&1
  echo "[$(date --iso-8601=seconds)] completed $tag on GPU $gpu"
}

run_seed 0 0 8 & pid0=$!
run_seed 1 1 2 & pid1=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
[[ "$status" == "0" ]] || exit "$status"

"$PYTHON_BIN" experiments/summarize_imagenet100_sit_factorized_guidance_screen.py \
  --root "$ROOT" \
  >"$LOG_DIR/summary.log" 2>&1

ROOT="$ROOT" CONTROL_ROOT="$CONTROL_ROOT" CONDITION_TAG="$CONDITION_TAG" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
control_root = Path(os.environ["CONTROL_ROOT"])
condition_tag = os.environ["CONDITION_TAG"]
for seed in (0, 1):
    candidate = json.loads(
        (root / "v500" / f"{condition_tag}_seed{seed}" / "nominal_intervention_fid5k.json").read_text()
    )
    control = json.loads(
        (control_root / "v500" / f"closed_g3_n5000_seed{seed}" / "nominal_intervention_fid5k.json").read_text()
    )
    candidate_pair = (candidate["noise_fingerprint"], candidate["label_fingerprint"])
    control_pair = (control["noise_fingerprint"], control["label_fingerprint"])
    if candidate_pair != control_pair:
        raise RuntimeError(f"seed {seed} is not paired with the formal controls")
    print(f"seed={seed} candidate_fid={candidate['fid']:.6f} closed_fid={control['fid']:.6f}")
PY

touch "$ROOT/FORMAL_COMPLETE"
echo "[$(date --iso-8601=seconds)] best response-direction FID-5K complete"
