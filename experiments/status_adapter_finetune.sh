#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

RUN_NAME=${RUN_NAME:-finetune_ditdh_s_adapter_from_official_ep5_lr2e5_4gpu}
RESULTS_DIR=${RESULTS_DIR:-$EQVAE_STAGE2_TRAINING}
SAMPLE_ROOT=${SAMPLE_ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR=${LOG_DIR:-$SAMPLE_ROOT/logs}
RUN_DIR="$RESULTS_DIR/$RUN_NAME"

echo "== tmux =="
tmux ls 2>/dev/null || true

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

echo
echo "== latest train steps =="
grep -E "\\| Step [0-9]+\\]" "$LOG_DIR/${RUN_NAME}_train.log" 2>/dev/null | tail -n 10 || true

echo
echo "== checkpoints =="
find "$RUN_DIR/checkpoints" -maxdepth 1 -type f -printf '%f %s %TY-%Tm-%Td %TH:%TM\n' 2>/dev/null | sort | tail -n 20 || true

echo
echo "== watcher tail =="
tail -n 30 "$LOG_DIR/${RUN_NAME}_eval_watcher.log" 2>/dev/null || true

echo
echo "== 5k adm json =="
find "$SAMPLE_ROOT" -maxdepth 1 -type f \( \
  -name "${RUN_NAME}_step*_n5000_adm_fid.json" -o \
  -name "${RUN_NAME}_step*_n5000_adm_adm_fid.json" \
\) -print 2>/dev/null | sort | while read -r path; do
  python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
print(f"{path} fid={obj.get('fid')} sfid={obj.get('sfid')} is={obj.get('inception_score')}")
PY
done

echo
echo "== trend summary =="
python "$EQVAE_ROOT/experiments/summarize_adapter_finetune.py" \
  --run-name "$RUN_NAME" \
  --sample-root "$SAMPLE_ROOT" 2>/dev/null || true

echo
echo "== preview grids =="
find "$SAMPLE_ROOT" -maxdepth 1 -type f -name "${RUN_NAME}_step*_grid.png" -printf '%p %s\n' 2>/dev/null | sort || true

echo
echo "== final50k adm json =="
find "$SAMPLE_ROOT" -maxdepth 1 -type f \( \
  -name "${RUN_NAME}_step*_n50000_adm_fid.json" -o \
  -name "${RUN_NAME}_step*_n50000_adm_adm_fid.json" \
\) -print 2>/dev/null | sort | while read -r path; do
  python - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
print(f"{path} fid={obj.get('fid')} sfid={obj.get('sfid')} is={obj.get('inception_score')}")
PY
done
