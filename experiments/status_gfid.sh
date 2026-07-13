#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_paths.sh"

OFFICIAL_FOLDER=${OFFICIAL_FOLDER:-official_ditdh_xl_ag_n50000_adm}
OFFICIAL_NOAG_FOLDER=${OFFICIAL_NOAG_FOLDER:-official_ditdh_xl_noag_n50000_adm}
OUR_FOLDER=${OUR_FOLDER:-ditdh_s_step20000_n50000_adm}
ADAPTER_RUN_NAME=${ADAPTER_RUN_NAME:-ditdh_s_dinov2_adapter_imagenet32768_s20000_1gpu}
ADAPTER_FOLDER=${ADAPTER_FOLDER:-${ADAPTER_RUN_NAME}_n50000_adm}
FAIR_ORIG_RUN=${FAIR_ORIG_RUN:-fair_ditdh_s_dinov2_original_gbs1024_epochwise_4gpu}
FAIR_ADAPTER_RUN=${FAIR_ADAPTER_RUN:-fair_ditdh_s_dinov2_adapter_gbs1024_epochwise_4gpu}
FAIR_ORIG_FOLDER=${FAIR_ORIG_FOLDER:-${FAIR_ORIG_RUN}_step0001251_n10000_adm}
FAIR_ADAPTER_FOLDER=${FAIR_ADAPTER_FOLDER:-${FAIR_ADAPTER_RUN}_step0001251_n10000_adm}
QTR_ORIG_RUN=${QTR_ORIG_RUN:-fair_ditdh_qtr56_dinov2_original_gbs1024_epochwise_4gpu}
QTR_ADAPTER_RUN=${QTR_ADAPTER_RUN:-fair_ditdh_qtr56_dinov2_adapter_gbs1024_epochwise_4gpu}
QTR_ORIG_FOLDER=${QTR_ORIG_FOLDER:-${QTR_ORIG_RUN}_step0005004_n10000_adm}
QTR_ADAPTER_FOLDER=${QTR_ADAPTER_FOLDER:-${QTR_ADAPTER_RUN}_step0005004_n10000_adm}
ROOT=${ROOT:-$EQVAE_STAGE2_SAMPLES}
LOG_DIR="$ROOT/logs"
TRAIN_ROOT=${TRAIN_ROOT:-$EQVAE_STAGE2_TRAINING}

count_pngs() {
  local dir="$1"
  if [ -d "$dir" ]; then
    find "$dir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l
  else
    echo 0
  fi
}

show_json() {
  local path="$1"
  if [ -s "$path" ]; then
    cat "$path"
  else
    echo "missing: $path"
  fi
}

echo "== time =="
date

echo
echo "== tmux =="
tmux ls 2>/dev/null || true

echo
echo "== processes =="
pgrep -af 'official_ditdh_xl_ag_n50000_adm|ditdh_s_step20000_n50000_adm|ditdh_s_dinov2_adapter|fair_ditdh_s_dinov2|fair_ditdh_qtr56|sample_ddp.py|compute_adm_fid.py|chain_after_official_gfid|monitor_official_gfid_sampling|run_adapter_latent_s_gfid|run_fair_ditdh_s_gfid|run_fair_qtr_epoch_compare' || true

echo
echo "== samples =="
echo "official_pngs $(count_pngs "$ROOT/$OFFICIAL_FOLDER")"
echo "official_noag_pngs $(count_pngs "$ROOT/$OFFICIAL_NOAG_FOLDER")"
echo "our_pngs $(count_pngs "$ROOT/$OUR_FOLDER")"
echo "adapter_pngs $(count_pngs "$ROOT/$ADAPTER_FOLDER")"
echo "fair_original_pngs $(count_pngs "$ROOT/$FAIR_ORIG_FOLDER")"
echo "fair_adapter_pngs $(count_pngs "$ROOT/$FAIR_ADAPTER_FOLDER")"
echo "qtr_original_pngs $(count_pngs "$ROOT/$QTR_ORIG_FOLDER")"
echo "qtr_adapter_pngs $(count_pngs "$ROOT/$QTR_ADAPTER_FOLDER")"
ls -lh "$ROOT/${OFFICIAL_FOLDER}.npz" "$ROOT/${OFFICIAL_NOAG_FOLDER}.npz" "$ROOT/${OUR_FOLDER}.npz" "$ROOT/${ADAPTER_FOLDER}.npz" "$ROOT/${FAIR_ORIG_FOLDER}.npz" "$ROOT/${FAIR_ADAPTER_FOLDER}.npz" 2>/dev/null || true
ls -lh "$ROOT/${QTR_ORIG_FOLDER}.npz" "$ROOT/${QTR_ADAPTER_FOLDER}.npz" 2>/dev/null || true

echo
echo "== checkpoints =="
ls -lh "$TRAIN_ROOT/$ADAPTER_RUN_NAME"/checkpoints/*.pt 2>/dev/null || true
ls -lh "$TRAIN_ROOT/$FAIR_ORIG_RUN"/checkpoints/*.pt 2>/dev/null || true
ls -lh "$TRAIN_ROOT/$FAIR_ADAPTER_RUN"/checkpoints/*.pt 2>/dev/null || true
ls -lh "$TRAIN_ROOT/$QTR_ORIG_RUN"/checkpoints/*.pt 2>/dev/null || true
ls -lh "$TRAIN_ROOT/$QTR_ADAPTER_RUN"/checkpoints/*.pt 2>/dev/null || true

echo
echo "== official log tail =="
tail -n 5 "$LOG_DIR/${OFFICIAL_FOLDER}.log" 2>/dev/null | sed -e 's/\r/\n/g' | tail -n 5 || true

echo
echo "== monitors =="
tail -n 8 "$LOG_DIR/${OFFICIAL_FOLDER}_monitor.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${OFFICIAL_FOLDER}_eval_watcher.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${OUR_FOLDER}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${OFFICIAL_NOAG_FOLDER}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${ADAPTER_RUN_NAME}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${ADAPTER_RUN_NAME}_train.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${FAIR_ORIG_RUN}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${FAIR_ORIG_RUN}_train.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${FAIR_ADAPTER_RUN}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${FAIR_ADAPTER_RUN}_train.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/fair_qtr4_after_s_epoch1.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/fair_qtr4_compare.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${QTR_ORIG_RUN}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${QTR_ORIG_RUN}_train.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${QTR_ADAPTER_RUN}_chain.log" 2>/dev/null || true
tail -n 8 "$LOG_DIR/${QTR_ADAPTER_RUN}_train.log" 2>/dev/null || true

echo
echo "== official ADM JSON =="
show_json "$ROOT/${OFFICIAL_FOLDER}_adm_fid.json"

echo
echo "== our ADM JSON =="
show_json "$ROOT/${OUR_FOLDER}_adm_fid.json"

echo
echo "== official no-AG ADM JSON =="
show_json "$ROOT/${OFFICIAL_NOAG_FOLDER}_adm_fid.json"

echo
echo "== adapter latent ADM JSON =="
show_json "$ROOT/${ADAPTER_FOLDER}_adm_fid.json"

echo
echo "== fair original ADM JSON =="
show_json "$ROOT/${FAIR_ORIG_FOLDER}_adm_fid.json"

echo
echo "== fair adapter ADM JSON =="
show_json "$ROOT/${FAIR_ADAPTER_FOLDER}_adm_fid.json"

echo
echo "== qtr original ADM JSON =="
show_json "$ROOT/${QTR_ORIG_FOLDER}_adm_fid.json"

echo
echo "== qtr adapter ADM JSON =="
show_json "$ROOT/${QTR_ADAPTER_FOLDER}_adm_fid.json"

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
