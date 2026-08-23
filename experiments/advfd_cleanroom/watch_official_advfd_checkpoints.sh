#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

CHECKPOINT_DIR="${CHECKPOINT_DIR:?Set CHECKPOINT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
REFERENCE_STATS="${REFERENCE_STATS:-/data/users/zhoushunyu/research_deps/advfd_reference_stats/guided_diffusion_stats.npz}"
FINAL_CHECKPOINT="${FINAL_CHECKPOINT:-${CHECKPOINT_DIR}/step_0009999.pth}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "${OUTPUT_DIR}/per_checkpoint"

while true; do
  shopt -s nullglob
  checkpoints=("${CHECKPOINT_DIR}"/step_*.pth)
  shopt -u nullglob

  for checkpoint in "${checkpoints[@]}"; do
    name="$(basename "${checkpoint}" .pth)"
    marker="${OUTPUT_DIR}/per_checkpoint/${name}.json"
    if [[ -f "${marker}" ]]; then
      continue
    fi
    temp_dir="${OUTPUT_DIR}/per_checkpoint/${name}.tmp"
    rm -rf "${temp_dir}"
    python experiments/advfd_cleanroom/diagnose_official_advfd_checkpoints.py \
      --reference-stats "${REFERENCE_STATS}" \
      --output-dir "${temp_dir}" \
      "${checkpoint}" >/dev/null
    mv "${temp_dir}/checkpoint_diagnostics.json" "${marker}"
    rm -rf "${temp_dir}"
    echo "Recorded ${checkpoint}"
  done

  python - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "per_checkpoint").glob("step_*.json")):
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
rows.sort(key=lambda row: int(row["filename_step"]))
if rows:
    fields = sorted({key for row in rows for key in row})
    with (root / "checkpoint_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "checkpoint_diagnostics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
PY

  if [[ -f "${FINAL_CHECKPOINT}" && -f "${OUTPUT_DIR}/per_checkpoint/step_0009999.json" ]]; then
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
