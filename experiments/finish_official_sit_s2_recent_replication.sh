#!/usr/bin/env bash
set -euo pipefail

cd /home/zhoushunyu/eqvae

INPUT_ROOT="${INPUT_ROOT:-/home/zhoushunyu/data/eqvae/imagenet_sit_flow/official_sit_s2_recent_replication_v1}"
STATE_PATH="${INPUT_ROOT}/pipeline_state.json"
SNAPSHOT_PATH="${INPUT_ROOT}/git_input_sha256.txt"
POLL_SECONDS="${POLL_SECONDS:-60}"

CODE_PATHS=(
  experiments/finish_official_sit_s2_recent_replication.sh
  experiments/launch_official_sit_s2_recent_replication.sh
  experiments/official_imagenet100_sit_s2.py
  experiments/package_official_sit_s2_recent_replication.py
  experiments/prepare_official_imagenet100_sit_s2.py
  experiments/run_imagenet100_sit_fid_curve.py
  experiments/run_official_sit_s2_recent_replication.py
  experiments/train_imagenet100_sit_frozen_internal_v_head.py
  experiments/train_imagenet100_sit_frozen_v_clean_head.py
  tests/test_imagenet100_sit_fid_curve.py
  tests/test_official_imagenet100_sit_s2.py
)

sha256sum "${CODE_PATHS[@]}" > "${SNAPSHOT_PATH}"

while true; do
  status="$({ python - "${STATE_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("waiting")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload.get("status") or "running")
PY
  } 2>/dev/null)"
  if [[ "${status}" == "complete" ]]; then
    break
  fi
  if ! tmux has-session -t official_sit_s2_repro 2>/dev/null; then
    echo "replication stopped before completion; state=${status}" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

sha256sum --check "${SNAPSHOT_PATH}"
python experiments/package_official_sit_s2_recent_replication.py --input-root "${INPUT_ROOT}"

PYTHONPATH=. pytest -q \
  tests/test_imagenet100_sit_fid_curve.py \
  tests/test_imagenet100_sit_internal_v_head.py \
  tests/test_imagenet100_sit_hidden_state_extrapolation.py \
  tests/test_imagenet100_sit_vx_dual_head.py \
  tests/test_official_imagenet100_sit_s2.py
git diff --check

if ! git diff --cached --quiet; then
  echo "Git index is not clean; refusing automatic commit" >&2
  exit 1
fi

git add -- \
  "${CODE_PATHS[@]}" \
  docs/IMAGENET100_OFFICIAL_SIT_S2_REPLICATION_ZH.md \
  docs/data/imagenet100_official_sit_s2_recent_replication

git commit -m "复现官方SiT-S/2近期冻结头实验"
