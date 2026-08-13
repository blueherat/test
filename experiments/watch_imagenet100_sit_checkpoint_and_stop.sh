#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${RUN_DIR:?set RUN_DIR to the exact training run directory}"
TRAIN_PID="${TRAIN_PID:?set TRAIN_PID to the torchrun parent PID}"
TARGET_STEP="${TARGET_STEP:-800000}"
POLL_SECONDS="${POLL_SECONDS:-5}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhoushunyu/miniconda3/envs/myenv/bin/python}"

CHECKPOINT="$RUN_DIR/checkpoints/step_$(printf '%08d' "$TARGET_STEP").pt"
MARKER="$RUN_DIR/STOPPED_AT_STEP_$(printf '%08d' "$TARGET_STEP")"

expected_process() {
  [[ -r "/proc/$TRAIN_PID/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline" | grep -Fq -- "--output-dir $RUN_DIR"
}

if ! expected_process; then
  echo "PID $TRAIN_PID is not the expected training process for $RUN_DIR" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] waiting for atomic checkpoint $CHECKPOINT"
while [[ ! -f "$CHECKPOINT" ]]; do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "training process $TRAIN_PID exited before $CHECKPOINT appeared" >&2
    exit 3
  fi
  sleep "$POLL_SECONDS"
done

"$PYTHON_BIN" -c \
  'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); assert int(p["step"]) == int(sys.argv[2])' \
  "$CHECKPOINT" "$TARGET_STEP"

if ! expected_process; then
  echo "PID $TRAIN_PID changed identity before termination" >&2
  exit 4
fi

echo "[$(date --iso-8601=seconds)] checkpoint verified; sending SIGTERM to torchrun PID $TRAIN_PID"
kill -TERM "$TRAIN_PID"

for _ in $(seq 1 120); do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    printf 'step=%s\ncheckpoint=%s\nstopped_at=%s\n' \
      "$TARGET_STEP" "$CHECKPOINT" "$(date --iso-8601=seconds)" >"$MARKER.tmp"
    mv "$MARKER.tmp" "$MARKER"
    echo "[$(date --iso-8601=seconds)] training stopped after checkpoint $TARGET_STEP"
    exit 0
  fi
  sleep 1
done

echo "training process $TRAIN_PID did not exit within 120 seconds of SIGTERM" >&2
exit 5
