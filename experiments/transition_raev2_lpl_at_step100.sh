#!/usr/bin/env bash
set -euo pipefail

repo=/home/zhoushunyu/eqvae
data_root=/home/zhoushunyu/data/eqvae
old_session=raev2_lpl_cycle_to800
new_session=raev2_lpl_cycle_to1000
watcher_session=raev2_lpl_transition_to1000
checkpoint="$data_root/experiments/raev2_lpl_pilot/lpl_official_800_strict_from10/checkpoints/branch-0000100-global-0100180.pt"
transition_log="$data_root/experiments/raev2_lpl_pilot/lpl_official_800_strict_from10/transition_to1000.log"

if tmux has-session -t "$new_session" 2>/dev/null; then
  echo "new long-running session already exists: $new_session"
  exit 0
fi
if tmux has-session -t "$watcher_session" 2>/dev/null; then
  echo "transition watcher already exists: $watcher_session"
  exit 0
fi
if ! tmux has-session -t "$old_session" 2>/dev/null; then
  echo "old training session does not exist: $old_session" >&2
  exit 1
fi

orchestrator_pattern="^(/home/zhoushunyu/data|/data/users/zhoushunyu)/eqvae/envs/raev2/bin/python experiments/run_raev2_lpl_checkpoint_cycle.py --target-step 800 "
orchestrator_pid="$(pgrep -f "$orchestrator_pattern" | head -n 1)"
if [[ -z "$orchestrator_pid" ]]; then
  echo "could not locate the step-800 orchestrator" >&2
  exit 1
fi

# The child torchrun writes directly to its log and continues while the parent
# orchestrator is stopped. This prevents the old schedule from starting the
# next sample/train command after the step-100 checkpoint is complete.
kill -STOP "$orchestrator_pid"
printf '%s paused orchestrator pid=%s\n' "$(date --iso-8601=seconds)" "$orchestrator_pid" >> "$transition_log"

tmux new-session -d -s "$watcher_session" \
  "bash -lc 'set -euo pipefail; while [[ ! -f \"$checkpoint\" ]] || pgrep -f \"train_raev2_strict_lpl.p[y].*--max-updates 100\" >/dev/null; do sleep 5; done; printf \"%s step100 checkpoint complete\\n\" \"\$(date --iso-8601=seconds)\" >> \"$transition_log\"; tmux kill-session -t \"$old_session\"; sleep 5; cd \"$repo\"; bash experiments/launch_raev2_lpl_checkpoint_cycle.sh >> \"$transition_log\" 2>&1'"
tmux set-option -t "$watcher_session" remain-on-exit on

echo "paused orchestrator: $orchestrator_pid"
echo "watcher: $watcher_session"
echo "checkpoint: $checkpoint"
echo "transition log: $transition_log"
