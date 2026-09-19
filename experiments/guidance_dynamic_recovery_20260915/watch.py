"""Read-only 15-minute health sampling during this queue's fault recovery."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import time
from . import k, AMENDMENT, verify


def main():
    verify()
    root = k.ROOT / 'repair_20260915' / 'watch'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    k.atomic(root / 'launch.json', dict(pid=os.getpid(), interval_seconds=900,
        runtime_amendment_sha256=k.sha(AMENDMENT)))
    previous = None
    stable = 0
    while True:
        state = k.read(k.ROOT / 'status.json')
        alive = Path(f"/proc/{state['pid']}/cmdline").exists()
        if alive:
            alive = b'guidance_dynamic_recovery_20260915.pipeline' in Path(f"/proc/{state['pid']}/cmdline").read_bytes()
        model = state.get('current_model') or 'sit_small'
        idea = state.get('current_idea') or 'real'
        progress = k.model_root(model) / 'training' / idea / 'progress.json'
        value = k.read(progress) if progress.exists() else {}
        failed = (k.ROOT / 'failure.json').exists()
        check = k.model_root(model) / 'checks_runtime2/complete.json'
        check_passed = check.exists() and k.read(check).get('passed', False)
        pairs = state.get('complete_model_idea_pairs', 0)
        completed_jobs = sum(row['state'] == 'complete' for row in k.read(k.ROOT / 'jobs.json').values())
        token = (pairs, value.get('step', 0), model, idea, completed_jobs,
                 state.get('current_phase'), state.get('current_step'))
        active = bool(state.get('active_jobs'))
        advancing = previous is not None and token != previous
        healthy = alive and not failed and state['phase'] not in ('paused', 'draining_failure')
        stable = stable + 1 if healthy and check_passed and advancing else 0
        row = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            controller_alive=alive, phase=state['phase'], check_passed=check_passed,
            failure_present=failed, training_step=value.get('step', 0), current_model=state.get('current_model'),
            current_idea=state.get('current_idea'), current_phase=state.get('current_phase'),
            complete_model_idea_pairs=pairs, active_jobs=state.get('active_jobs', []),
            ready_jobs=state.get('ready_jobs', []), stable_intervals=stable,
            needs_attention=failed or (not alive and state['phase'] != 'complete'),
            progress_since_last_poll=advancing,
            note='Read-only health log; no automatic code changes or retries of failed jobs.')
        k.atomic(root / 'status.json', row)
        with (root / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if state['phase'] == 'complete' or stable >= 2:
            k.atomic(root / 'complete.json', dict(complete=True, reason='queue_complete' if
                state['phase'] == 'complete' else 'two_advancing_healthy_intervals', last=row))
            return
        if failed or (not alive and state['phase'] != 'complete'):
            k.atomic(root / 'needs_attention.json', row)
        previous = token
        # Short sleeps allow ordinary signals to stop this read-only watcher.
        for _ in range(30):
            if (root / 'STOP').exists():
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
