"""Pause existing trainers through their checkpoint-preserving stop marker."""
import argparse
import json
from pathlib import Path
import time


def pause(run, target, receipt):
    from experiments.adversarial_weak_training_20260915 import common as c
    c.atomic(receipt, dict(run=str(run), target=target, phase='waiting', utc=c.now()))
    while not (run/'exit.json').exists():
        row = json.loads((run/'progress.json').read_text())
        if row['step'] >= target-1:
            # The preceding update has been logged. Let the next update enter
            # its loop, then request a safe stop before any subsequent update.
            # A delayed watcher may save later; never discard completed work.
            if row['step'] == target-1:
                time.sleep(1.)
            (run/'STOP_AFTER_CURRENT').write_text(
                f'User requested checkpoint evaluation; intended boundary {target}.\n')
            break
        time.sleep(.2)
    while not (run/'exit.json').exists():
        time.sleep(1.)
    status = c.read(run/'exit.json')
    latest = c.read(run/'latest.json')
    if status['exit_code'] != 0 or latest['phase'] != 'paused':
        raise RuntimeError(dict(exit=status, latest=latest))
    if latest['step'] < target or c.sha(latest['checkpoint']) != latest['sha256']:
        raise RuntimeError('Pause checkpoint missing, corrupt, or earlier than requested')
    c.atomic(receipt, dict(run=str(run), target=target, phase='paused',
                          step=latest['step'], checkpoint=latest['checkpoint'],
                          sha256=latest['sha256'], utc=c.now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--target', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    pause(args.run, args.target, args.receipt)
