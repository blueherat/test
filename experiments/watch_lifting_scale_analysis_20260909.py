"""Refresh the read-only curve analysis as the already-running sweep completes.

This observer cannot launch, resume, stop, or alter a sampling configuration.
It exits on a terminal queue state or the loss of the specified controller.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')

from experiments.lifting_scale_sweep_20260909 import ROOT, WORK, read, sha, atomic


def main(pid):
    expected_request = '6563f5d43b3c52f1ebded624076135d583cf9e932f54e4c224d9e642412a05db'
    last = None
    while True:
        assert sha(ROOT / 'request.json') == expected_request
        status = read(ROOT / 'status.json')
        proc = Path(f'/proc/{pid}/cmdline')
        try:
            command = proc.read_bytes()
        except FileNotFoundError:
            command = b''
        live = ((b'experiments.run_lifting_scale_sweep_20260909' in command and b'--run-prepared' in command) or
                (b'experiments.recover_lifting_scale_sweep_20260909' in command and b'--resume-adopted' in command))
        results = ROOT / 'results.json'
        signature = (sha(results) if results.exists() else None,
                     status['phase'] if status['phase'] in ('complete', 'failed') else 'active')
        if signature != last:
            cmd = [sys.executable, '-m', 'experiments.analyze_lifting_scale_sweep_20260909']
            if status['phase'] == 'complete':
                cmd.append('--require-complete')
            subprocess.run(cmd, cwd=WORK, check=True)
            last = signature
        terminal = status['phase'] in ('complete', 'failed')
        state = dict(controller_pid=pid, controller_live=live, sampling_status=status,
                     phase='complete' if status['phase'] == 'complete' else
                           ('stopped' if terminal or not live else 'watching'),
                     request_sha256=expected_request, research_goal_achieved=False)
        atomic(ROOT / 'analysis_observer.json', state)
        if terminal or not live:
            print(state, flush=True)
            if status['phase'] != 'complete':
                raise RuntimeError('Sweep did not complete; existing results retained; no automatic restart.')
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller-pid', type=int, required=True)
    main(parser.parse_args().controller_pid)
