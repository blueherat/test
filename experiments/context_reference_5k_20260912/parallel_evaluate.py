"""Finish independent arms concurrently while the original watcher awaits its current child."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
from experiments.guidance_pasted_20260912 import common as c, evaluate
from . import core as m


def one(arm):
    m.configure()
    m.verify('sit_small')
    root = m.ROOT / 'sit_small' / m.STAGE / arm
    while not all((root / f'rank{rank}/complete.json').exists() for rank in range(2)):
        status = c.read(m.ROOT / 'status.json')
        if status['phase'] == 'failed' or not Path('/proc', str(status['pid'])).exists():
            raise RuntimeError('Sampling controller stopped')
        time.sleep(5)
    assert m.collect('sit_small', arm)
    evaluate.evaluate('sit_small', m.STAGE, arm)


def main(watcher, current_child):
    watcher_path = Path('/proc', str(watcher))
    cmd = (watcher_path / 'cmdline').read_bytes()
    assert b'experiments.context_reference_5k_20260912.run' in cmd and b'--evaluate' in cmd
    child_cmd = Path('/proc', str(current_child), 'cmdline').read_bytes()
    assert b'compute_adm_fid.py' in child_cmd and b'/context_base/' in child_cmd
    jobs = []
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    os.kill(watcher, signal.SIGSTOP)
    try:
        # Confirm the watcher is blocked on precisely this already-running arm.
        children = (watcher_path / 'task' / str(watcher) / 'children').read_text().split()
        assert children == [str(current_child)]
        for arm in ('native_base', 'adg', 'native_66'):
            log = (m.ROOT / f'parallel_{arm}.log').open('a')
            proc = subprocess.Popen([c.PYTHON, '-u', '-m', 'experiments.context_reference_5k_20260912.parallel_evaluate', '--arm', arm],
                cwd=c.WORK, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                stdout=log, stderr=subprocess.STDOUT)
            jobs.append((proc, log, arm))
        while any(proc.poll() is None for proc, _, _ in jobs):
            rows = [dict(pid=proc.pid, arm=arm, returncode=proc.poll()) for proc, _, arm in jobs]
            c.atomic(m.ROOT / 'parallel_evaluation.json', dict(phase='running', watcher_pid=watcher,
                original_current_child=current_child, only_watcher_paused=True, generation_unchanged=True, jobs=rows))
            if any(row['returncode'] not in (None, 0) for row in rows):
                raise RuntimeError(rows)
            time.sleep(5)
        assert all(proc.returncode == 0 for proc, _, _ in jobs)
        c.atomic(m.ROOT / 'parallel_evaluation.json', dict(phase='complete', watcher_pid=watcher,
            original_current_child=current_child, generation_unchanged=True, arms=[arm for _, _, arm in jobs]))
    finally:
        for proc, log, _ in jobs:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            log.close()
        if watcher_path.exists():
            os.kill(watcher, signal.SIGCONT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=m.ARMS)
    parser.add_argument('--watcher', type=int)
    parser.add_argument('--current-child', type=int)
    args = parser.parse_args()
    if args.arm:
        one(args.arm)
    else:
        main(args.watcher, args.current_child)
