"""Adopt the surviving workers after the original controller disappeared.

The frozen sampling code, inputs, point order, and evaluation are unchanged.
Completed samples are reused; no configuration is inferred from a timeout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.run_lifting_scale_sweep_20260909 import evaluate, wait_files
from experiments.lifting_scale_sweep_20260909 import ROOT, WORK, read, sha, atomic

REQUEST_SHA = '6563f5d43b3c52f1ebded624076135d583cf9e932f54e4c224d9e642412a05db'
ACTIVE_WORKERS = []


class Adopted:
    def __init__(self, pid, model, rank):
        self.pid = pid
        self.expected = [b'experiments.run_lifting_scale_sweep_20260909',
                         model.encode(), str(rank).encode()]
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        assert all(v in command for v in self.expected), command
        self.start = Path(f'/proc/{pid}/stat').read_text().split()[21]

    def poll(self):
        try:
            stat = Path(f'/proc/{self.pid}/stat').read_text().split()
            return None if stat[21] == self.start and stat[2] != 'Z' else -1
        except FileNotFoundError:
            return -1

    def terminate(self):
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)

    def wait(self):
        while self.poll() is None:
            time.sleep(.5)
        return None  # The exit status of a non-child is not observable here.


def run():
    lock = (ROOT / 'recovery_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not Path('/proc/3421465').exists(), 'Original controller is still present'
    assert sha(ROOT / 'request.json') == REQUEST_SHA
    request = read(ROOT / 'request.json')
    for path, digest in request['sources'].items():
        assert sha(path) == digest, path
    prior = read(ROOT / 'status.json')
    assert prior['model'] == 'sit_xl' and prior['arm'] == 'lifting_a1.000', prior
    adopted = [Adopted(pid, prior['model'], rank) for rank, pid in enumerate(prior['pids'])]
    assert len(adopted) == 4
    ACTIVE_WORKERS.extend(adopted)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    evidence = ROOT / 'controller_recovery' / stamp
    evidence.mkdir(parents=True)
    (evidence / 'recovery_controller.py').write_bytes(Path(__file__).read_bytes())
    for name in ('status.json', 'results.json', 'analysis_observer.json'):
        (evidence / name).write_bytes((ROOT / name).read_bytes())
    atomic(evidence / 'recovery.json', dict(
        reason='Original controller PID and exec handle absent; four worker processes survived and completed alpha=1 shards.',
        original_controller=3421465, adopted_workers=prior['pids'],
        controller_pid=os.getpid(), request_sha256=REQUEST_SHA,
        sampling_sources_unchanged=True, source_sha256=sha(Path(__file__)),
        termination_cause='Unknown; no assertion of numerical failure or OOM.',
        research_goal_achieved=False))
    results = read(ROOT / 'results.json')
    records = {(r['model'], r['arm']): r for r in results}
    started = time.perf_counter()
    for name in request['model_order']:
        spec = request['models'][name]
        out = ROOT / name
        for path, digest in spec['assets'].items():
            assert sha(path) == digest, path
        for rec in spec['reused'].values():
            assert sha(rec['sample_path']) == rec['sample_sha256']
            for path, digest in rec['source_files'].items():
                assert sha(path) == digest, path
            if (name, rec['arm']) not in records:
                results.append(rec); records[(name, rec['arm'])] = rec
        atomic(ROOT / 'results.json', results)
        logs = []
        if name == prior['model']:
            workers = adopted
            checks = read(out / 'preflight_passed.json')
            assert checks['passed']
        else:
            workers = []
            for rank in range(4):
                log = (out / f'worker{rank}.log').open('a'); logs.append(log)
                workers.append(subprocess.Popen(
                    [sys.executable, '-m', 'experiments.run_lifting_scale_sweep_20260909',
                     '--model', name, '--rank', str(rank)], cwd=WORK,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                ACTIVE_WORKERS.append(workers[-1])
            atomic(ROOT / 'status.json', dict(phase='preflight', model=name,
                   pids=[w.pid for w in workers], controller_pid=os.getpid(), research_goal_achieved=False))
            wait_files([out / f'preflight_rank{rank}.json' for rank in range(4)], workers)
            checks = [read(out / f'preflight_rank{rank}.json') for rank in range(4)]
            assert all(c['passed'] for c in checks)
            assert all(c['runtime_sources'] == checks[0]['runtime_sources'] for c in checks)
            atomic(out / 'preflight_passed.json', dict(passed=True, checks=checks))
        try:
            for point in spec['new_points']:
                directory = out / point['arm']
                key = (name, point['arm'])
                if key in records:
                    record = records[key]
                    assert record['request_sha256'] == REQUEST_SHA
                    if record['complete']:
                        assert sha(record['sample_path']) == record['sample_sha256']
                    assert (directory / 'advance.json').exists()
                    continue
                atomic(ROOT / 'status.json', dict(phase='sampling', model=name, arm=point['arm'],
                       pids=[w.pid for w in workers], controller_pid=os.getpid(), research_goal_achieved=False))
                wait_files([directory / f'rank{rank}/summary.json' for rank in range(4)], workers)
                atomic(ROOT / 'status.json', dict(phase='evaluating', model=name, arm=point['arm'],
                       pids=[w.pid for w in workers], controller_pid=os.getpid(), research_goal_achieved=False))
                record = evaluate(name, point, spec)
                results.append(record); records[key] = record
                atomic(ROOT / 'results.json', results)
                print(json.dumps(record), flush=True)
                atomic(directory / 'advance.json', dict(complete=True))
            codes = [w.wait() for w in workers]
            if name != prior['model']:
                assert codes == [0] * 4, codes
            atomic(out / 'complete.json', dict(all_points_processed=True, points=len(spec['new_points']),
                   reused_points=len(spec['reused']), worker_exit_codes=codes,
                   adopted_exit_status_unobservable=name == prior['model'],
                   numerical_failures=sum(r['model'] == name and not r['complete'] for r in results)))
        finally:
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
            for worker in workers:
                worker.wait()
            for log in logs:
                log.close()
        assert sha(ROOT / 'request.json') == REQUEST_SHA
        for path, digest in request['sources'].items():
            assert sha(path) == digest, path
    atomic(ROOT / 'status.json', dict(phase='complete', controller_pid=os.getpid(),
           wall_seconds_since_recovery=time.perf_counter() - started, results=len(results),
           numerical_failures=sum(not r['complete'] for r in results), research_goal_achieved=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-adopted', action='store_true', required=True)
    parser.parse_args()
    try:
        run()
    except Exception as error:
        for worker in ACTIVE_WORKERS:
            if worker.poll() is None:
                worker.terminate()
        for worker in ACTIVE_WORKERS:
            worker.wait()
        atomic(ROOT / 'status.json', dict(phase='failed', error=repr(error), controller_pid=os.getpid(), research_goal_achieved=False))
        raise
