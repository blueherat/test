"""One independent matched pair of weak-model fits per physical GPU."""
from __future__ import annotations
import argparse
import fcntl
import os
import subprocess
import time
from .data import ROOT, WORK, METHODS
from . import train
from experiments.lifting_scale_sweep_20260909 import atomic, read

PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
JOBS = [('native', 'defensive'), ('clt', 'heat'),
        ('factor', 'plain_mix'), ('local_label', 'random_label')]


def run():
    train.verify()
    with (ROOT/'training_controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes, streams = [], []
        (ROOT/'training').mkdir(exist_ok=True)
        try:
            for rank in range(4):
                log = (ROOT/'training'/f'rank{rank}.log').open('a')
                streams.append(log)
                command = [PYTHON, '-u', '-m', __package__+'.train_batch', '--rank', str(rank)]
                process = subprocess.Popen(command, cwd=WORK, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank),
                             OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
                processes.append(process)
            atomic(ROOT/'training_processes.json', dict(controller_pid=os.getpid(),
                worker_pids=[p.pid for p in processes], jobs=JOBS))
            while any(p.poll() is None for p in processes):
                assert not any(p.poll() not in (None, 0) for p in processes), [p.poll() for p in processes]
                progress = {method:read(ROOT/'training'/method/'status.json')
                    for method in METHODS if (ROOT/'training'/method/'status.json').exists()}
                atomic(ROOT/'training_status.json', dict(phase='running', methods=progress,
                    pid=os.getpid(), updated_unix=time.time()))
                time.sleep(10)
            assert all(p.returncode == 0 for p in processes)
            complete = all((ROOT/'training'/method/'complete.json').exists() for method in METHODS)
            atomic(ROOT/'training_status.json', dict(
                phase='complete' if complete else 'stopped_after_step',
                methods=METHODS, worker_exit_codes=[p.returncode for p in processes],
                updated_unix=time.time(), no_sampling_launched=True))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait(timeout=30)
            for log in streams:
                log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=range(4))
    args = parser.parse_args()
    if args.rank is None:
        run()
    else:
        for method in JOBS[args.rank]:
            if (ROOT/'STOP_AFTER_CURRENT').exists():
                break
            train.train(method)
