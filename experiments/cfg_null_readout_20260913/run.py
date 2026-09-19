"""Two sampling workers and two independent evaluator partitions, with bounded lifetime."""
import argparse
import os
import subprocess
import time
from experiments.guidance_pasted_20260912 import common as c, evaluate
from . import core as m


def watch(rank):
    m.configure()
    done = {}
    arms = m.ARMS[rank::2]
    while len(done) < len(arms):
        for arm in arms:
            if arm in done or not m.collect(arm):
                continue
            done[arm] = evaluate.evaluate('sit_small', m.STAGE, arm)
            c.atomic(m.ROOT / 'sit_small' / m.STAGE / f'results_eval{rank}.json', list(done.values()))
        if len(done) < len(arms):
            time.sleep(5)


def controller():
    m.prepare()
    jobs = []
    for kind in ('sample', 'evaluate'):
        for rank in range(2):
            module = 'experiments.cfg_null_readout_20260913.' + ('core' if kind == 'sample' else 'run')
            arguments = ['--rank', str(rank), '--world', '2', '--parent', str(os.getpid())] if kind == 'sample' else ['--evaluate', str(rank)]
            log = (m.ROOT / f'{kind}{rank}.log').open('a')
            proc = subprocess.Popen([c.PYTHON, '-u', '-m', module, *arguments], cwd=c.WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(2 + rank) if kind == 'sample' else '',
                    OMP_NUM_THREADS='2' if kind == 'sample' else '4', OPENBLAS_NUM_THREADS='2' if kind == 'sample' else '4'),
                stdout=log, stderr=subprocess.STDOUT)
            jobs.append((proc, log, kind, rank))
    try:
        while any(proc.poll() is None for proc, _, _, _ in jobs):
            rows = [dict(pid=proc.pid, kind=kind, rank=rank, returncode=proc.poll()) for proc, _, kind, rank in jobs]
            c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase='running', jobs=rows))
            if any(r['returncode'] not in (None, 0) for r in rows):
                raise RuntimeError(rows)
            time.sleep(5)
        results = [c.read(m.ROOT / 'sit_small' / m.STAGE / arm / 'metrics.json') for arm in m.ARMS]
        c.atomic(m.ROOT / 'sit_small' / m.STAGE / 'results.json', results)
        c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase='complete', arms=7, images=2800))
    except BaseException as error:
        c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase='failed', error=repr(error)))
        raise
    finally:
        for proc, log, _, _ in jobs:
            if proc.poll() is None:
                proc.terminate()
            proc.wait()
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.evaluate is None:
        controller()
    else:
        watch(args.evaluate)
