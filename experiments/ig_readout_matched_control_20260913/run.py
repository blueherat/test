"""Bounded train, sample, evaluate sequence for one matched control."""
import argparse
import os
import subprocess
import time
from experiments.guidance_pasted_20260912 import common as c, evaluate
from . import core as m


def controller():
    m.prepare_training()
    for stage, arguments in [('train', ['--train']), ('sample', [])]:
        with (m.ROOT / f'{stage}.log').open('a') as log:
            proc = subprocess.Popen([c.PYTHON, '-u', '-m', 'experiments.ig_readout_matched_control_20260913.core',
                *arguments, '--parent', str(os.getpid())], cwd=c.WORK,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'),
                stdout=log, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase=stage, child_pid=proc.pid))
                    time.sleep(5)
                if proc.returncode:
                    raise RuntimeError((stage, proc.returncode))
            finally:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
    m.collect()
    c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase='evaluate'))
    result = evaluate.evaluate('sit_small', m.STAGE, m.ARM)
    c.atomic(m.ROOT / 'sit_small' / m.STAGE / 'results.json', [result])
    c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase='complete', new_arms=1, new_images=1000))


if __name__ == '__main__':
    try:
        controller()
    except BaseException as error:
        c.atomic(m.ROOT / 'status.json', dict(pid=os.getpid(), phase='failed', error=repr(error)))
        raise
