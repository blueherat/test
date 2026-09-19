from __future__ import annotations
import fcntl
import os
import subprocess
import time
from . import catalog as c, train, pipeline
from experiments.lifting_scale_sweep_20260909 import atomic

PYTHON='/home/zhoushunyu/miniconda3/envs/myenv/bin/python'


def group(tasks,phase):
    processes=[];streams=[]
    try:
        for gpu,args in tasks:
            name='_'.join(args[-2:]).replace('/','_')
            stream=(c.ROOT/f'{phase}_{name}.log').open('a');streams.append(stream)
            processes.append(subprocess.Popen([PYTHON,'-u',*args],cwd=c.WORK,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')))
        atomic(c.ROOT/'status.json',dict(phase=phase,controller_pid=os.getpid(),worker_pids=[p.pid for p in processes]))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError(f'{phase} failed: {[p.poll() for p in processes]}')
            time.sleep(2)
        assert all(p.returncode==0 for p in processes)
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:p.wait(timeout=30)
        for stream in streams:stream.close()


def main():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    with (c.ROOT/'orchestration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        train.prepare()
        for start in (0,):
            group([(r,['-m','experiments.cfg_posterior_reference_20260912.train','--train',method])
                for r,method in enumerate(c.METHODS[start:start+4])],'training')
            if (c.ROOT/'STOP_AFTER_CURRENT').exists():return
        pipeline.pipeline()


if __name__=='__main__':main()
