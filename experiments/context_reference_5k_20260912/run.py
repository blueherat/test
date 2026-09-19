"""Bounded synchronous controller; inspect child handles before any recovery."""
import argparse
import os
import subprocess
import time
from experiments.guidance_pasted_20260912 import common as c,evaluate
from . import core as m


def watch(model):
    m.configure();done={}
    while len(done)<len(m.ARMS):
        for arm in m.ARMS:
            if arm in done:continue
            # A batch NPZ is renamed just before its JSON sidecar is written.
            # Only collect after both writers have closed this entire arm.
            arm_root=m.ROOT/model/m.STAGE/arm
            if not all((arm_root/f'rank{rank}/complete.json').exists() for rank in range(2)):continue
            if not m.collect(model,arm):continue
            done[arm]=evaluate.evaluate(model,m.STAGE,arm)
            c.atomic(m.ROOT/model/m.STAGE/'results.json',[done[a] for a in m.ARMS if a in done])
        if len(done)<len(m.ARMS):time.sleep(5)
    c.atomic(m.ROOT/model/m.STAGE/'evaluation_complete.json',dict(complete=True,arms=len(done)))


def controller():
    m.prepare();jobs=[]
    for model,rank,world,gpu in [('sit_small',0,2,1),('sit_small',1,2,2)]:
        log=(m.ROOT/f'{model}_rank{rank}.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.context_reference_5k_20260912.core',
            '--model',model,'--rank',str(rank),'--world',str(world),'--parent',str(os.getpid())],
            cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,model,'sample',rank))
    for model in ('sit_small',):
        log=(m.ROOT/f'{model}_evaluate.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.context_reference_5k_20260912.run','--evaluate',model],
            cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,model,'evaluate',None))
    try:
        while any(p.poll() is None for p,_,_,_,_ in jobs):
            rows=[dict(pid=p.pid,model=model,kind=kind,rank=rank,returncode=p.poll()) for p,_,model,kind,rank in jobs]
            c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='running',jobs=rows))
            failed=[r for r in rows if r['returncode'] not in (None,0)]
            if failed:
                c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='failed',jobs=rows))
                raise RuntimeError(failed)
            time.sleep(8)
        assert all(p.returncode==0 for p,_,_,_,_ in jobs)
        c.atomic(m.ROOT/'status.json',dict(pid=os.getpid(),phase='complete',arms=4,images=20000))
    finally:
        for p,log,_,_,_ in jobs:
            if p.poll() is None:p.terminate()
            p.wait()
            log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',choices=c.MODELS);a=p.parse_args()
    if a.evaluate:watch(a.evaluate)
    else:controller()
