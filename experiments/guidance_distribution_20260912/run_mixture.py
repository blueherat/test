"""A bounded six-arm experiment, with independent CPU feature evaluation."""
import argparse
import os
import subprocess
import time
from experiments.guidance_pasted_20260912 import common as c,evaluate
from . import mixture as m


def watch(model,track):
    m.configure();done={}
    while len(done)<len(m.KINDS):
        for kind in m.KINDS:
            root=m.ROOT/model/m.stage(track)/kind
            if kind in done:continue
            if not (root/'summary.json').exists():
                if not m.collect(model,track,kind):continue
            done[kind]=evaluate.evaluate(model,m.stage(track),kind)
            c.atomic(m.ROOT/model/m.stage(track)/'results.json',[done[k] for k in m.KINDS if k in done])
        if len(done)<len(m.KINDS):time.sleep(5)
    c.atomic(m.ROOT/model/m.stage(track)/'evaluation_complete.json',dict(complete=True,arms=len(done)))


def controller(track,models):
    for model in models:m.prepare(model,track)
    plan=[]
    if 'sit_small' in models:plan.append(('sit_small',0,1,0))
    if 'raev2' in models:plan.extend([('raev2',rank,3,rank+1) for rank in range(3)])
    jobs=[]
    for model,rank,world,gpu in plan:
        log=(m.ROOT/f'{model}_{track}_mixture_rank{rank}.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_distribution_20260912.mixture',
            '--worker','--model',model,'--track',track,'--rank',str(rank),'--world',str(world),'--parent',str(os.getpid())],
            cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,model,'sample',rank))
    for model in models:
        log=(m.ROOT/f'{model}_{track}_mixture_evaluate.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_distribution_20260912.run_mixture',
            '--evaluate','--model',model,'--track',track],cwd=c.WORK,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log,model,'evaluate',None))
    try:
        while any(p.poll() is None for p,_,_,_,_ in jobs):
            state=[dict(pid=p.pid,model=model,kind=kind,rank=rank,returncode=p.poll()) for p,_,model,kind,rank in jobs]
            c.atomic(m.ROOT/(track+'_mixture_status.json'),dict(phase='running',pid=os.getpid(),jobs=state))
            failed=[x for x in state if x['returncode'] not in (None,0)]
            if failed:raise RuntimeError(failed)
            time.sleep(8)
        assert all(p.returncode==0 for p,_,_,_,_ in jobs)
        c.atomic(m.ROOT/(track+'_mixture_status.json'),dict(phase='complete',pid=os.getpid(),models=models))
    finally:
        for p,log,_,_,_ in jobs:
            if p.poll() is None:p.terminate()
            log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=('ig','cfg'),required=True)
    p.add_argument('--model',choices=c.MODELS,nargs='+',default=list(c.MODELS))
    p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    if a.evaluate:
        assert len(a.model)==1;watch(a.model[0],a.track)
    else:controller(a.track,a.model)
