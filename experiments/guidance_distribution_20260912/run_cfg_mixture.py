"""Per-model source collection, classifier fitting, and the fixed CFG screen."""
import argparse
import os
import subprocess
import time
from experiments.guidance_pasted_20260912 import common as c
from . import cfg_source as source,mixture as m


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def controller(model,wait_pids):
    source.prepare(model)
    world=1 if model=='sit_small' else 3
    gpus=[0] if model=='sit_small' else [1,2,3]
    status=m.ROOT/(model+'_cfg_pipeline_status.json')
    # Both process termination and complete markers are required for GPU handoff.
    while True:
        done=all((m.ROOT/model/m.stage('ig')/kind/f'rank{rank}'/'complete.json').exists()
                 for kind in m.KINDS for rank in range(world))
        if done and not any(alive(pid) for pid in wait_pids):break
        if not any(alive(pid) for pid in wait_pids) and not done:
            raise RuntimeError('IG samplers exited without complete rank receipts')
        c.atomic(status,dict(phase='waiting_for_ig_sampling',pid=os.getpid(),wait_pids=wait_pids))
        time.sleep(5)
    jobs=[]
    try:
        for rank,gpu in enumerate(gpus):
            log=(m.ROOT/f'{model}_cfg_source_rank{rank}.log').open('a')
            p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_distribution_20260912.cfg_source',
                '--worker','--model',model,'--rank',str(rank),'--world',str(world),'--parent',str(os.getpid())],
                cwd=c.WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                stdout=log,stderr=subprocess.STDOUT)
            jobs.append((p,log))
        while any(p.poll() is None for p,_ in jobs):
            c.atomic(status,dict(phase='source_collection',pid=os.getpid(),
                jobs=[dict(pid=p.pid,returncode=p.poll()) for p,_ in jobs]))
            assert all(p.poll() in (None,0) for p,_ in jobs)
            time.sleep(8)
        assert all(p.returncode==0 for p,_ in jobs)
        for _,log in jobs:log.close()
        jobs=[]
        c.atomic(status,dict(phase='source_classifier_fit',pid=os.getpid()))
        source.fit(model)
        m.prepare(model,'cfg')
        for rank,gpu in enumerate(gpus):
            log=(m.ROOT/f'{model}_cfg_mixture_rank{rank}.log').open('a')
            p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_distribution_20260912.mixture',
                '--worker','--model',model,'--track','cfg','--rank',str(rank),'--world',str(world),
                '--parent',str(os.getpid())],cwd=c.WORK,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                stdout=log,stderr=subprocess.STDOUT)
            jobs.append((p,log))
        log=(m.ROOT/f'{model}_cfg_mixture_evaluate.log').open('a')
        p=subprocess.Popen([c.PYTHON,'-u','-m','experiments.guidance_distribution_20260912.run_mixture',
            '--evaluate','--model',model,'--track','cfg'],cwd=c.WORK,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
            stdout=log,stderr=subprocess.STDOUT)
        jobs.append((p,log))
        while any(p.poll() is None for p,_ in jobs):
            c.atomic(status,dict(phase='quality_screen',pid=os.getpid(),
                jobs=[dict(pid=p.pid,returncode=p.poll()) for p,_ in jobs]))
            assert all(p.poll() in (None,0) for p,_ in jobs)
            time.sleep(8)
        assert all(p.returncode==0 for p,_ in jobs)
        c.atomic(status,dict(phase='complete',pid=os.getpid(),model=model))
    finally:
        for p,log in jobs:
            if p.poll() is None:p.terminate()
            log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--wait-pids',type=int,nargs='+',required=True);a=p.parse_args()
    controller(a.model,a.wait_pids)
