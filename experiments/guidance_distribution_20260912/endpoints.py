"""Collect paired actual S/W/U endpoint distributions, then official features."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_pasted_20260912 import evaluate

ROOT=c.EXPS/'guidance_distribution_20260912'
PROTOCOL=c.WORK/'docs/GUIDANCE_DISTRIBUTION_DATA_PROTOCOL_20260912_ZH.md'
N=1000
STAGE='endpoint_sources'
KINDS=('strong','weak','null')


def configure():
    c.ROOT=ROOT


def prepare():
    configure()
    sources=c.source_manifest([Path(__file__).resolve(),PROTOCOL])
    for model in c.MODELS:
        bank=c.prepare_bank(model,STAGE,N,2026121291)
        request=dict(model=model,samples=N,seed=2026121291,kinds=KINDS,fit_indices=[0,499],test_indices=[500,999],
            sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths(model)},
            inputs={str(p):c.sha(p) for p in bank.glob('*.npy')},
            reference_outside_window='strong',sampler='Heun64' if model=='sit_small' else 'Euler100_shift8',
            second_noise_not_used=True,offline_source_collection=True)
        p=ROOT/model/STAGE/'request.json'
        if p.exists():assert c.read(p)==dict(request,kinds=list(KINDS))
        else:c.atomic(p,request)


@torch.inference_mode()
def worker(model,kinds,parent):
    configure()
    request=c.read(ROOT/model/STAGE/'request.json');rh=c.sha(ROOT/model/STAGE/'request.json')
    for group in ('sources','assets','inputs'):
        for p,h in request[group].items():assert c.sha(p)==h,p
    rt=c.runtime(model);noise,_,labels=c.bank(model,STAGE);batch=rt.batch
    for kind in kinds:
        root=ROOT/model/STAGE/kind
        for start in range(0,N,batch):
            c.check_parent(parent)
            path=root/'batches'/f'{start:04d}.npz'
            if path.exists():continue
            z=c.cuda(noise[start:start+batch]);y=c.cuda(labels[start:start+batch])
            def field(x,t,left,step,substage):
                if kind=='weak' and c.amount(rt,left,'ig'):
                    return rt.field(x,t,'base')
                if kind=='null' and c.amount(rt,left,'cfg'):
                    rt.labels=torch.full_like(y,c.CLASSES[model])
                    try:return rt.field(x,t,'full')
                    finally:rt.labels=y
                return rt.field(x,t,'full')
            torch.cuda.synchronize();begin=time.perf_counter()
            with rt.context():latent,counts=c.integrate(rt,z,y,field)
            pixels=rt.decode(latent);torch.cuda.synchronize();seconds=time.perf_counter()-begin
            c.save_batch(path,pixels,latent.float().cpu().numpy(),np.array(y.cpu()),dict(seconds=seconds,
                full_calls=counts['full'],prefix_calls=counts['prefix']),rh,start,c.array_sha(noise[start:start+batch]))
            if start%100==0:c.atomic(root/'progress.json',dict(complete=start+len(y),total=N,pid=os.getpid()))
        files=sorted((root/'batches').glob('*.npz'));pixels=[];total=0.;full=prefix=0.;records=[]
        for path in files:
            with np.load(path) as d:
                n=len(d['arr_0']);pixels.append(d['arr_0']);total+=float(d['seconds'])
                full+=int(d['full_calls'])*n;prefix+=int(d['prefix_calls'])*n
                assert np.isfinite(d['latents']).all()
            records.append(dict(file=str(path),sha256=c.sha(path)))
        value=np.concatenate(pixels);assert len(value)==N
        np.savez(root/'samples.npz',arr_0=value)
        c.atomic(root/'summary.json',dict(complete=True,model=model,stage=STAGE,arm=kind,primary_samples=N,
            generated_paths=N,seconds=total,full_calls_per_output=full/N,prefix_calls_per_output=prefix/N,
            samples_sha256=c.sha(root/'samples.npz'),records=records,request_sha256=rh))
        print(model,kind,'complete',round(total,2),flush=True)


def controller():
    prepare()
    plan=[('sit_small',KINDS,0),('raev2',('strong',),1),('raev2',('weak',),2),('raev2',('null',),3)]
    jobs=[]
    for model,kinds,gpu in plan:
        log=(ROOT/f'{model}_{"_".join(kinds)}.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
        proc=subprocess.Popen([c.PYTHON,'-m','experiments.guidance_distribution_20260912.endpoints','--worker',
            '--model',model,'--kinds',*kinds,'--parent',str(os.getpid())],env=env,cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log,model,kinds))
    for model in c.MODELS:
        log=(ROOT/f'{model}_evaluation.log').open('a')
        proc=subprocess.Popen([c.PYTHON,'-m','experiments.guidance_distribution_20260912.endpoints','--evaluate','--model',model],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'),
            cwd=c.WORK,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((proc,log,model,('evaluation',)))
    while any(p.poll() is None for p,_,_,_ in jobs):
        c.atomic(ROOT/'endpoint_status.json',dict(phase='running',pid=os.getpid(),
            jobs=[dict(pid=p.pid,model=m,kinds=k,returncode=p.poll()) for p,_,m,k in jobs]))
        failed=[(p.pid,p.poll()) for p,_,_,_ in jobs if p.poll() not in (None,0)]
        if failed:raise RuntimeError(failed)
        time.sleep(8)
    for p,log,_,_ in jobs:
        log.close();assert p.returncode==0
    c.atomic(ROOT/'endpoint_status.json',dict(phase='complete',pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--evaluate',action='store_true')
    p.add_argument('--model',choices=c.MODELS);p.add_argument('--kinds',nargs='+',choices=KINDS);p.add_argument('--parent',type=int,default=0);a=p.parse_args()
    if a.worker:worker(a.model,a.kinds,a.parent)
    elif a.evaluate:configure();evaluate.watch(a.model,STAGE,KINDS)
    else:controller()
