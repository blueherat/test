"""Coupled-particle CFG: share the extra guidance, keep conditional fields local."""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import torch
from . import common as c

STAGE='cfg_screen_400'
PROTOCOL=c.WORK/'docs/CFG_SHARED_PUSH_PROTOCOL_20260912_ZH.md'


def configs():
    return [dict(arm=f'{mode}_{level}',mode=mode,factor=factor) for level,factor in [('base',1.),('half',.5)]
        for mode in ('independent','shared_independent','shared_antithetic')]


@torch.inference_mode()
def sample(rt,first,second,labels,spec,zero=False):
    assert len(first)==len(second)==len(labels)
    n=len(first)
    if spec['mode']=='shared_antithetic': second=-first
    noise=torch.cat((first,second)); targets=torch.cat((labels,labels))
    def field(z,t,left,step,stage):
        s=rt.field(z,t,'full'); a=0. if zero else c.amount(rt,left,'cfg',spec['factor'])
        if not a: return s
        rt.labels=torch.full_like(targets,c.CLASSES[rt.name])
        try: u=rt.field(z,t,'full')
        finally:rt.labels=targets
        gap=s-u
        if spec['mode']!='independent':
            shared=(gap[:n]+gap[n:])*.5;gap=torch.cat((shared,shared))
        return s+a*gap
    with rt.context():return c.integrate(rt,noise,targets,field)


@torch.inference_mode()
def preflight(rt,rank):
    first,second,labels=c.bank(rt.name,STAGE); b=c.PAIR_BATCH[rt.name]
    f,s,y=[c.cuda(a[:b]) for a in (first,second,labels)]
    original=dict(arm='check',mode='independent',factor=1.)
    z,counts=sample(rt,f,s,y,original)
    anti,_=sample(rt,f,-f,y,original)
    # No interaction: changing the other branch cannot change the primary branch.
    assert torch.equal(z[:b],anti[:b])
    zeros=[sample(rt,f,s,y,dict(original,mode=mode),zero=True)[0][:b]
        for mode in ('independent','shared_independent','shared_antithetic')]
    assert all(torch.equal(zeros[0],v) for v in zeros[1:])
    # Same-state guidance sharing reduces to native CFG, including finite solver arithmetic.
    native,_=sample(rt,f,f,y,original)
    shared,_=sample(rt,f,f,y,dict(original,mode='shared_independent'))
    assert torch.equal(native,shared)
    assert counts['prefix']==0
    expected=224 if rt.name=='sit_small' else 200
    assert counts['full']==expected,counts
    c.atomic(c.ROOT/rt.name/STAGE/f'preflight{rank}.json',dict(passed=True,primary_partner_invariant_exact=True,
        all_zero_primary_exact=True,identical_pair_native_exact=True,full_calls_per_output=expected,
        prefix_calls=0,runtime_sources=rt.sources))


def prepare():
    c.ROOT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),PROTOCOL,c.WORK/'experiments/guidance_pasted_20260912/run_cfg.py']
    sources=c.source_manifest(paths)
    for model in c.MODELS:
        inputs=c.prepare_bank(model,STAGE)
        request=dict(model=model,samples=c.SAMPLES,primary_branch=0,pair_batch=c.PAIR_BATCH[model],
            configs=configs(),conceptual_cells=8,alias_independent_antithetic_primary=True,
            sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths(model)},
            inputs={str(p):c.sha(p) for p in inputs.glob('*.npy')},seed=c.SEED,
            solver='heun64' if model=='sit_small' else 'native_euler100_shift8',
            guidance_extra_peak=c.CFG_PEAK[model],cutoff=.75 if model=='sit_small' else None,
            purpose='paired 400 independent primaries for screening, not confirmation')
        path=c.ROOT/model/STAGE/'request.json'
        if path.exists():assert c.read(path)==request
        else:c.atomic(path,request)


def worker(model,rank,world,parent_pid):
    request_path=c.ROOT/model/STAGE/'request.json';request=c.read(request_path);h=c.sha(request_path)
    for kind in ('sources','assets','inputs'):
        for p,digest in request[kind].items():assert c.sha(p)==digest,p
    rt=c.runtime(model);preflight(rt,rank)
    first,second,labels=c.bank(model,STAGE);b=c.PAIR_BATCH[model]
    for spec in configs():
        out=c.ROOT/model/STAGE/spec['arm']/f'rank{rank}'
        for start in range(rank*b,c.SAMPLES,world*b):
            c.check_parent(parent_pid)
            path=out/f'batch{start:04d}.npz'
            if path.exists():
                receipt=c.read(path.with_suffix('.json'));assert receipt['sha256']==c.sha(path) and receipt['request_sha256']==h
                continue
            f,s,y=[c.cuda(a[start:start+b]) for a in (first,second,labels)]
            torch.cuda.synchronize();begin=time.perf_counter()
            z,counts=sample(rt,f,s,y,spec)
            pixels=rt.decode(z);torch.cuda.synchronize();seconds=time.perf_counter()-begin
            assert counts['prefix']==0
            c.save_batch(path,pixels[:b],z[:b].cpu().numpy(),np.array(labels[start:start+b]),
                dict(seconds=seconds,full_calls=counts['full'],pair_pixels=pixels[b:],
                    second_latents=z[b:].cpu().numpy()),h,start,c.array_sha(first[start:start+b]))
            if start%(50*b)==rank*b:c.atomic(c.ROOT/model/STAGE/f'progress{rank}.json',dict(arm=spec['arm'],start=start,rank=rank,seconds=seconds))
        c.atomic(out/'complete.json',dict(complete=True,request_sha256=h))
        print(model,rank,spec['arm'],'complete',flush=True)
    c.atomic(c.ROOT/model/STAGE/f'worker{rank}_complete.json',dict(complete=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model',choices=c.MODELS)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=2);p.add_argument('--parent-pid',type=int,default=0);a=p.parse_args()
    if a.prepare:prepare()
    else:worker(a.model,a.rank,a.world,a.parent_pid)
