"""One field evaluation per native slot, with a fixed reflected-view schedule."""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c

ROOT=c.EXPS/'reflection_guidance_20260912'
PROTOCOL=c.WORK/'docs/REFLECTION_GUIDANCE_PROTOCOL_20260912_ZH.md'
STAGE='screen_400';N=400;SEED=2026121381
ARMS=('strong_native','strong_abba','cfg_native','cfg_half','cfg_fixed_flip','cfg_abba',
      'cfg_mismatched','ig_native','ig_half','ig_fixed_flip','ig_abba')


def configure():c.ROOT=ROOT


def orientation(arm,step):
    if arm.endswith('fixed_flip'):return True
    if arm.endswith('abba') or arm.endswith('mismatched'):return step%4 in (1,2)
    return False


def transform(x,flip):return torch.flip(x,(-1,)) if flip else x


def field(rt,z,t,left,step,arm):
    track=arm.split('_')[0];flip=orientation(arm,step)
    query=transform(z,flip)
    strength=c.amount(rt,left,track,.5 if arm.endswith('half') else 1.) if track!='strong' else 0.
    if track=='ig' and strength:
        strong,weak=rt.pair(query,t)
        return transform(strong+strength*(strong-weak),flip)
    strong=transform(rt.field(query,t,'full'),flip)
    if not strength:return strong
    assert track=='cfg'
    reverse=(not flip) if arm=='cfg_mismatched' else flip
    labels=rt.labels
    rt.labels=torch.full_like(labels,c.CLASSES[rt.name])
    try:weak=transform(rt.field(transform(z,reverse),t,'full'),reverse)
    finally:rt.labels=labels
    return strong+strength*(strong-weak)


def sample(rt,noise,labels,arm):
    with rt.context():return c.integrate(rt,noise,labels,lambda z,t,left,i,j:field(rt,z,t,left,i,arm))


def expected(model,arm):return (224 if model=='sit_small' else 200) if arm.startswith('cfg') else (128 if model=='sit_small' else 100)


def verify(model):
    rp=ROOT/model/STAGE/'request.json';r=c.read(rp)
    for group in ('sources','assets','inputs'):
        for p,h in r[group].items():assert c.sha(p)==h,p
    return rp,r


def prepare():
    configure()
    for model in c.MODELS:
        bank=c.prepare_bank(model,STAGE,N,SEED)
        r=dict(model=model,arms=list(ARMS),samples=N,seed=SEED,stage=STAGE,
            single_path=True,extra_prefix=False,
            sources=c.source_manifest([Path(__file__).resolve(),PROTOCOL]),
            assets={str(p):c.sha(p) for p in c.asset_paths(model)},
            inputs={str(p):c.sha(p) for p in bank.glob('*.npy')})
        p=ROOT/model/STAGE/'request.json'
        if p.exists():assert c.read(p)==r
        else:c.atomic(p,r)


@torch.inference_mode()
def checks(rt,rank):
    noise,_,labels=c.bank(rt.name,STAGE);x=c.cuda(noise[:2]);y=c.cuda(labels[:2]);rows=[]
    assert torch.equal(transform(transform(x,True),True),x)
    for track in ('ig','cfg'):
        native,calls=sample(rt,x,y,track+'_native')
        def direct(z,t,left,i,j):
            a=c.amount(rt,left,track)
            if track=='ig' and a:
                s,w=rt.pair(z,t);return s+a*(s-w)
            s=rt.field(z,t,'full')
            if not a:return s
            rt.labels=torch.full_like(y,c.CLASSES[rt.name])
            try:w=rt.field(z,t,'full')
            finally:rt.labels=y
            return s+a*(s-w)
        with rt.context():other,_=c.integrate(rt,x,y,direct)
        assert torch.equal(native,other)
        fixed,othercalls=sample(rt,x,y,track+'_fixed_flip')
        mirrored,_=sample(rt,transform(x,True),y,track+'_native')
        assert torch.equal(fixed,transform(mirrored,True))
        assert calls==othercalls==dict(full=expected(rt.name,track),prefix=0),(track,calls)
        rows.append(dict(track=track,native_exact=True,fixed_conjugacy_exact=True,counts=calls))
    flags=np.asarray([orientation('cfg_abba',i) for i in range(len(rt.grid)-1)])
    h=np.abs(np.diff(rt.grid.cpu().numpy()))
    c.atomic(ROOT/rt.name/STAGE/f'checks{rank}.json',dict(passed=True,records=rows,
        reflected_steps=int(flags.sum()),total_steps=len(flags),reflected_time_mass=float(h[flags].sum()),
        runtime_sources=rt.sources))


@torch.inference_mode()
def worker(model,rank,world,parent):
    configure();rp,request=verify(model);rh=c.sha(rp);rt=c.runtime(model);checks(rt,rank)
    noise,_,labels=c.bank(model,STAGE)
    for arm in ARMS:
        root=ROOT/model/STAGE/arm/f'rank{rank}'
        for start in range(rank*rt.batch,N,world*rt.batch):
            c.check_parent(parent);p=root/f'batch{start:04d}.npz'
            if p.exists():
                r=c.read(p.with_suffix('.json'));assert c.sha(p)==r['sha256'] and r['request_sha256']==rh
                continue
            x=c.cuda(noise[start:start+rt.batch]);y=c.cuda(labels[start:start+rt.batch])
            torch.cuda.synchronize();begin=time.perf_counter()
            z,counts=sample(rt,x,y,arm);pixels=rt.decode(z)
            torch.cuda.synchronize();seconds=time.perf_counter()-begin
            assert counts==dict(full=expected(model,arm),prefix=0),(arm,counts)
            c.save_batch(p,pixels,z.float().cpu().numpy(),y.cpu().numpy(),
                dict(seconds=seconds,full_calls=counts['full'],prefix_calls=0),rh,start,c.array_sha(noise[start:start+len(y)]))
            if start%(25*rt.batch)==rank*rt.batch:
                c.atomic(ROOT/model/STAGE/f'progress{rank}.json',dict(pid=os.getpid(),arm=arm,start=start,world=world))
        c.atomic(root/'complete.json',dict(complete=True,request_sha256=rh))
        print(model,rank,arm,'complete',flush=True)
    c.atomic(ROOT/model/STAGE/f'worker{rank}_complete.json',dict(complete=True))


def collect(model,arm):
    root=ROOT/model/STAGE/arm
    if (root/'summary.json').exists():return c.read(root/'summary.json')
    files=sorted(root.glob('rank*/batch*.npz'),key=lambda p:int(c.read(p.with_suffix('.json'))['start']))
    if sum(len(np.load(p)['labels']) for p in files)!=N:return None
    noise,_,labels=c.bank(model,STAGE);pixels=[];records=[];coverage=[];seconds=0.
    for p in files:
        meta=c.read(p.with_suffix('.json'));assert c.sha(p)==meta['sha256']
        with np.load(p) as d:
            start=int(d['start']);n=len(d['labels']);coverage.extend(range(start,start+n))
            np.testing.assert_array_equal(d['labels'],labels[start:start+n])
            assert str(d['noise_sha256'])==c.array_sha(noise[start:start+n])
            assert str(d['request_sha256'])==c.sha(ROOT/model/STAGE/'request.json')
            assert int(d['full_calls'])==expected(model,arm) and int(d['prefix_calls'])==0
            assert np.isfinite(d['latents']).all()
            pixels.append(d['arr_0']);seconds+=float(d['seconds'])
        records.append(dict(file=str(p),sha256=meta['sha256']))
    assert coverage==list(range(N))
    np.savez(root/'samples.npz',arr_0=np.concatenate(pixels))
    r=dict(complete=True,model=model,stage=STAGE,arm=arm,primary_samples=N,generated_paths=N,
        full_calls_per_output=expected(model,arm),prefix_calls_at_inference=0,seconds=seconds,
        samples_sha256=c.sha(root/'samples.npz'),records=records)
    c.atomic(root/'summary.json',r);return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--model',choices=c.MODELS)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1)
    p.add_argument('--parent',type=int,default=0);a=p.parse_args()
    if a.prepare:prepare()
    else:worker(a.model,a.rank,a.world,a.parent)
