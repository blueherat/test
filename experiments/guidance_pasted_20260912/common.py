from pathlib import Path
import json
import os
import time
import numpy as np
import torch
from experiments.lifting_scale_sweep_20260909 import Runtime, WORK, EXPS, atomic, read, sha, array_sha, asset_paths, source_paths

ROOT = EXPS/'guidance_pasted_20260912'
MODELS = ('sit_small','raev2')
SAMPLES = 400
SEED = 2026121251
PYTHON = '/home/zhoushunyu/miniconda3/envs/myenv/bin/python'
LATENTS = {'sit_small':(4,32,32), 'raev2':(1024,16,16)}
CLASSES = {'sit_small':100, 'raev2':1000}
PAIR_BATCH = {'sit_small':4,'raev2':2}
CFG_PEAK = {'sit_small':1.25,'raev2':1.0}
IG_PEAK = {'sit_small':.8,'raev2':.78}


def runtime(model):
    rt = Runtime(model)
    rt.model.eval().requires_grad_(False)
    if model == 'sit_small': rt.grid = torch.linspace(0,1,65,device='cuda')
    return rt


def check_parent(pid):
    if pid and os.getppid() != pid: raise RuntimeError('Controller exited; stopping worker')


def prepare_bank(model, stage, n=SAMPLES, seed=SEED):
    root = ROOT/model/stage/'inputs'
    if root.exists(): return root
    root.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for name in ('first','second'):
        arr = np.lib.format.open_memmap(root/(name+'.npy'),mode='w+',dtype=np.float32,shape=(n,*LATENTS[model]))
        for start in range(0,n,8): arr[start:start+8] = rng.standard_normal((min(8,n-start),*LATENTS[model]),dtype=np.float32)
        arr.flush()
    if n >= CLASSES[model]:
        labels = np.arange(n,dtype=np.int64)%CLASSES[model]; rng.shuffle(labels)
    else: labels = rng.permutation(CLASSES[model])[:n].astype(np.int64)
    np.save(root/'labels.npy',labels)
    return root


def bank(model,stage):
    root = ROOT/model/stage/'inputs'
    return [np.load(root/(name+'.npy'),mmap_mode='r') for name in ('first','second','labels')]


def cuda(x): return torch.from_numpy(np.array(x)).cuda()


def amount(rt, left, source, factor=1.):
    if rt.name == 'sit_small':
        if source == 'cfg': return factor*CFG_PEAK[rt.name] if left < .75 else 0.
        return factor*IG_PEAK[rt.name]*(6/7 if left < .25 else 1.) if left < .5 else 0.
    if source == 'cfg': return factor*CFG_PEAK[rt.name]
    return factor*IG_PEAK[rt.name] if .1 <= left <= 1. else 0.


def integrate(rt, noise, labels, field):
    rt.labels = labels; z = noise.clone(); before = rt.counts.copy()
    for step,(t,u) in enumerate(zip(rt.grid[:-1],rt.grid[1:])):
        h = u-t; left = float(t)
        first = field(z,t,left,step,0)
        if rt.name == 'sit_small':
            second = field(z+h*first,u,left,step,1)
            z = z+(h/2)*(first+second)
        else: z = z+h*first
        if not torch.isfinite(z).all() or z.abs().max() > 1e6:
            raise FloatingPointError(f'Invalid state at step {step}')
    return z,{k:rt.counts[k]-before[k] for k in before}


def source_manifest(extra):
    paths = {Path(__file__).resolve(),*map(Path,extra)}
    for model in MODELS: paths.update(source_paths(model))
    paths.update([WORK/'external/RAEv2/src/stage2/models/DDT.py',
        WORK/'experiments/imagenet100_sit_internal_v_head.py'])
    return {str(p.resolve()):sha(p) for p in sorted(paths)}


def save_batch(path, pixels, latents, labels, stats, request_hash, start, noise_hash):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,arr_0=pixels,latents=latents,labels=labels,start=start,
            request_sha256=request_hash,noise_sha256=noise_hash,**stats)
    tmp.replace(path)
    atomic(path.with_suffix('.json'),dict(sha256=sha(path),request_sha256=request_hash,start=start))


def collect(model,stage,arm,n=SAMPLES):
    root=ROOT/model/stage/arm
    files=sorted(root.glob('rank*/batch*.npz'),key=lambda p:int(read(p.with_suffix('.json'))['start']))
    if sum(len(np.load(p)['arr_0']) for p in files) != n: return None
    images=[];starts=[];times=[];calls=[]
    labels_expected=bank(model,stage)[2]
    records=[]
    for p in files:
        meta=read(p.with_suffix('.json')); assert sha(p)==meta['sha256']
        with np.load(p) as d:
            start=int(d['start']); count=len(d['arr_0']);starts.extend(range(start,start+count))
            assert str(d['request_sha256'])==meta['request_sha256']
            np.testing.assert_array_equal(d['labels'],labels_expected[start:start+count])
            assert d['arr_0'].dtype==np.uint8 and d['arr_0'].shape==(count,256,256,3)
            assert np.isfinite(d['latents']).all()
            images.append(d['arr_0']);times.append(float(d['seconds']));calls.append(float(d['full_calls']))
        records.append(dict(file=str(p),sha256=meta['sha256']))
    assert starts==list(range(n))
    output=root/'samples.npz'
    if not output.exists(): np.savez(output,arr_0=np.concatenate(images))
    summary=dict(complete=True,model=model,stage=stage,arm=arm,primary_samples=n,
        generated_paths=2*n,primary_branch=0,seconds=sum(times),full_calls_per_output=min(calls),
        full_calls_per_independent_primary=2*min(calls),prefix_calls_at_inference=0,
        samples_sha256=sha(output),records=records)
    assert min(calls)==max(calls)
    atomic(root/'summary.json',summary)
    return summary
