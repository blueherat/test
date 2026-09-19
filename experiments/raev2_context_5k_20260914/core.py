"""Frozen 5K bank, shared original sampler, actual call accounting and replay."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as old
from experiments.raev2_context_20k_20260914 import sample as prior,train as tr

ROOT = c.EXPS/'raev2_context_5k_20260914'
STAGE = 'confirm_5000'
BASE = ROOT/'raev2'/STAGE
N = 5000
BATCH = 16
SEED = 2026091421
ARMS = ('native_base','context20k_half')
PROTOCOL = c.WORK/'docs/RAEV2_CONTEXT_5K_PROTOCOL_20260914_ZH.md'


def configure():
    c.ROOT = ROOT


def prepare():
    configure()
    assert c.read(tr.ROOT/'completion_verification.json')['passed']
    training = c.read(tr.TRAIN/'summary.json')
    assert training['steps'] == 20000 and training['replay_exact']
    assert c.sha(tr.TRAIN/'head.pt') == training['head_sha256']
    bank = BASE/'inputs'
    bank.mkdir(parents=True,exist_ok=True)
    stamp = bank/'complete.json'
    if not stamp.exists():
        g = np.random.default_rng(SEED)
        tmp = bank/'first.tmp.npy'
        noise = np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float32,shape=(N,*c.LATENTS['raev2']))
        for start in range(0,N,8):
            noise[start:start+8] = g.standard_normal((min(8,N-start),*c.LATENTS['raev2']),dtype=np.float32)
        noise.flush()
        del noise
        tmp.replace(bank/'first.npy')
        labels = np.arange(N,dtype=np.int64)%1000
        g.shuffle(labels)
        np.save(bank/'labels.npy',labels)
        c.atomic(stamp,dict(complete=True,samples=N,seed=SEED,paths_per_sample=1,
            inputs={str(p):c.sha(p) for p in bank.glob('*.npy')}))
    saved = c.read(stamp)
    assert saved['samples'] == N and saved['seed'] == SEED
    for path,digest in saved['inputs'].items():
        assert c.sha(path) == digest,path
    labels = np.load(bank/'labels.npy')
    unique,counts = np.unique(labels,return_counts=True)
    assert np.array_equal(unique,np.arange(1000)) and np.all(counts == 5)
    sources = c.source_manifest([Path(__file__).resolve(),Path(old.__file__).resolve(),Path(prior.__file__).resolve(),
        Path(tr.__file__).resolve(),PROTOCOL,Path(__file__).with_name('run.py'),
        c.WORK/'experiments/evaluate_raev2_official_samples.py'])
    request = dict(model='raev2',stage=STAGE,samples=N,seed=SEED,arms=list(ARMS),batch=BATCH,
        alpha={'native_base':.78,'context20k_half':.39},checkpoint_step=20000,
        sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths('raev2')},inputs=saved['inputs'],
        heads={str(p):c.sha(p) for p in (tr.TRAIN/'head.pt',tr.TRAIN/'summary.json',tr.TRAIN/'request.json')},
        selection={str(p):c.sha(p) for p in (tr.ROOT/'screen_decision.json',tr.ROOT/'completion_verification.json')},
        sampler='Euler100_shift8',activity=[.1,1.],single_path=True,full_calls=100,extra_prefix=0,
        evaluation=dict(device='cuda',gpu=2,batch=64,reference='imagenet_256_fid_stats',same_for_both_arms=True))
    path = BASE/'request.json'
    if path.exists():
        assert c.read(path) == request
    else:
        c.atomic(path,request)
    c.atomic(ROOT/'status.json',dict(phase='prepared',samples_per_arm=N,arms=list(ARMS)))
    print('Prepared two-arm 5K bank',flush=True)


def verify():
    request = old.verify_request(BASE/'request.json')
    for p,digest in request['selection'].items():
        assert c.sha(p) == digest,p
    return request


def bank():
    return np.load(BASE/'inputs/first.npy',mmap_mode='r'),np.load(BASE/'inputs/labels.npy')


def load(rt):
    state = torch.load(tr.TRAIN/'head.pt',map_location='cpu',weights_only=True)
    assert state['step'] == 20000
    head = old.make_head(rt).eval().requires_grad_(False)
    head.load_state_dict(state['ema']['context'])
    return {'context20k':head}


@torch.inference_mode()
def preflight():
    configure()
    verify()
    rt = c.runtime('raev2')
    heads = load(rt)
    source_noise = np.load(prior.OLD_SCREEN/'inputs/first.npy',mmap_mode='r')
    labels = np.load(prior.OLD_SCREEN/'inputs/labels.npy')
    x,y = c.cuda(source_noise[:4]),c.cuda(labels[:4])
    counter = prior.Calls(rt,heads)
    rows=[]
    for arm in ARMS:
        capture = old.Capture(rt) if arm.startswith('context') else None
        counter.reset()
        z,counts = prior.trajectory(rt,heads,capture,x,y,arm)
        pixels = rt.decode(z)
        old_root = prior.OLD_SCREEN/'native_base' if arm=='native_base' else tr.ROOT/'raev2/paired_400'/arm
        paths = list(old_root.glob('rank*/batch0000.npz'))
        assert len(paths) == 1
        with np.load(paths[0]) as data:
            np.testing.assert_array_equal(z.cpu().numpy(),data['latents'])
            np.testing.assert_array_equal(pixels,data['arr_0'])
        expected = sum(bool(c.amount(rt,float(t),'ig')) for t in rt.grid[:-1]) if arm.startswith('context') else 0
        assert counts == dict(full=100,prefix=0) and counter.blocks == [100]*30
        assert counter.heads == {'context20k':expected}
        rows.append(dict(arm=arm,samples=4,batch=4,old_latents_and_pixels_exact=True,
            block_calls=list(counter.blocks),mlp_calls=counter.heads['context20k']))
        if capture is not None:
            capture.close()
    assert all(not p.requires_grad and p.grad is None for p in rt.model.parameters())
    counter.close()
    c.atomic(ROOT/'preflight.json',dict(passed=True,old_replays=rows,
        strong_frozen=True,new_quality_images=0,formal_batch=BATCH,
        note='Old batch4 replay is exact; formal paired 5K uses common batch16.'))
    print('Preflight passed: both retained configurations replay exactly',flush=True)


@torch.inference_mode()
def worker(rank,world,parent):
    configure()
    verify()
    assert c.read(ROOT/'preflight.json')['passed']
    rt = c.runtime('raev2')
    heads = load(rt)
    counter = prior.Calls(rt,heads)
    noise,labels = bank()
    rh = c.sha(BASE/'request.json')
    expected = sum(bool(c.amount(rt,float(t),'ig')) for t in rt.grid[:-1])
    for arm in ARMS:
        capture = old.Capture(rt) if arm.startswith('context') else None
        out = BASE/arm/f'rank{rank}'
        for start in range(rank*BATCH,N,world*BATCH):
            c.check_parent(parent)
            path = out/f'batch{start:04d}.npz'
            if path.exists():
                meta=c.read(path.with_suffix('.json'))
                assert c.sha(path)==meta['sha256'] and meta['request_sha256']==rh
                continue
            x,y = c.cuda(noise[start:start+BATCH]),c.cuda(labels[start:start+BATCH])
            counter.reset()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z,counts = prior.trajectory(rt,heads,capture,x,y,arm)
            pixels = rt.decode(z)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-begin
            assert counts==dict(full=100,prefix=0) and counter.blocks==[100]*30
            assert counter.heads['context20k']==(expected if arm.startswith('context') else 0)
            assert pixels.shape==(len(y),256,256,3) and pixels.dtype==np.uint8
            c.save_batch(path,pixels,z.cpu().numpy(),y.cpu().numpy(),
                dict(seconds=elapsed,full_calls=100,prefix_calls=0,block_calls=np.array(counter.blocks),
                    mlp_calls=counter.heads['context20k'],batch=len(y)),rh,start,c.array_sha(noise[start:start+len(y)]))
            c.atomic(BASE/f'progress{rank}.json',dict(pid=os.getpid(),arm=arm,last_start=start,
                last_batch=len(y),world=world,batch=BATCH,seconds_last_batch=elapsed))
        c.atomic(out/'complete.json',dict(complete=True,request_sha256=rh))
        print(rank,arm,'complete',flush=True)
        if capture is not None:
            capture.close()
    counter.close()


def collect(arm):
    configure()
    root=BASE/arm
    if (root/'summary.json').exists():
        return c.read(root/'summary.json')
    assert all((root/f'rank{rank}/complete.json').exists() for rank in range(3))
    files=sorted(root.glob('rank*/batch*.npz'),key=lambda p:int(c.read(p.with_suffix('.json'))['start']))
    noise,labels=bank()
    pixels=[];records=[];coverage=[];seconds=0.;rh=c.sha(BASE/'request.json')
    active=c.read(ROOT/'preflight.json')['old_replays'][1]['mlp_calls'] if arm.startswith('context') else 0
    for path in files:
        meta=c.read(path.with_suffix('.json'))
        assert c.sha(path)==meta['sha256'] and meta['request_sha256']==rh
        with np.load(path) as d:
            start,n=int(d['start']),len(d['labels']);coverage.extend(range(start,start+n))
            np.testing.assert_array_equal(d['labels'],labels[start:start+n])
            assert str(d['noise_sha256'])==c.array_sha(noise[start:start+n])
            assert str(d['request_sha256'])==rh and np.isfinite(d['latents']).all()
            assert int(d['full_calls'])==100 and int(d['prefix_calls'])==0
            assert np.array_equal(d['block_calls'],np.full(30,100)) and int(d['mlp_calls'])==active
            pixels.append(d['arr_0']);seconds+=float(d['seconds'])
        records.append(dict(file=str(path),sha256=meta['sha256']))
    assert coverage==list(range(N))
    np.savez(root/'samples.npz',arr_0=np.concatenate(pixels))
    summary=dict(complete=True,model='raev2',stage=STAGE,arm=arm,primary_samples=N,generated_paths=N,
        seconds=seconds,full_calls_per_output=100,prefix_calls_at_inference=0,mlp_calls_per_output=active,
        samples_sha256=c.sha(root/'samples.npz'),request_sha256=rh,records=records)
    c.atomic(root/'summary.json',summary)
    return summary


@torch.inference_mode()
def benchmark():
    configure();verify()
    rt=c.runtime('raev2');heads=load(rt);noise,labels=bank()
    x,y=c.cuda(noise[:BATCH]),c.cuda(labels[:BATCH]);seconds={a:[] for a in ARMS}
    for repeat in range(4):
        for arm in ARMS if repeat%2==0 else ARMS[::-1]:
            capture=old.Capture(rt) if arm.startswith('context') else None
            torch.cuda.synchronize();begin=time.perf_counter()
            z,counts=prior.trajectory(rt,heads,capture,x,y,arm)
            rt.decode(z);torch.cuda.synchronize();elapsed=time.perf_counter()-begin
            assert counts==dict(full=100,prefix=0)
            if capture is not None:capture.close()
            if repeat:seconds[arm].append(elapsed)
    medians={a:float(np.median(v)) for a,v in seconds.items()}
    c.atomic(ROOT/'benchmark.json',dict(complete=True,batch=BATCH,repeats=3,warmups=1,seconds=seconds,
        medians=medians,relative_mlp_change=medians[ARMS[1]]/medians[ARMS[0]]-1,
        full_calls=100,extra_prefix=0,includes_decode=True,mlp_parameters=5635744,
        different_fixed_guidance_coefficients=True,additional_training=False))
    print('Benchmark complete',medians,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true')
    p.add_argument('--benchmark',action='store_true');p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world',type=int,default=3);p.add_argument('--parent',type=int,default=0);a=p.parse_args()
    if a.prepare:prepare()
    elif a.preflight:preflight()
    elif a.benchmark:benchmark()
    else:worker(a.rank,a.world,a.parent)
