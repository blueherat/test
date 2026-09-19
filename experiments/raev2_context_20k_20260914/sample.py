"""Same sampler, two pre-existing strengths, and frozen retained baselines."""
import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_distribution_20260912 import local_head as old
from . import train as tr

ROOT = tr.ROOT
OLD_SCREEN = c.EXPS/'input_local_completion_20260912/raev2/screen_400'
SCREEN = 'paired_400'
CONFIRM = 'confirm_1000'
CONTROL_ARMS = ('native_base','native_half','context_base','context_half')
SCREEN_ARMS = ('context20k_base','context20k_half')


def configure():
    c.ROOT = ROOT


def prepare(stage, chosen=None):
    configure()
    old.verify_request(tr.TRAIN/'request.json')
    summary = c.read(tr.TRAIN/'summary.json')
    assert summary['complete'] and summary['steps'] == 20000 and summary['replay_exact']
    assert c.sha(tr.TRAIN/'head.pt') == summary['head_sha256']
    root = ROOT/'raev2'/stage
    if stage == SCREEN:
        n, arms = 400, list(SCREEN_ARMS)
        old.verify_request(OLD_SCREEN/'request.json')
        bank = root/'inputs'
        bank.mkdir(parents=True,exist_ok=True)
        for name in ('first.npy','second.npy','labels.npy'):
            target = OLD_SCREEN/'inputs'/name
            p = bank/name
            if not p.exists():
                p.symlink_to(target)
            assert c.sha(p) == c.sha(target)
        comparisons = {str(OLD_SCREEN/arm/f):c.sha(OLD_SCREEN/arm/f)
            for arm in CONTROL_ARMS for f in ('samples.npz','metrics.json','summary.json')}
    else:
        assert stage == CONFIRM and chosen in SCREEN_ARMS
        n, arms = 1000, ['native_base','native_half',chosen.replace('20k','3k'),chosen]
        bank = c.prepare_bank('raev2',stage,n,2026091411)
        comparisons = {}
    sources = c.source_manifest([Path(__file__).resolve(),Path(tr.__file__).resolve(),tr.PROTOCOL,
        c.WORK/'experiments/raev2_context_20k_20260914/run.py',
        c.WORK/'experiments/evaluate_raev2_official_samples.py'])
    request = dict(model='raev2',stage=stage,samples=n,arms=arms,chosen=chosen,
        source_noise='old paired 400' if stage == SCREEN else 'new balanced 1000 seed2026091411',
        sources=sources,assets={str(p):c.sha(p) for p in c.asset_paths('raev2')},
        inputs={str(p):c.sha(p) for p in bank.glob('*.npy')},
        heads={str(p):c.sha(p) for p in (tr.TRAIN/'head.pt',tr.TRAIN/'request.json',tr.TRAIN/'summary.json',tr.OLD/'head.pt')},
        comparisons=comparisons,single_path=True,alpha_by_suffix={'base':.78,'half':.39},
        grid='Euler100_shift8',active_time_interval=[.1,1.],full_calls=100,extra_prefix=0)
    rp = root/'request.json'
    if rp.exists():
        assert c.read(rp) == request
    else:
        c.atomic(rp,request)
    return request


def verify(stage):
    rp = ROOT/'raev2'/stage/'request.json'
    request = old.verify_request(rp)
    for path,digest in request['comparisons'].items():
        assert c.sha(path) == digest,path
    return request


def load(rt):
    retained = old.load_heads(rt)
    h20 = old.make_head(rt).eval().requires_grad_(False)
    state = torch.load(tr.TRAIN/'head.pt',map_location='cpu',weights_only=True)
    assert state['step'] == 20000
    h20.load_state_dict(state['ema']['context'])
    return {'context3k':retained['context'],'context20k':h20,'local':retained['local']}


class Calls:
    def __init__(self, rt, heads):
        self.blocks = [0]*len(rt.model.blocks)
        self.heads = {k:0 for k in heads}
        self.handles = []
        for i,block in enumerate(rt.model.blocks):
            def count(module,args,value,index=i):
                self.blocks[index] += 1
            self.handles.append(block.register_forward_hook(count))
        for key,head in heads.items():
            def count(module,args,value,name=key):
                self.heads[name] += 1
            self.handles.append(head.register_forward_hook(count))
    def reset(self):
        self.blocks[:] = [0]*len(self.blocks)
        for key in self.heads:
            self.heads[key] = 0
    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def trajectory(rt, heads, capture, x, labels, arm):
    if arm.startswith('native'):
        chosen,kind = {},arm
    else:
        key,suffix = arm.rsplit('_',1)
        chosen,kind = {'context':heads[key]},'context_'+suffix
    with rt.context():
        return c.integrate(rt,x,labels,lambda z,t,left,i,j:old.field(rt,chosen,capture,kind,z,t,left))


@torch.inference_mode()
def preflight():
    configure()
    verify(SCREEN)
    rt = c.runtime('raev2')
    heads = load(rt)
    checks = old.checks(rt,{'local':heads['local'],'context':heads['context20k']})
    capture = old.Capture(rt)
    noise,_,labels = c.bank('raev2',SCREEN)
    x,y = c.cuda(noise[:4]),c.cuda(labels[:4])
    calls = Calls(rt,heads)
    rows = []
    for suffix in ('base','half'):
        calls.reset()
        z,counts = trajectory(rt,heads,capture,x,y,'context3k_'+suffix)
        pixels = rt.decode(z)
        files = list((OLD_SCREEN/('context_'+suffix)).glob('rank*/batch0000.npz'))
        assert len(files) == 1
        with np.load(files[0]) as data:
            np.testing.assert_array_equal(z.cpu().numpy(),data['latents'])
            np.testing.assert_array_equal(pixels,data['arr_0'])
        expected = sum(bool(c.amount(rt,float(t),'ig')) for t in rt.grid[:-1])
        assert counts == dict(full=100,prefix=0)
        assert calls.blocks == [100]*30
        assert calls.heads == {'context3k':expected,'context20k':0,'local':0}
        rows.append(dict(arm='context3k_'+suffix,samples=4,latents_and_pixels_bit_exact=True,
            block_calls=list(calls.blocks),head_calls=dict(calls.heads)))
    rt.labels = y
    for time_value in (.9,.5,.15):
        t = torch.tensor(time_value,device='cuda')
        with rt.context():
            rt.pair(x,t)
            captured = heads['context20k'](capture.values['context'],capture.values['condition'])
            direct = old.features(rt,x,rt.times(x,t),y)
            predicted = heads['context20k'](direct['context'],direct['condition'])
        assert torch.equal(captured,predicted) and torch.isfinite(captured).all()
    calls.close()
    capture.close()
    c.atomic(ROOT/'preflight.json',dict(passed=True,original_checks=checks,old_3k_replays=rows,
        new_head_direct_and_shared_exact=True,trainable_strong_parameters=False,quality_images_generated=0))
    print('Preflight passed: old 3K replay exact for 8 images, new head shared features exact',flush=True)


@torch.inference_mode()
def worker(stage,rank,world,parent):
    configure()
    request = verify(stage)
    assert c.read(ROOT/'preflight.json')['passed']
    rt = c.runtime('raev2')
    heads = load(rt)
    capture = old.Capture(rt)
    calls = Calls(rt,heads)
    noise,_,labels = c.bank('raev2',stage)
    n = request['samples']
    rh = c.sha(ROOT/'raev2'/stage/'request.json')
    expected = sum(bool(c.amount(rt,float(t),'ig')) for t in rt.grid[:-1])
    assert rt.batch == 4
    for arm in request['arms']:
        output = ROOT/'raev2'/stage/arm/f'rank{rank}'
        for start in range(rank*4,n,world*4):
            c.check_parent(parent)
            path = output/f'batch{start:04d}.npz'
            if path.exists():
                meta = c.read(path.with_suffix('.json'))
                assert c.sha(path) == meta['sha256'] and meta['request_sha256'] == rh
                continue
            x,y = c.cuda(noise[start:start+4]),c.cuda(labels[start:start+4])
            calls.reset()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            z,counts = trajectory(rt,heads,capture,x,y,arm)
            pixels = rt.decode(z)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-begin
            assert counts == dict(full=100,prefix=0) and calls.blocks == [100]*30
            expected_heads = {k:0 for k in heads}
            if not arm.startswith('native'):
                expected_heads[arm.rsplit('_',1)[0]] = expected
            assert calls.heads == expected_heads
            c.save_batch(path,pixels,z.cpu().numpy(),y.cpu().numpy(),
                dict(seconds=seconds,full_calls=100,prefix_calls=0,block_calls=np.array(calls.blocks),
                    head_calls=np.array([calls.heads[k] for k in heads])),rh,start,c.array_sha(noise[start:start+len(y)]))
            if start % (10*4*world) == rank*4:
                c.atomic(ROOT/'raev2'/stage/f'progress{rank}.json',dict(pid=os.getpid(),arm=arm,start=start,total=n))
        c.atomic(output/'complete.json',dict(complete=True,request_sha256=rh))
        print(stage,rank,arm,'complete',flush=True)
    calls.close()
    capture.close()


def collect(stage,arm):
    root = ROOT/'raev2'/stage/arm
    if (root/'summary.json').exists():
        return c.read(root/'summary.json')
    request = c.read(root.parent/'request.json')
    files = sorted(root.glob('rank*/batch*.npz'),key=lambda p:int(c.read(p.with_suffix('.json'))['start']))
    if sum(len(np.load(p)['labels']) for p in files) != request['samples']:
        return None
    noise,_,labels = c.bank('raev2',stage)
    images,records,coverage = [],[],[]
    seconds = 0.
    rh = c.sha(root.parent/'request.json')
    for path in files:
        meta = c.read(path.with_suffix('.json'))
        assert c.sha(path) == meta['sha256'] and meta['request_sha256'] == rh
        with np.load(path) as data:
            start,n = int(data['start']),len(data['labels'])
            coverage.extend(range(start,start+n))
            np.testing.assert_array_equal(data['labels'],labels[start:start+n])
            assert str(data['noise_sha256']) == c.array_sha(noise[start:start+n])
            assert str(data['request_sha256']) == rh
            assert np.isfinite(data['latents']).all()
            assert int(data['full_calls']) == 100 and int(data['prefix_calls']) == 0
            assert np.array_equal(data['block_calls'],np.full(30,100))
            images.append(data['arr_0'])
            seconds += float(data['seconds'])
        records.append(dict(file=str(path),sha256=meta['sha256']))
    assert coverage == list(range(request['samples']))
    np.savez(root/'samples.npz',arr_0=np.concatenate(images))
    summary = dict(complete=True,model='raev2',stage=stage,arm=arm,primary_samples=request['samples'],
        generated_paths=request['samples'],seconds=seconds,full_calls_per_output=100,prefix_calls_at_inference=0,
        samples_sha256=c.sha(root/'samples.npz'),records=records)
    c.atomic(root/'summary.json',summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--stage',choices=(SCREEN,CONFIRM),default=SCREEN)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=3)
    parser.add_argument('--parent',type=int,default=0)
    args = parser.parse_args()
    if args.preflight:
        preflight()
    else:
        worker(args.stage,args.rank,args.world,args.parent)
