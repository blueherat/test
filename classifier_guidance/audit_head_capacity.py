"""Matched held-out prediction errors and head-only generation quality."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.train_imagenet100_sit_flow import sample_sdvae_posterior
from experiments.lifting_scale_sweep_20260909 import SMALL_HEAD
from .capacity_heads import make
from .sampler import prepare_adapter

ROOT = Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_mlp_capacity_diffusion_20260920')


def load_heads(adapter):
    heads = dict(native=adapter.native.eval().requires_grad_(False),
                 shallow=adapter.loaded_head('real'))
    paths = dict(native=Path(SMALL_HEAD), shallow=k.model_root('sit_small')/'training/real/head.pt')
    for variant in ('moderate', 'large'):
        path = ROOT/variant/'training_50k/checkpoint_050000.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['step'] == 50000 and state['frozen'] == fingerprint(adapter.model)
        head = make(variant).cuda().eval().requires_grad_(False)
        head.load_state_dict(state['ema'], strict=True)
        heads[variant] = head; paths[variant] = path
    provenance = {name: dict(path=str(path), sha256=c.sha(path), parameters=sum(p.numel() for p in heads[name].parameters()))
                  for name, path in paths.items()}
    return heads, provenance


@torch.no_grad()
def predictions(adapter, heads, provenance, run):
    moments = torch.from_numpy(np.array(np.load(k.SIT_DATA/'validation_moments.npy', mmap_mode='r'))).cuda()
    labels = torch.from_numpy(np.load(k.SIT_DATA/'validation_labels.npy')).long().cuda()
    assert len(labels) == 5000
    g = torch.Generator(device='cuda').manual_seed(2026092091)
    clean = sample_sdvae_posterior(moments, torch.randn((5000,4,32,32), generator=g, device='cuda'))
    del moments
    names = ['strong', *heads]
    errors = {name: np.zeros((10, 5000), dtype=np.float64) for name in names}
    gaps = {name: np.zeros((10, 5000), dtype=np.float64) for name in heads}
    bins = []
    for bin_index in range(10):
        for start in range(0, 5000, 32):
            x = clean[start:start+32]; y = labels[start:start+32]
            eps = torch.randn(x.shape, generator=g, device='cuda')
            t = (bin_index + torch.rand(len(x), generator=g, device='cuda'))/10
            z = t[:,None,None,None]*x + (1-t[:,None,None,None])*eps
            target = adapter.patch(x-eps)
            strong, _ = adapter.full(z,t,y)
            strong = adapter.patch(strong)
            features = dict(adapter.values)
            predictions = dict(strong=strong)
            for name, head in heads.items():
                predictions[name] = head(features['context'],features['condition']).float()
            for name, pred in predictions.items():
                errors[name][bin_index,start:start+len(x)] = (pred-target).square().flatten(1).mean(1).double().cpu().numpy()
                if name in gaps:
                    gaps[name][bin_index,start:start+len(x)] = (pred-strong).square().flatten(1).mean(1).double().cpu().numpy()
            adapter.values.clear()
        row = dict(time_bin=[bin_index/10,(bin_index+1)/10], mse={name:float(v[bin_index].mean()) for name,v in errors.items()})
        bins.append(row); c.atomic(run/'progress.json',dict(completed_bins=len(bins),total_bins=10,latest=row))
    result = dict(complete=True, real_validation_images=5000, time_strata=10, total_states=50000,
        seed=2026092091, precision='deployed FP32/TF32; same features, targets, labels and noises for every head',
        held_out=True, provenance=provenance, bins=bins,
        models={name:dict(velocity_mse=float(e.mean()), early_mse=float(e[:5].mean()), late_mse=float(e[5:].mean()),
                         image_mean_sem=float(e.mean(0).std(ddof=1)/np.sqrt(5000))) for name,e in errors.items()},
        paired_difference_from_shallow={name:dict(mean=float((e-errors['shallow']).mean()),
            image_mean_sem=float((e-errors['shallow']).mean(0).std(ddof=1)/np.sqrt(5000))) for name,e in errors.items()},
        strong_gap_mse={name:float(value.mean()) for name,value in gaps.items()})
    np.savez(run/'per_image_errors.npz',**errors,**{name+'_strong_gap':value for name,value in gaps.items()})
    c.atomic(run/'result.json',result)
    print(json.dumps(result['models']),flush=True)


class WeakForward:
    def __init__(self, adapter, head, example, labels):
        self.adapter, self.head = adapter, head
        self.x=example.clone(); self.t=torch.full((len(example),),.25,device='cuda'); self.y=labels.clone()
        prepare_adapter(adapter,head,example.device)
        stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3): self.eager(self.x,self.t,self.y)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream): self.output=self.eager(self.x,self.t,self.y)
        torch.cuda.current_stream().wait_stream(stream)

    def eager(self,x,t,y):
        value=self.adapter.features(x,t.expand(len(x)),y)
        return self.adapter.unpatch(self.head(value['context'],value['condition'])).float()

    def __call__(self,x,t,y):
        self.x.copy_(x);self.t.copy_(t);self.y.copy_(y)
        self.graph.replay();return self.output.clone()


@torch.no_grad()
def integrate_weak(field,z,y):
    state=z.clone();grid=torch.linspace(0,1,65,device='cuda')
    for t,u in zip(grid[:-1],grid[1:]):
        first=field(state,t,y); second=field(state+(u-t)*first,u,y)
        state=state+(u-t)/2*(first+second)
    return state


@torch.no_grad()
def generate(adapter,head,provenance,run,name):
    bank=k.model_root('sit_small')/'quality_inputs'
    noise=np.load(bank/'noise.npy',mmap_mode='r');labels=np.load(bank/'labels.npy')
    z=torch.from_numpy(np.array(noise[:8])).cuda();y=torch.from_numpy(labels[:8].copy()).cuda()
    field=WeakForward(adapter,head,z,y)
    actual=integrate_weak(field,z,y); reference=integrate_weak(field.eager,z,y)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    c.atomic(run/'parity.json',dict(passed=True,weak_only_heun_exact=True))
    path=run/'pixels.npy'
    pixels=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(5000,256,256,3))
    records=[]
    for start in range(0,5000,8):
        begin=time.perf_counter()
        z=torch.from_numpy(np.array(noise[start:start+8])).cuda();y=torch.from_numpy(labels[start:start+8].copy()).cuda()
        endpoint=integrate_weak(field,z,y)
        if not torch.isfinite(endpoint).all():raise FloatingPointError(f'Nonfinite weak endpoint: {name} at {start}')
        pixels[start:start+8]=adapter.pixels(endpoint)
        records.append(dict(start=start,samples=8,noise_sha256=c.original.array_sha(noise[start:start+8]),
                            labels=y.cpu().tolist(),seconds=time.perf_counter()-begin))
        if start%80==0:c.atomic(run/'progress.json',dict(samples=start+8,total=5000,updated_utc=c.now()))
        adapter.values.clear()
    pixels.flush()
    sample_path=run/'samples.npz'
    with sample_path.open('wb') as f:np.savez(f,arr_0=pixels)
    del pixels
    c.atomic(run/'summary.json',dict(complete=True,valid=True,n=5000,variant=name,mode='weak_only',
        samples_path=str(sample_path),samples_sha256=c.sha(sample_path),records=records,
        provenance=provenance,noise_sha256=c.sha(bank/'noise.npy'),labels_sha256=c.sha(bank/'labels.npy'),
        solver='64-step Heun, batch8',precision='FP32/TF32 as deployed',
        field='W alone for entire trajectory; no strong field, CFG or IG extrapolation'))
    command=[c.PYTHON,'-u','-m','experiments.adversarial_weak_training_20260915.score',
             '--stage',str(run),'--shared-gpu',os.environ['CUDA_VISIBLE_DEVICES']]
    torch.cuda.empty_cache()
    with (run/'score_holder.log').open('w') as f:
        subprocess.run(command,cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,check=True)
    c.atomic(run/'complete.json',dict(complete=True,fid=c.read(run/'metrics.json')['fid'],updated_utc=c.now()))
    c.atomic(run/'progress.json',dict(samples=5000,total=5000,complete=True,updated_utc=c.now()))


def main(args):
    rank,world=c.setup()
    if world!=1:raise ValueError('Use one GPU per independent audit')
    request=c.read(args.output/'request.json')
    for path,digest in request['sources'].items():
        if c.sha(path)!=digest:raise RuntimeError(f'Source changed: {path}')
    adapter=Adapter('sit_small');heads,provenance=load_heads(adapter)
    signatures={name:fingerprint(head) for name,head in heads.items()};strong=fingerprint(adapter.model)
    if args.mode=='prediction':predictions(adapter,heads,provenance,args.output)
    else:generate(adapter,heads[args.variant],provenance[args.variant],args.output,args.variant)
    assert fingerprint(adapter.model)==strong
    assert signatures=={name:fingerprint(head) for name,head in heads.items()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small',),default='sit_small')
    p.add_argument('--mode',choices=('prediction','weak-only'),required=True)
    p.add_argument('--variant',choices=('native','shallow','moderate','large'),default='shallow')
    main(p.parse_args())
