"""Paired 5K SSG evaluation using the author's 50-step Heun/Euler convention."""
import argparse
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace, MethodType

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915 import config as k
from experiments.guidance_dynamic_50k_20260915.models import fingerprint
from .jit_ssg import Runtime, ROOT, LITERATURE
from .sampler import prepare_adapter


class Field:
    def __init__(self, runtime, mode, w, cfg):
        self.runtime,self.mode,self.w,self.cfg=runtime,mode,w,cfg

    def __call__(self,z,t,labels):
        t=t.expand(len(z)); tv=t[:,None,None,None]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if self.mode=='weak':
                prediction=self.runtime.net.forward_intermediate(z,t,labels)
                return (prediction-z)/(1-tv).clamp_min(.05)
            cond,weak=self.runtime.net.forward_with_intermediate(z,t,labels)
            active=(tv<1) & ((tv>.1) if self.cfg!=1 else torch.ones_like(tv,dtype=torch.bool))
            scale=torch.full_like(tv,self.w)
            clean=torch.where(active,weak+scale*(cond-weak),cond) if self.w!=1 else cond
            velocity=(clean-z)/(1-tv).clamp_min(.05)
            if self.cfg==1: return velocity
            uncond,uw=self.runtime.net.forward_with_intermediate(z,t,torch.full_like(labels,1000))
            uc=torch.where(active,uw+scale*(uncond-uw),uncond) if self.w!=1 else uncond
            uv=(uc-z)/(1-tv).clamp_min(.05)
            return torch.where(active,uv+self.cfg*(velocity-uv),velocity)


def optimize_constants(runtime):
    # Reuse the audited constant cache and FP32-attention implementation. The
    # backbone view excludes the separately registered, frozen adapter.
    from types import SimpleNamespace
    from experiments import jit_internal_guidance as jig
    view=SimpleNamespace(name='jit',model=runtime.net,jig=jig)
    prepare_adapter(view,runtime.head,torch.device('cuda'))


def official_field(runtime,w,cfg):
    # Call the published sampler method directly as an independent reference.
    if str(LITERATURE) not in sys.path:sys.path.insert(0,str(LITERATURE))
    spec=importlib.util.spec_from_file_location('eqvae_ssg_denoiser_reference',LITERATURE/'denoiser_ssg.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obj=SimpleNamespace(net=runtime.net,ssg_scale=w,ssg_interval=(0. if cfg==1 else .1,1.),
        cfg_scale=cfg,cfg_interval=(.1,1.),t_eps=.05,num_classes=1000)
    obj._apply_ssg=MethodType(module.DenoiserSSG._apply_ssg,obj)
    def forward(z,t,y):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return module.DenoiserSSG._forward_sample(obj,z,t.expand(len(z))[:,None,None,None],y)
    return forward


class GraphField:
    def __init__(self,field,z,y):
        self.x=z.clone();self.t=torch.full((len(z),),.2,device='cuda');self.y=y.clone()
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for _ in range(3):field(self.x,self.t,self.y)
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.out=field(self.x,self.t,self.y)
        torch.cuda.current_stream().wait_stream(stream)

    def __call__(self,x,t,y):
        self.x.copy_(x);self.t.copy_(t);self.y.copy_(y);self.graph.replay()
        return self.out.clone()


@torch.no_grad()
def integrate(field,z,y):
    grid=torch.linspace(0,1,51,device='cuda');state=z.clone()
    for i in range(50):
        t,u=grid[i],grid[i+1]
        first=field(state,t,y)
        if i<49:
            second=field(state+(u-t)*first,u,y)
            state=state+(u-t)*(.5*(first+second))
        else:state=state+(u-t)*first
    return state


@torch.no_grad()
def main(args):
    _,world=c.setup();assert world==1
    run=args.output;request=c.read(run/'request.json')
    for path,digest in request['sources'].items():
        if c.sha(path)!=digest:raise RuntimeError(f'Source changed: {path}')
    torch.set_float32_matmul_precision('high')
    rt=Runtime(args.blocks)
    checkpoint_sha=None
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        assert state['config']['blocks']==args.blocks and state['frozen']==rt.frozen_hash()
        if not args.audit_only:assert state['step']==50000
        rt.head.load_state_dict(state['ema']);checkpoint_sha=c.sha(args.checkpoint)
    elif args.mode!='baseline':raise ValueError('A weak-head checkpoint is required')
    rt.net.eval().requires_grad_(False);rt.head.eval().requires_grad_(False)
    frozen=rt.frozen_hash();head_hash=fingerprint(rt.head)
    bank=k.model_root('jit')/'quality_inputs'
    noise=np.load(bank/'noise.npy',mmap_mode='r');labels=np.load(bank/'labels.npy')
    assert len(noise)==len(labels)==5000 and np.all(np.bincount(labels,minlength=1000)==5)
    z=torch.from_numpy(np.array(noise[:8])).cuda();y=torch.from_numpy(labels[:8].copy()).cuda()
    field=Field(rt,args.mode,args.w,args.cfg)
    reference=integrate(official_field(rt,args.w,args.cfg) if args.mode!='weak' else field,z,y)
    eager=integrate(field,z,y)
    torch.testing.assert_close(eager,reference,rtol=0,atol=0)
    optimize_constants(rt);graph=GraphField(field,z,y)
    actual=integrate(graph,z,y)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    c.atomic(run/'parity.json',dict(passed=True,unoptimized_endpoint_exact=True,
        published_ssg_sampling_endpoint_exact=args.mode!='weak',
        solver='49 Heun updates plus final Euler, 50 intervals',batch=8))
    if args.audit_only:
        c.atomic(run/'complete.json',dict(complete=True,audit_only=True));return
    pixels=np.lib.format.open_memmap(run/'pixels.npy',mode='w+',dtype=np.uint8,shape=(5000,256,256,3))
    records=[]
    for start in range(0,5000,8):
        z=torch.from_numpy(np.array(noise[start:start+8])).cuda()
        y=torch.from_numpy(labels[start:start+8].copy()).cuda()
        endpoint=integrate(graph,z,y)
        if not torch.isfinite(endpoint).all():raise FloatingPointError(f'Endpoint at {start}')
        pixels[start:start+8]=np.round(np.clip((endpoint.float().cpu().numpy().transpose(0,2,3,1)+1)/2*255,0,255)).astype(np.uint8)
        records.append(dict(start=start,samples=8,labels=y.cpu().tolist(),noise_sha256=c.original.array_sha(noise[start:start+8])))
        if start%80==0:c.atomic(run/'progress.json',dict(samples=start+8,total=5000,updated_utc=c.now()))
    pixels.flush();sample_path=run/'samples.npz'
    with sample_path.open('wb') as f:np.savez(f,arr_0=pixels)
    del pixels
    summary=dict(complete=True,valid=True,n=5000,blocks=args.blocks,mode=args.mode,w=args.w,extra_a=args.w-1,cfg=args.cfg,
        samples_path=str(sample_path),samples_sha256=c.sha(sample_path),checkpoint_sha256=checkpoint_sha,
        noise_sha256=c.sha(bank/'noise.npy'),labels_sha256=c.sha(bank/'labels.npy'),records=records,
        strong_unchanged=rt.frozen_hash()==frozen,head_unchanged=fingerprint(rt.head)==head_hash,
        solver='50 intervals: Heun except final Euler',ssg_interval='full' if args.cfg==1 else '(.1,1)',
        precision='published BF16 autocast',reference=str(args.reference),reference_sha256=c.sha(args.reference))
    assert summary['strong_unchanged'] and summary['head_unchanged']
    c.atomic(run/'summary.json',summary)
    del graph,field,rt;torch.cuda.empty_cache()
    cmd=[c.PYTHON,str(c.WORK/'experiments/evaluate_raev2_official_samples.py'),
        '--branch',f'ssg={sample_path}','--output',str(run/'official.csv'),
        '--batch-size','64','--device','cuda','--fid-reference',str(args.reference),
        '--feature-cache-dir',str(run/'feature_cache')]
    with (run/'evaluation.log').open('w') as f:subprocess.run(cmd,cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,check=True)
    result=c.read(run/'official.json')[0]
    assert np.isfinite(result['fid']) and np.isfinite(result['inception_score'])
    c.atomic(run/'metrics.json',dict(summary,fid=result['fid'],inception_score=result['inception_score']))
    c.atomic(run/'complete.json',dict(complete=True,fid=result['fid']))
    c.atomic(run/'progress.json',dict(samples=5000,total=5000,complete=True,updated_utc=c.now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('jit',),default='jit')
    p.add_argument('--blocks',type=int,choices=(0,1,2),required=True)
    p.add_argument('--mode',choices=('baseline','guided','weak'),default='guided')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--w',type=float,default=1.7)
    p.add_argument('--cfg',type=float,default=1.)
    p.add_argument('--reference',type=Path,default=ROOT/'literature/fid_stats/jit_in256_stats.npz')
    p.add_argument('--audit-only',action='store_true')
    main(p.parse_args())
