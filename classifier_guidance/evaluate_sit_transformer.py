"""SiT adapter comparison: paired prediction errors, full/legacy guidance, W-only FID."""
import argparse
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.adversarial_weak_training_20260915 import common as c
from experiments.guidance_dynamic_50k_20260915.models import Adapter, fingerprint
from experiments.guidance_dynamic_50k_20260915.sampling import integrate as original_integrate
from .capacity_heads import make
from .sampler import prepare_adapter
from .audit_head_capacity import predictions

ROOT = Path('/home/zhoushunyu/data/eqvae/projects/classifier_guidance/sit_transformer_capacity_20260921')
VARIANTS = ('shallow', 'linear', 'block1', 'block2')


def load_head(adapter,path,audit=False):
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['objective']=='capacity_diffusion_only' and state['frozen']==fingerprint(adapter.model)
    assert state['step']==50000 or audit
    head=make(state['config']['variant'],adapter.model).cuda().eval().requires_grad_(False)
    head.load_state_dict(state['ema'],strict=True)
    return head,state


class Field:
    def __init__(self,adapter,head,mode):self.adapter,self.head,self.mode=adapter,head,mode

    def __call__(self,z,t,y,amount,active):
        a=self.adapter
        if self.mode=='weak':
            features=a.features(z,t.expand(len(z)),y)
            return a.unpatch(self.head(features['context'],features['condition'])).float()
        full,_=a.full(z,t,y)
        if not active:return full
        weak=a.unpatch(self.head(a.values['context'],a.values['condition'])).float()
        return full+amount*(full-weak)


class GraphField:
    def __init__(self,field,z,y):
        self.x=z.clone();self.t=torch.full((),.2,device='cuda');self.y=y.clone()
        self.amount=torch.full((),1.,device='cuda');self.captures={}
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.no_grad():
            for active in (False,True):
                for _ in range(3):field(self.x,self.t,self.y,self.amount,active)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):output=field(self.x,self.t,self.y,self.amount,active)
                self.captures[active]=(graph,output)
        torch.cuda.current_stream().wait_stream(stream)

    def __call__(self,z,t,y,amount,active):
        self.x.copy_(z);self.t.copy_(t);self.y.copy_(y);self.amount.fill_(amount)
        graph,output=self.captures[active];graph.replay();return output.clone()


@torch.no_grad()
def integrate(field,z,y,coefficient,window,mode):
    state=z.clone();grid=torch.linspace(0,1,65,device='cuda')
    for i,(t,u) in enumerate(zip(grid[:-1],grid[1:])):
        active=mode=='guided' and coefficient!=0 and (window=='full' or i<32)
        amount=coefficient*(6/7 if window=='legacy' and i<16 else 1.) if active else 0.
        first=field(state,t,y,amount,active)
        second=field(state+(u-t)*first,u,y,amount,active)
        state=state+(u-t)/2*(first+second)
    return state


@torch.no_grad()
def main(args):
    _,world=c.setup();assert world==1
    run=args.output;request=c.read(run/'request.json')
    for path,digest in request['sources'].items():
        if c.sha(path)!=digest:raise RuntimeError(f'Source changed: {path}')
    adapter=Adapter('sit_small');frozen=fingerprint(adapter.model)
    if args.mode=='prediction':
        heads={};provenance={}
        for variant in VARIANTS:
            path=ROOT/variant/'training_50k/checkpoint_050000.pt'
            heads[variant],_=load_head(adapter,path)
            provenance[variant]=dict(path=str(path),sha256=c.sha(path),parameters=sum(p.numel() for p in heads[variant].parameters()))
        signatures={n:fingerprint(h) for n,h in heads.items()}
        predictions(adapter,heads,provenance,run)
        assert fingerprint(adapter.model)==frozen and signatures=={n:fingerprint(h) for n,h in heads.items()}
        c.atomic(run/'complete.json',dict(complete=True,mode='prediction'));return
    if args.checkpoint:
        head,state=load_head(adapter,args.checkpoint,args.audit_only);digest=c.sha(args.checkpoint)
        variant=state['config']['variant']
    else:
        assert args.mode=='baseline'
        variant='shallow';head=make(variant).cuda().eval().requires_grad_(False);digest=None
    signature=fingerprint(head)
    bank=c.original.model_root('sit_small')/'quality_inputs'
    noise=np.load(bank/'noise.npy',mmap_mode='r');labels=np.load(bank/'labels.npy')
    assert len(labels)==5000 and np.all(np.bincount(labels,minlength=100)==50)
    z=torch.from_numpy(np.array(noise[:8])).cuda();y=torch.from_numpy(labels[:8].copy()).cuda()
    field=Field(adapter,head,args.mode)
    try:
        if args.mode=='guided' and args.window=='legacy':
            reference,_=original_integrate(adapter,head,'real',args.coefficient,z,y)
        else:reference=integrate(field,z,y,args.coefficient,args.window,args.mode)
    except FloatingPointError as exc:
        c.atomic(run/'invalid.json',dict(valid=False,start=0,reason=str(exc),coefficient=args.coefficient))
        c.atomic(run/'complete.json',dict(complete=True,valid=False,fid=None));return
    if not torch.isfinite(reference).all() or reference.abs().max()>1e6:
        c.atomic(run/'invalid.json',dict(valid=False,start=0,reason='divergent eager endpoint',coefficient=args.coefficient))
        c.atomic(run/'complete.json',dict(complete=True,valid=False,fid=None));return
    prepare_adapter(adapter,head,torch.device('cuda'))
    graph=GraphField(field,z,y)
    actual=integrate(graph,z,y,args.coefficient,args.window,args.mode)
    torch.testing.assert_close(actual,reference,rtol=0,atol=0)
    c.atomic(run/'parity.json',dict(passed=True,unaccelerated_endpoint_exact=True,
        historical_sampler_reference=args.mode=='guided' and args.window=='legacy',window=args.window,mode=args.mode))
    if args.audit_only:
        c.atomic(run/'complete.json',dict(complete=True,audit_only=True));return
    pixels=np.lib.format.open_memmap(run/'pixels.npy',mode='w+',dtype=np.uint8,shape=(5000,256,256,3))
    records=[]
    for start in range(0,5000,8):
        z=torch.from_numpy(np.array(noise[start:start+8])).cuda();y=torch.from_numpy(labels[start:start+8].copy()).cuda()
        endpoint=integrate(graph,z,y,args.coefficient,args.window,args.mode)
        if not torch.isfinite(endpoint).all() or endpoint.abs().max()>1e6:
            c.atomic(run/'invalid.json',dict(valid=False,start=start,reason='divergent endpoint',coefficient=args.coefficient))
            c.atomic(run/'complete.json',dict(complete=True,valid=False,fid=None));return
        pixels[start:start+8]=adapter.pixels(endpoint)
        records.append(dict(start=start,samples=8,labels=y.cpu().tolist(),noise_sha256=c.original.array_sha(noise[start:start+8])))
        if start%80==0:c.atomic(run/'progress.json',dict(samples=start+8,total=5000,updated_utc=c.now()))
    pixels.flush();sample_path=run/'samples.npz'
    with sample_path.open('wb') as f:np.savez(f,arr_0=pixels)
    del pixels
    assert fingerprint(adapter.model)==frozen and fingerprint(head)==signature
    c.atomic(run/'summary.json',dict(complete=True,valid=True,n=5000,variant=variant,mode=args.mode,window=args.window,
        coefficient=args.coefficient,coefficient_convention='extra a in S+a*f(t)*(S-W)',
        samples_path=str(sample_path),samples_sha256=c.sha(sample_path),records=records,
        checkpoint_sha256=digest,noise_sha256=c.sha(bank/'noise.npy'),labels_sha256=c.sha(bank/'labels.npy'),
        strong_unchanged=True,head_unchanged=True,solver='64-step Heun; shared left-step guidance amount',
        precision='FP32/TF32 as previous SiT experiments'))
    del graph,field,adapter,head;torch.cuda.empty_cache()
    command=[c.PYTHON,'-u','-m','experiments.adversarial_weak_training_20260915.score',
        '--stage',str(run),'--shared-gpu',os.environ['CUDA_VISIBLE_DEVICES']]
    with (run/'score_holder.log').open('w') as f:subprocess.run(command,cwd=c.WORK,stdout=f,stderr=subprocess.STDOUT,check=True)
    c.atomic(run/'complete.json',dict(complete=True,valid=True,fid=c.read(run/'metrics.json')['fid']))
    c.atomic(run/'progress.json',dict(complete=True,samples=5000,total=5000,updated_utc=c.now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=('sit_small',),default='sit_small')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--mode',choices=('guided','weak','baseline','prediction'),default='guided')
    p.add_argument('--window',choices=('legacy','full'),default='legacy')
    p.add_argument('--coefficient',type=float,default=1.)
    p.add_argument('--audit-only',action='store_true')
    main(p.parse_args())
