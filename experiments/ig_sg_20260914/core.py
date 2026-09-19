"""Add a raw-strong SG residual to a fixed native internal-guidance field.

SiT uses the established Heun64 IG window; JiT/RAEv2 use native Euler100.
No CFG is added. The SG residual is never formed from an already guided field.
"""
import math
import numpy as np
import torch
from experiments import lifting_scale_sweep_20260909 as common
from experiments.weak_reference_20260914.sampler import probe_generators,pair_residual

ROOT=common.EXPS/'ig_sg_20260914'
SHAPES={'sit_small':(4,32,32),'jit':(3,256,256),'raev2':(1024,16,16)}


class Runtime:
    def __init__(self,name):
        self.name=name;self.rt=common.Runtime(name);self.model=self.rt.model
        self.observed_queries=0
        # First transformer block is shared by the full and internal predictions.
        blocks=self.model.blocks if hasattr(self.model,'blocks') else self.model.trunk
        self.counter=blocks[0].register_forward_pre_hook(self.count)
    def count(self,module,args):self.observed_queries+=1
    @property
    def calls(self):return self.rt.counts['full']
    def grid(self,steps,device):
        if self.name=='raev2':return self.rt.native.shifted_time_grid(steps,8.,device)
        return torch.linspace(0,1,steps+1,device=device)
    def query(self,z,t,labels,need_weak=True):
        self.rt.labels=labels
        if need_weak:return self.rt.pair(z,t)
        return self.rt.field(z,t,'full'),None
    def context(self):return self.rt.context()
    def decode(self,z):return self.rt.decode(z)
    def close(self):self.counter.remove()


def alpha(name,left):
    if name=='sit_small':return .8*(6/7 if left<.25 else 1.) if left<.5 else 0.
    if name=='jit':return .3 if left<.5 else 0.
    return .78 if .1<=left<=1. else 0.


def mix(name,full,weak,a):
    if not a:return full
    if name=='raev2':return weak+(1+a)*(full-weak)
    return full+a*(full-weak)


def configs(name):
    steps=64 if name=='sit_small' else 100
    solver='heun' if name=='sit_small' else 'euler'
    return [dict(arm='ig',kind='baseline',omega=0,steps=steps,solver=solver),
        dict(arm='ig_sg_w1',kind='sg',omega=1,shift=.01,steps=steps,solver=solver),
        dict(arm='ig_log_k02_w1',kind='log',omega=1,kappa=.2,steps=steps,solver=solver),
        dict(arm='ig_sg_cost',kind='baseline',omega=0,steps=128 if name=='sit_small' else 199,solver=solver),
        dict(arm='ig_log_cost',kind='baseline',omega=0,steps=191 if name=='sit_small' else 300,solver=solver)]


@torch.inference_mode()
def sample(rt,noise,labels,config,trace=False):
    kind=config['kind'];omega=float(config.get('omega',0));steps=int(config['steps'])
    if kind not in ('baseline','sg','log') or config['solver'] not in ('euler','heun'):raise ValueError(config)
    if steps<2 or noise.dtype!=torch.float32 or labels.dtype!=torch.long:raise ValueError(config)
    for value in (omega,float(config.get('kappa',.2)),float(config.get('shift',.01))):
        if not math.isfinite(value) or value<0:raise ValueError(config)
    if kind=='baseline' and omega:raise ValueError(config)
    if (kind=='log' and not config.get('kappa',.2)) or (kind=='sg' and not config.get('shift',.01)):omega=0.
    generators=probe_generators(noise,config.get('probe_seed',2026091401)) if kind=='log' and omega else None
    before=rt.calls;observed=rt.observed_queries;expected=0;rows=[];z=noise.clone()
    grid=rt.grid(steps,z.device)
    def field(state,t,a,stage):
        nonlocal expected
        full,weak=rt.query(state,t,labels,need_weak=bool(a));expected+=1
        value=mix(rt.name,full,weak,a);delta=torch.zeros_like(full)
        if omega and kind=='sg':
            ref=(t+config.get('shift',.01)).clamp_max(1) if rt.name=='raev2' else (t-config.get('shift',.01)).clamp_min(0)
            if not torch.equal(ref,t):
                other,_=rt.query(state,ref,labels,need_weak=False);expected+=1
                delta=omega*(full-other)
        elif generators is not None:
            probe=torch.stack([torch.randn(z.shape[1:],device=z.device,dtype=z.dtype,generator=g) for g in generators])
            sigma=config.get('kappa',.2)*(t if rt.name=='raev2' else 1-t)
            if float(sigma)>0:
                plus,_=rt.query(state+sigma*probe,t,labels,need_weak=False)
                minus,_=rt.query(state-sigma*probe,t,labels,need_weak=False);expected+=2
                delta=omega*pair_residual(full,plus,minus)
        if trace:
            rms=lambda x:x.float().square().flatten(1).mean(1).sqrt()
            const=lambda x:state.new_full((len(state),),x)
            rows.append(torch.stack((const(float(t)),const(a),const(stage),rms(state),rms(delta)),1).cpu().numpy())
        return value+delta if omega else value
    with rt.context():
        for t,u in zip(grid[:-1],grid[1:]):
            a=alpha(rt.name,float(t));h=u-t
            first=field(z,t,a,0)
            if config['solver']=='heun':
                second=field(z+h*first,u,a,1)  # Hold native IG gate at the left stage.
                z=z+(h/2)*(first+second)
            else:z=z+h*first
            if not torch.isfinite(z).all() or z.abs().max()>1e6:raise FloatingPointError(config)
    actual=rt.calls-before
    assert actual==expected==rt.observed_queries-observed,(actual,expected,rt.observed_queries-observed)
    assert rt.rt.counts['prefix']==0
    return z,dict(full=actual,prefix=0,trace=np.stack(rows) if rows else np.empty((0,)))
