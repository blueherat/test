"""Experimental angular log-density weak reference with score pullback.

At fixed t, T_+=(cos a I + sin a J), T_-=T_+^T, J^T=-J, J^2=-I.
q(x) is proportional to exp((log p(mu+T_+(x-mu)) + log p(mu+T_-(x-mu)))/2).
Its score MUST include T_+^T and T_-^T. Every query preserves ||x-mu||.
The prototype fixes spherical Gaussians around the supplied class mean;
general covariance invariance additionally requires whitening (toy checked).
"""
from pathlib import Path
from functools import lru_cache
import math
import numpy as np
import torch
from experiments.weak_reference_20260914 import sampler as plain

MEANS=Path('/home/zhoushunyu/data/eqvae/experiments/sit_measure_guidance_20260912/class_means.npy')
SOURCE_FILES=[Path(__file__),Path(__file__).with_name('angular_check.py'),MEANS,*plain.SOURCE_FILES]


@lru_cache(maxsize=1)
def means(device):
    return torch.from_numpy(np.load(MEANS)).to(device)


def j_apply(x,permutation):
    flat=x.flatten(1);pairs=flat[:,permutation].reshape(len(x),-1,2)
    rotated=torch.stack((-pairs[:,:,1],pairs[:,:,0]),-1).flatten(1)
    out=torch.empty_like(flat);out[:,permutation]=rotated
    return out.reshape_as(x)


def rotate(x,perm,angle):
    return math.cos(angle)*x+math.sin(angle)*j_apply(x,perm)


def correction(rt,z,t,labels,full,perm,angle,center):
    mu=t*center
    zp=mu+rotate(z-mu,perm,angle);zm=mu+rotate(z-mu,perm,-angle)
    vp=plain.sg.query(rt,zp,t,labels);vm=plain.sg.query(rt,zm,t,labels)
    pulled=.5*(rotate(vp,perm,-angle)+rotate(vm,perm,angle))
    # v=z/t+(1-t)*score/t; affine mean contribution has this nonsingular limit.
    return full-pulled+(math.cos(angle)-1)*center


@torch.inference_mode()
def sample(rt,noise,labels,config,snapshots=False):
    if config['kind']!='angular':return plain.sample(rt,noise,labels,config,snapshots)
    angle0=float(config.get('angle',.2));omega=float(config.get('omega',1))
    if not math.isfinite(angle0) or not 0<=angle0<=.5 or not math.isfinite(omega) or omega<0:
        raise ValueError(config)
    if config.get('solver','euler')!='euler':raise ValueError(config)
    if angle0==0 or omega==0:
        return plain.sample(rt,noise,labels,dict(config,kind='baseline',omega=0),snapshots)
    z=noise.clone();steps=int(config.get('steps',64));before=rt.counts.copy()
    center=means(str(z.device))[labels]
    generator=torch.Generator(device=z.device).manual_seed(config.get('rotation_seed',2026091421))
    trace=[];saved={'step_000':z.clone()} if snapshots else None
    with rt.context():
        for k in range(steps):
            t=k/steps;angle=angle0*(1-t)
            v,full,_,_=plain.sg.velocity(rt,z,t,labels,dict(config,kind='baseline',omega=0))
            # Rotation is independent of x, common across images, freshly drawn each step.
            perm=torch.randperm(z[0].numel(),device=z.device,generator=generator)
            delta=omega*correction(rt,z,t,labels,full,perm,angle,center)
            z=z+(v+delta)/steps
            if not torch.isfinite(z).all():raise FloatingPointError((config['arm'],k))
            if snapshots:
                rms=lambda x:x.square().flatten(1).mean(1).sqrt()
                constant=lambda value:z.new_full((len(z),),value)
                trace.append(torch.stack((constant(t),constant(angle),rms(z),rms(delta)),-1))
                if k+1 in {steps//4,steps//2,3*steps//4,steps}:saved[f'step_{k+1:03d}']=z.clone()
    counts={key:rt.counts[key]-before[key] for key in ('full','prefix')}
    expected=3*steps+bool(config.get('alpha',0))*sum(k/steps<config.get('cutoff',.75) for k in range(steps))
    assert counts==dict(full=expected,prefix=0)
    result=dict(latents=z,counts=counts)
    if snapshots:result.update(trace=torch.stack(trace).cpu().numpy(),snapshots=saved)
    return result
