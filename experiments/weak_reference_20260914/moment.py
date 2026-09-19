"""Direction 2: diagonal-moment Gaussian-convolution weak reference.

Uses training-calibrated class means and per-coordinate variances as estimates
of strong marginal moments. Full cross-coordinate covariance is not estimated.
The score query includes all time/input/output coordinate transformations.
"""
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from experiments.weak_reference_20260914 import sampler as plain

STATS=Path('/home/zhoushunyu/data/eqvae/experiments/weak_reference_20260914/moment_stats.npz')
SOURCE_FILES=[Path(__file__),Path(__file__).with_name('moment_check.py'),STATS,*plain.SOURCE_FILES]


@lru_cache(maxsize=1)
def statistics(device):
    with np.load(STATS) as d:return [torch.from_numpy(d[k]).to(device) for k in ('mean','variance')]


def correction(rt,z,t,labels,full,kappa,mean,variance):
    if t==0:return torch.zeros_like(z)
    beta=1-t;tau=(kappa*beta)**2
    marginal_variance=t*t*variance+beta*beta
    inverse_A=torch.sqrt(1+tau/marginal_variance)
    r=np.sqrt(beta*beta+tau);scale=t+r;ref_t=t/scale
    mu=t*mean;mapped=mu+inverse_A*(z-mu);ref_z=mapped/scale
    reference=plain.sg.query(rt,ref_z.float(),ref_t,labels)
    # Compute the coordinate-canceling terms in FP64 to reduce subtractive loss.
    s=(t*full.double()-z.double())/beta
    weak_score=inverse_A.double()*(ref_t*reference.double()-ref_z.double())/r
    return (beta/t*(s-weak_score)).float()


@torch.inference_mode()
def sample(rt,noise,labels,config,snapshots=False):
    if config['kind']!='moment':return plain.sample(rt,noise,labels,config,snapshots)
    omega=float(config.get('omega',1));kappa=float(config.get('kappa',.2))
    if not np.isfinite([omega,kappa]).all() or min(omega,kappa)<0:raise ValueError(config)
    if config.get('solver','euler')!='euler':raise ValueError(config)
    if omega==0 or kappa==0:return plain.sample(rt,noise,labels,dict(config,kind='baseline',omega=0),snapshots)
    mean,variance=statistics(str(noise.device));mean=mean[labels];variance=variance[labels]
    z=noise.clone();steps=int(config.get('steps',64));before=rt.counts.copy();trace=[]
    saved={'step_000':z.clone()} if snapshots else None
    with rt.context():
        for k in range(steps):
            t=k/steps
            v,full,_,_=plain.sg.velocity(rt,z,t,labels,dict(config,kind='baseline',omega=0))
            delta=omega*correction(rt,z,t,labels,full,kappa,mean,variance)
            z=z+(v+delta)/steps
            if not torch.isfinite(z).all():raise FloatingPointError((config['arm'],k))
            if snapshots:
                rms=lambda x:x.square().flatten(1).mean(1).sqrt()
                trace.append(torch.stack((z.new_full((len(z),),t),rms(z),rms(delta)),-1))
                if k+1 in {steps//4,steps//2,3*steps//4,steps}:saved[f'step_{k+1:03d}']=z.clone()
    counts={key:rt.counts[key]-before[key] for key in ('full','prefix')}
    expected=2*steps-1+bool(config.get('alpha',0))*sum(k/steps<config.get('cutoff',.75) for k in range(steps))
    assert counts==dict(full=expected,prefix=0)
    result=dict(latents=z,counts=counts)
    if snapshots:result.update(trace=torch.stack(trace).cpu().numpy(),snapshots=saved)
    return result
