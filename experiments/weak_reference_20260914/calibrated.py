"""State-independent structured smoothing with held-out DSM scalar calibration.

The spatial kernel is a fixed circular convolution with unit coordinate
variance. The band contrast smooths the score at two input-space scales;
its expectation suppresses the highest frequencies of the score field.
Neither operator is claimed to be a new mathematical smoothing operator.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from experiments.weak_reference_20260914 import sampler as plain

SOURCE_FILES=[Path(__file__),*plain.SOURCE_FILES]


def structure(probe,kernel):
    if kernel in ('white','band'):return probe
    if kernel!='lowpass':raise ValueError(kernel)
    line=probe.new_tensor([1.,2.,1.])
    filt=(line[:,None]*line[None,:])/6.  # sum of squared 2D coefficients = 1.
    weights=filt[None,None].expand(probe.shape[1],1,3,3)
    return F.conv2d(F.pad(probe,(1,1,1,1),mode='circular'),weights,groups=probe.shape[1])


def residual(rt,z,t,labels,full,probe,kernel):
    offset=.2*(1-t)*structure(probe,kernel)
    plus=plain.sg.query(rt,z+offset,t,labels);minus=plain.sg.query(rt,z-offset,t,labels)
    weak=.5*(plus.float()+minus.float())
    if kernel!='band':return full.float()-weak
    plus2=plain.sg.query(rt,z+2*offset,t,labels);minus2=plain.sg.query(rt,z-2*offset,t,labels)
    return weak-.5*(plus2.float()+minus2.float())


@torch.inference_mode()
def sample(rt,noise,labels,config,snapshots=False):
    if config['kind']!='calibrated':return plain.sample(rt,noise,labels,config,snapshots)
    weights=np.asarray(config['weights'])
    if weights.shape!=(8,) or not np.isfinite(weights).all() or weights.min()<0 or weights.max()>3:
        raise ValueError(weights)
    if config.get('solver','euler')!='euler':raise ValueError(config)
    z=noise.clone();steps=int(config.get('steps',64));before=rt.counts.copy()
    generators=plain.probe_generators(noise,config.get('probe_seed',2026091401))
    expected=0;trace=[];saved={'step_000':z.clone()} if snapshots else None
    with rt.context():
        for k in range(steps):
            t=k/steps;w=float(weights[min(int(t*8),7)])
            v,full,_,_=plain.sg.velocity(rt,z,t,labels,dict(config,kind='baseline',omega=0))
            expected+=1+int(bool(config.get('alpha',0)) and t<config.get('cutoff',.75))
            probe=torch.stack([torch.randn(z.shape[1:],device=z.device,dtype=z.dtype,generator=g) for g in generators])
            delta=torch.zeros_like(z)
            if w:
                delta=w*residual(rt,z,t,labels,full,probe,config['kernel'])
                expected+=4 if config['kernel']=='band' else 2
            z=z+(v+delta)/steps
            if not torch.isfinite(z).all():raise FloatingPointError((config['arm'],k))
            if snapshots:
                rms=lambda x:x.square().flatten(1).mean(1).sqrt()
                constant=lambda value:z.new_full((len(z),),value)
                trace.append(torch.stack((constant(t),constant(w),rms(z),rms(delta)),-1))
                if k+1 in {steps//4,steps//2,3*steps//4,steps}:saved[f'step_{k+1:03d}']=z.clone()
    counts={key:rt.counts[key]-before[key] for key in ('full','prefix')}
    assert counts==dict(full=expected,prefix=0)
    result=dict(latents=z,counts=counts)
    if snapshots:result.update(trace=torch.stack(trace).cpu().numpy(),snapshots=saved)
    return result
