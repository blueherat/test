"""Same-time antithetic score contrast on JiT and RAEv2 native velocities."""
import math
import numpy as np
import torch
from experiments.self_guidance_cross_model_20260913 import core as native
from experiments.weak_reference_20260914.sampler import pair_residual, probe_generators

common = native.common
ROOT = common.EXPS/'weak_reference_20260914'
Runtime = native.Runtime
SHAPES = native.SHAPES


def configs(name):
    base = 'cfg' if name=='jit' else 'ig'
    return [dict(arm=base,base=base,kind='baseline',omega=0,steps=100),
            dict(arm=base+'_log_k02_w1',base=base,kind='log',omega=1,kappa=.2,steps=100),
            dict(arm=base+'_equal_nfe',base=base,kind='baseline',omega=0,
                 steps=200 if name=='jit' else 300)]


@torch.inference_mode()
def sample(rt,noise,labels,config,trace=False):
    if config['kind']!='log':
        return native.sample(rt,noise,labels,config,trace)
    omega, kappa = float(config['omega']), float(config['kappa'])
    if any(not math.isfinite(x) or x<0 for x in (omega,kappa)):
        raise ValueError(config)
    if config.get('solver','euler')!='euler':raise ValueError(config)
    if omega==0 or kappa==0:
        return native.sample(rt,noise,labels,dict(config,kind='baseline',omega=0),trace)
    before=rt.calls; z=noise.clone(); rows=[]
    generators=probe_generators(noise,config.get('probe_seed',2026091401))
    grid=rt.grid(int(config['steps']),z.device)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=noise.is_cuda):
        for k,(t,u) in enumerate(zip(grid[:-1],grid[1:])):
            base,full,_=native.field(rt,z,t,labels,dict(config,kind='baseline',omega=0))
            sigma=kappa*((1-t) if rt.name=='jit' else t)
            offset=sigma*torch.stack([torch.randn(z.shape[1:],device=z.device,
                dtype=z.dtype,generator=g) for g in generators])
            plus,_=rt.query(z+offset,t,labels)
            minus,_=rt.query(z-offset,t,labels)
            residual=pair_residual(full,plus,minus)
            z=z+(u-t)*(base+omega*residual)
            if not torch.isfinite(z).all() or z.abs().max()>1e6:
                raise FloatingPointError((rt.name,config['arm'],k))
            if trace:
                rms=lambda x:x.float().square().flatten(1).mean(1).sqrt()
                rows.append(torch.stack((torch.ones_like(rms(z))*t,rms(z),
                    rms(omega*residual)),1).cpu().numpy())
    counts=rt.calls-before
    expected=int(config['steps'])*(4 if config['base']=='cfg' else 3)
    assert counts==expected,(counts,expected)
    return z,dict(full=counts,prefix=0,trace=np.stack(rows) if rows else np.empty((0,)))
