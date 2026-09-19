"""Frozen candidates/baselines plus a separately calibrated time-only control."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from . import baselines as b, independent_blend as blend

GAIN_TABLE = b.common.EXPS/'cfg_transport_search_20260913/gain_probe/calibration.npz'
SOURCE_FILES = [Path(b.__file__).resolve(),Path(blend.__file__).resolve(),GAIN_TABLE]


@lru_cache(maxsize=1)
def gain_table():
    with np.load(GAIN_TABLE) as d: gain = np.array(d['mean_gain'],dtype=np.float64)
    assert gain.shape==(48,2) and np.isfinite(gain).all()
    return gain


def field(rt,z,t,labels,history,alpha,gain,active):
    old=rt.labels
    try:
        rt.labels=labels; time=z.new_tensor(t)
        conditional=rt.field(z,time,'full')
        if not active:return conditional,history
        rt.labels=torch.full_like(labels,100)
        unconditional=rt.field(z,time,'full')
    finally:rt.labels=old
    gap=conditional-unconditional
    previous=torch.zeros_like(gap) if history is None else history
    momentum=gap+(-.5)*previous
    bounded=b.cap(momentum,2*b.norm(gap))
    clean=z+(1-time)*conditional
    direction=bounded-b.projection(bounded,clean)
    return conditional+(alpha*float(gain))*direction,momentum.detach()


@torch.inference_mode()
def sample(rt,noise,labels,config,snapshots=False):
    if config['kind'] != 'apg_time_gain':
        return blend.sample(rt,noise,labels,config,snapshots)
    assert config.get('steps',64)==64 and config.get('cutoff',.75)==.75
    assert config.get('beta',-.5)==-.5
    gains=np.ones((48,2)) if config.get('unit_gain_check',False) else gain_table()
    alpha=float(config.get('alpha',2.)); assert alpha>0
    before=rt.counts.copy();rt.labels=labels;z=noise.clone();history=None;trace=[]
    saved={'step_000':z.detach().clone()} if snapshots else None
    with rt.context():
        for k in range(64):
            t,h=k/64,1/64;active=k<48
            left,right=gains[k] if active else (0.,0.)
            first,proposal=field(rt,z,t,labels,history,alpha,left,active)
            second,_=field(rt,z+h*first,t+h,labels,history,alpha,right,active)
            z=z+(h/2)*(first+second)
            if active:history=proposal
            if not torch.isfinite(z).all() or z.abs().max()>1e6:
                raise FloatingPointError((config['arm'],k))
            if snapshots:
                trace.append(torch.stack((z.new_full((len(z),),k),z.new_full((len(z),),t),
                    z.new_full((len(z),),left),z.new_full((len(z),),right),
                    z.square().flatten(1).mean(1).sqrt(),first.square().flatten(1).mean(1).sqrt()),-1))
                if k+1 in (16,32,48,64):saved[f'step_{k+1:03d}']=z.detach().clone()
    counts={key:rt.counts[key]-before[key] for key in ('full','prefix')}
    assert counts==dict(full=224,prefix=0)
    result=dict(latents=z,counts=counts)
    if snapshots:result.update(snapshots=saved,trace=torch.stack(trace).cpu().numpy())
    return result


if __name__=='__main__':
    import json
    c=b.common;rt=c.runtime('sit_small')
    with np.load(c.EXPS/'cfg_transport_search_20260913/baseline_1k/inputs.npz') as d:
        z=c.cuda(d['noise'][:8]);labels=c.cuda(d['labels'][:8])
    plain=b.sample(rt,z,labels,dict(kind='apg',alpha=2.))
    unit=sample(rt,z,labels,dict(kind='apg_time_gain',alpha=2.,unit_gain_check=True))
    assert torch.equal(plain['latents'],unit['latents'])
    calibrated=sample(rt,z,labels,dict(kind='apg_time_gain',alpha=2.),True)
    assert len(calibrated['snapshots'])==5 and calibrated['trace'].shape==(64,8,6)
    out=dict(actual_samples=8,unit_gain_apg_bitwise=True,counts=calibrated['counts'],
             gain_min=float(gain_table().min()),gain_max=float(gain_table().max()),
             calibration_sha256=c.sha(GAIN_TABLE))
    target=c.WORK/'docs/research/cfg_transport_search_20260913/time_gain_model_check.json'
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)
