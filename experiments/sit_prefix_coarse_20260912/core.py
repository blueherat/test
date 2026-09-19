from __future__ import annotations
import numpy as np
import torch
from . import catalog,models
from experiments import sit_guidance_fusion_20260910 as previous
from experiments.lifting_scale_sweep_20260909 import read,sha

FAMILIES=catalog.FAMILIES
configurations=catalog.configurations
install=previous.install
limiting_checks=previous.limiting_checks


def get_weak(rt,method):
    if not hasattr(rt,'trained_weak_prefixes'):rt.trained_weak_prefixes={}
    if method not in rt.trained_weak_prefixes:
        folder=catalog.ROOT/'training'/method;receipt=read(folder/'complete.json')
        assert receipt['checkpoint_sha256']==sha(folder/'model.pt')
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        assert saved['method']==method and saved['step']==catalog.STEPS
        assert saved['request_sha256']==receipt['request_sha256']
        model=models.WeakPrefix(rt)
        model.load_state_dict(saved['ema'],strict=True)
        rt.trained_weak_prefixes[method]=model.eval().requires_grad_(False)
    return rt.trained_weak_prefixes[method]


@torch.inference_mode()
def sample(rt,noise,labels,config,*,zero=False):
    if config['parameters'].get('inherited_exact'):
        return previous.sample(rt,noise,labels,config,zero=zero)
    rt.labels=labels;before=rt.counts.copy();z=noise.clone()
    method=config['parameters']['head_method']
    weak=get_weak(rt,method) if method!='original' and not zero else None
    for k in range(64):
        t,h=k/64,1/64
        amount=0. if zero else previous.old.amount_at(config,t)
        def velocity(x,tv):
            ts=x.new_tensor(tv)
            strong=rt.field(x,ts,'full')
            if not amount:return strong
            if method=='original':value=rt.field(x,ts,'base')
            else:
                rt.counts['prefix']+=1
                value=weak(x,rt.times(x,ts),labels)
            return strong+amount*(strong-value)
        first=velocity(z,t);second=velocity(z+h*first,t+h)
        z=z+(h/2)*(first+second)
        if not torch.isfinite(z).all() or z.abs().max()>1e6:
            raise FloatingPointError(f'{config["arm"]}: invalid state at step {k}')
    full=rt.counts['full']-before['full'];prefix=rt.counts['prefix']-before['prefix']
    assert full==128 and prefix==(0 if zero else 64),(full,prefix)
    return z,dict(full_calls=full,prefix_calls=prefix,auxiliary_full_calls=0,
        diagnostic_queries=0,strang_active_steps=0,diagnostics=np.zeros((len(z),5)),
        weak_method=method,independent_weak_prefix=True,guidance_uses_external_semantics=False)
