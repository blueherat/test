from __future__ import annotations
import copy
from dataclasses import replace
import numpy as np
import torch
from . import catalog
from experiments import sit_guidance_fusion_20260910 as previous
from experiments.lifting_scale_sweep_20260909 import read,sha

FAMILIES=catalog.FAMILIES
configurations=catalog.configurations
install=previous.install
limiting_checks=previous.limiting_checks


def get_head(rt,method):
    if method=='original':return rt.head
    if not hasattr(rt,'internal_coarse_heads'):rt.internal_coarse_heads={}
    if method not in rt.internal_coarse_heads:
        folder=catalog.ROOT/'training'/method
        receipt=read(folder/'complete.json')
        assert receipt['checkpoint_sha256']==sha(folder/'model.pt')
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        assert saved['method']==method and saved['step']==catalog.STEPS
        assert saved['request_sha256']==receipt['request_sha256']
        module=copy.deepcopy(rt.head.module)
        # The legacy runtime has a diagnostic hook on its original head. The
        # independent replacement needs no hooks and leaves the original intact.
        for layer in module.modules():
            layer._forward_hooks.clear()
            layer._forward_pre_hooks.clear()
        module.load_state_dict(saved['ema'],strict=True)
        module.eval().requires_grad_(False)
        rt.internal_coarse_heads[method]=replace(rt.head,module=module,
            checkpoint=str(folder/'model.pt'),checkpoint_sha256=receipt['checkpoint_sha256'])
    return rt.internal_coarse_heads[method]


@torch.inference_mode()
def sample(rt,noise,labels,config,*,zero=False):
    if config['parameters'].get('inherited_exact'):
        return previous.sample(rt,noise,labels,config,zero=zero)
    rt.labels=labels
    before=rt.counts.copy()
    original=rt.head
    rt.head=get_head(rt,config['parameters']['head_method'])
    z=noise.clone()
    try:
        for k in range(64):
            t,h=k/64,1/64
            amount=0. if zero else previous.old.amount_at(config,t)
            def velocity(x,tv):
                if amount==0:return rt.field(x,x.new_tensor(tv),'full')
                strong,weak=rt.pair(x,x.new_tensor(tv))
                return strong+amount*(strong-weak)
            first=velocity(z,t)
            second=velocity(z+h*first,t+h)
            z=z+(h/2)*(first+second)
            if not torch.isfinite(z).all() or z.abs().max()>1e6:
                raise FloatingPointError(f'{config["arm"]}: invalid state at step {k}')
    finally:
        rt.head=original
    full=rt.counts['full']-before['full']
    prefix=rt.counts['prefix']-before['prefix']
    assert full==128 and prefix==0,(full,prefix)
    return z,dict(full_calls=full,prefix_calls=prefix,auxiliary_full_calls=0,
        diagnostic_queries=0,strang_active_steps=0,diagnostics=np.zeros((len(z),5)),
        head_method=config['parameters']['head_method'],guidance_uses_external_semantics=False)
