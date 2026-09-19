from __future__ import annotations
import torch
from . import catalog as c, models
from experiments.sit_reference_compilation_20260912 import core as previous
from experiments.lifting_scale_sweep_20260909 import read, sha

FAMILIES=c.FAMILIES
configurations=c.configurations
install=previous.install
limiting_checks=previous.limiting_checks


def get_head(rt,method):
    if not hasattr(rt,'risk_heads'):rt.risk_heads={}
    if method not in rt.risk_heads:
        folder=c.ROOT/'training'/method;receipt=read(folder/'complete.json')
        assert receipt['passed'] and receipt['checkpoint_sha256']==sha(folder/'model.pt')
        assert receipt['request_sha256']==sha(c.ROOT/'training_request.json')
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        assert saved['method']==method and saved['step']==c.STEPS
        assert saved['request_sha256']==receipt['request_sha256']
        head=models.make_head(rt,method);head.load_state_dict(saved['ema'],strict=True)
        rt.risk_heads[method]=head.eval().requires_grad_(False)
    return rt.risk_heads[method]


@torch.inference_mode()
def sample(rt,noise,labels,config,*,zero=False):
    method=config['parameters'].get('method')
    if method not in c.METHODS:return previous.sample(rt,noise,labels,config,zero=zero)
    head=get_head(rt,method) if not zero else None
    if not hasattr(rt,'compiled_heads'):rt.compiled_heads={}
    prior=rt.compiled_heads.get('ig_shallow')
    try:
        if head is not None:rt.compiled_heads['ig_shallow']=head
        routed=dict(config,parameters=dict(config['parameters'],method='ig_shallow'))
        value,stats=previous.sample(rt,noise,labels,routed,zero=zero)
        return value,dict(stats,head_method=method)
    finally:
        if prior is None:rt.compiled_heads.pop('ig_shallow',None)
        else:rt.compiled_heads['ig_shallow']=prior
