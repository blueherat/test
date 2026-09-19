from __future__ import annotations
import numpy as np
import torch
from . import catalog as c, models
from experiments.sit_strong_reference_20260912 import core as previous
from experiments.lifting_scale_sweep_20260909 import read, sha

FAMILIES=c.FAMILIES
configurations=c.configurations
install=previous.install
limiting_checks=previous.limiting_checks


def get_head(rt,method):
    if not hasattr(rt,'compiled_heads'):rt.compiled_heads={}
    if method not in rt.compiled_heads:
        folder=c.ROOT/'training'/method;receipt=read(folder/'complete.json')
        assert receipt['checkpoint_sha256']==sha(folder/'model.pt')
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=False)
        assert saved['method']==method and saved['step']==c.STEPS and saved['request_sha256']==receipt['request_sha256']
        head=models.make_head(rt,method,c.CODEWORDS);head.load_state_dict(saved['ema'],strict=True)
        rt.compiled_heads[method]=head.eval().requires_grad_(False)
    return rt.compiled_heads[method]


@torch.inference_mode()
def sample(rt,noise,labels,config,*,zero=False):
    if config['parameters'].get('inherited_reference'):
        return previous.sample(rt,noise,labels,config,zero=zero)
    method=config['parameters']['method'];source=config['source'];rt.labels=labels
    head=get_head(rt,method if source=='ig' else 'cfg_categorical') if not zero else None
    if source=='cfg' and not zero:
        if not hasattr(rt,'compiled_codebook'):
            rt.compiled_codebook=torch.load(c.ROOT/'data/codebook.pt',map_location='cpu',weights_only=False)['centers'].cuda()
        centers=rt.compiled_codebook
    depth=4 if method=='ig_shallow' else 12
    before=rt.counts.copy();z=noise.clone()
    with models.Capture(rt,depths=(depth,)) as capture:
        try:
            for k in range(64):
                t,h=k/64,1/64;amount=0. if zero else previous.previous.old.amount_at(config,t)
                def velocity(x,tv):
                    strong=rt.field(x,x.new_tensor(tv),'full')
                    if not amount:return strong
                    if source=='ig':
                        weak=models.unpatchify(rt,head(capture.values[depth],capture.context))
                        return strong+amount*(strong-weak)
                    cond_logits=head(capture.values[12],capture.context)
                    try:
                        rt.labels=torch.full_like(labels,100)
                        rt.field(x,x.new_tensor(tv),'full')
                        null_logits=head(capture.values[12],capture.context)
                    finally:rt.labels=labels
                    delta=models.categorical_delta(cond_logits,null_logits,centers,amount,
                        'probability' if method=='cfg_probability' else 'mean')
                    return strong+models.unpatchify(rt,delta)/(1-tv)
                first=velocity(z,t);second=velocity(z+h*first,t+h);z=z+(h/2)*(first+second)
                if not torch.isfinite(z).all() or z.abs().max()>1e6:
                    raise FloatingPointError(f'{config["arm"]}: invalid state at {k}')
        finally:rt.labels=labels
    full=rt.counts['full']-before['full'];prefix=rt.counts['prefix']-before['prefix']
    assert full==(128 if zero or source=='ig' else 224) and prefix==0,(full,prefix)
    return z,dict(full_calls=full,prefix_calls=prefix,auxiliary_full_calls=0,diagnostic_queries=0,
        strang_active_steps=0,diagnostics=np.zeros((len(z),5)),head_method=method,
        independent_weak_prefix=False,guidance_uses_external_semantics=False)
