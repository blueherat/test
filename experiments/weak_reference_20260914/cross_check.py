"""Real model zero parity and full-length finite output checks."""
import argparse
import json
import torch
from experiments.weak_reference_20260914 import cross_core as c


def main(name):
    rt=c.Runtime(name)
    torch.manual_seed(2026091402)
    x=torch.randn(2,*c.SHAPES[name],device='cuda');y=torch.tensor([2,61],device='cuda')
    base=c.configs(name)[0]; cfg=c.configs(name)[1]
    ref,counts=c.native.sample(rt,x,y,base)
    for key in ('omega','kappa'):
        zero,nfe=c.sample(rt,x,y,dict(cfg,**{key:0}))
        assert torch.equal(zero,ref) and nfe['full']==counts['full']
    guided,nfe=c.sample(rt,x,y,cfg)
    assert torch.isfinite(guided).all() and not torch.equal(guided,ref)
    pixels=rt.decode(guided)
    assert pixels.shape==(2,256,256,3)
    result=dict(model=name,model_passed=True,zero_scale_and_radius_native_bitwise=True,
        counts=nfe['full'],latent_delta_rms=float((guided-ref).square().mean().sqrt()),
        max_memory_allocated=torch.cuda.max_memory_allocated())
    c.common.atomic(c.ROOT/'checks'/f'{name}.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['jit','raev2'])
    main(p.parse_args().model)
