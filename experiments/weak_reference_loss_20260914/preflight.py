"""CPU-only checks before admission; real model checks are a queued GPU task."""
import ast
import json
from pathlib import Path

import numpy as np
import torch

from . import config as k
from . import idle
from .objectives import smoothing_pair, excess_weights, weighted_mse


def main():
    assert not torch.cuda.is_initialized()
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    g=torch.Generator().manual_seed(1729)
    x=torch.randn((12,2,3,3),generator=g,dtype=torch.float64)
    noise=torch.randn(x.shape,generator=g,dtype=torch.float64)
    t=torch.linspace(.02,.98,len(x),dtype=torch.float64)
    native=x-noise
    z0,v0=smoothing_pair(x,t,noise,.7,torch.zeros(len(x),dtype=torch.bool))
    assert torch.equal(v0,native)
    assert torch.equal(z0,t[:,None,None,None]*x+(1-t[:,None,None,None])*noise)
    z,v=smoothing_pair(x,t,noise,.7,torch.ones(len(x),dtype=torch.bool))
    a=t[:,None,None,None]
    d=((1-a)**2+a*a*.7**2).sqrt()
    conditional_y=x+a*.7**2/d*noise
    torch.testing.assert_close(v,(conditional_y-z)/(1-a),atol=1e-12,rtol=1e-12)
    mask=torch.arange(len(x))%2==0
    zm,vm=smoothing_pair(x,t,noise,.7,mask)
    assert torch.equal(vm[mask],v[mask]) and torch.equal(vm[~mask],native[~mask])
    assert torch.equal(zm[mask],z[mask]) and torch.equal(zm[~mask],z0[~mask])
    probability=torch.tensor([0.,.1,.49,.5,.6,.9,1.])
    w=excess_weights(probability)
    assert torch.all((w>=1)&(w<=2)) and torch.equal(w[:4],torch.ones(4))
    prediction=torch.randn((12,2,3,3),generator=g,requires_grad=True)
    weights=torch.ones(12)
    torch.testing.assert_close(weighted_mse(prediction,v,weights),(prediction-v.float()).square().mean())
    loss=weighted_mse(prediction,v,torch.linspace(.5,1.5,12))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    free=dict(memory_mib=0,utilization=0,compute_pids=[])
    assert idle.eligible(free)
    for update in (dict(memory_mib=600),dict(utilization=6),dict(compute_pids=[123])):
        assert not idle.eligible(dict(free,**update))
    assert not torch.cuda.is_initialized()
    request=k.prepare()
    assert not torch.cuda.is_initialized()
    result=dict(passed=True,cuda_initialized=False,syntax_passed=True,native_pair_bitwise=True,
        gaussian_conditional_target_identity=True,mixture_uses_whole_images=True,
        endpoint_weights_positive_bounded=True,weighted_loss_gradients_finite=True,
        idle_admission_rejects_busy_gpu=True,request_sha256=k.sha(k.ROOT/'request.json'),
        tau=request['tau'],gpu_checks_pending=True)
    k.atomic(k.ROOT/'cpu_preflight.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
