import math
import numpy as np
import torch
from . import models


def analytic_checks():
    dtype=torch.float64
    c=torch.tensor([[.25,.75]],dtype=dtype).log()
    u=torch.tensor([[.5,.5]],dtype=dtype).log()
    values=torch.tensor([[-1.],[1.]],dtype=dtype)
    mean=(c.softmax(-1)@values+models.categorical_delta(c,u,values,2.,'mean')).item()
    probability=(c.softmax(-1)@values+models.categorical_delta(c,u,values,2.,'probability')).item()
    assert abs(mean-1.5)<1e-14 and abs(probability-13/14)<1e-14
    torch.testing.assert_close(models.categorical_delta(c,u,values,0.,'probability'),torch.zeros((1,1),dtype=dtype),rtol=0,atol=0)
    torch.testing.assert_close(models.categorical_delta(c+8,u-3,values,2.,'probability'),
        models.categorical_delta(c,u,values,2.,'probability'),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(models.categorical_delta(c,c,values,2.,'probability'),torch.zeros((1,1),dtype=dtype),rtol=0,atol=1e-15)
    # For x'=lambda*x+a*delta, the constant perturbation has exact accumulated
    # endpoint error a*delta*(exp(lambda)-1)/lambda, attaining the scalar bound.
    a,delta,lam=.8,.1,.3
    exact=a*delta*math.expm1(lam)/lam
    n=10000;h=1/n;x=0.
    for _ in range(n):
        v=lam*x+a*delta;x+=h*(v+lam*(x+h*v)+a*delta)/2
    assert abs(x-exact)<1e-10
    rng=np.random.default_rng(17);z=torch.from_numpy(rng.normal(size=(2,4,32,32)))
    tokens=models.patchify(z)
    # Independent inverse without the model's channel convention.
    back=tokens.reshape(2,16,16,2,2,4).permute(0,5,1,3,2,4).reshape_as(z)
    assert torch.equal(back,z)
    return dict(passed=True,mean_extrapolation=mean,probability_mean=probability,
        probability_shift_invariant=True,zero_exact=True,scalar_path_error=exact,
        scalar_numerical_error=abs(x-exact),patch_order_exact=True)
