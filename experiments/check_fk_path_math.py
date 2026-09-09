"""Check coordinate conversion and pathwise derivative against analytic Gaussian quadrature."""
import json
import math
import numpy as np
import torch
from experiments.raev2_fk_path import correction, path_logweight


def heads(z,t):
    a=1-t
    return a*z/(a*a+t*t), 4*a*z/(4*a*a+t*t)


def main():
    z=torch.tensor([[.5],[-.7]],dtype=torch.float64)
    eps=torch.tensor([[.3],[-1.1]],dtype=torch.float64)
    # Production head outputs promote to FP32, so finite differences must resolve that quantization.
    drift,info=correction(heads,z,.5,eps,weight=2.,checkpoint_model=False)
    h=1.e-3
    def value(x):
        return torch.logsumexp(torch.stack([path_logweight(heads,x,.5,e,weight=2.) for e in (eps,-eps)]),0)-math.log(2)
    fd=(value(z+h)-value(z-h))/(2*h)
    error=float((-drift[:,0]-fd).abs().max())
    assert error<2.e-5,(drift,fd,error)
    # 64-node Gauss-Hermite checks the stochastic exponential expectation, not a deterministic proxy.
    nodes,weights=np.polynomial.hermite.hermgauss(64)
    zz=torch.full((64,1),.5,dtype=torch.float64,requires_grad=True)
    ee=torch.from_numpy(nodes[:,None]*math.sqrt(2))
    values=path_logweight(heads,zz,.5,ee,weight=2.)
    logc=torch.logsumexp(values+torch.log(torch.from_numpy(weights/math.sqrt(math.pi))),0)
    g,=torch.autograd.grad(logc,zz)
    quadrature_gradient=float(g.sum())
    delta=.125; u=.875; m=1+delta*(-2/(1+1)+1/(4+1))
    A=delta*(1/(1+1)-1/(4+1))**2
    B=delta*(1/(1+u)-1/(4+u))**2
    expected=4*(A+B*m*m/(1-2*B*delta))
    assert abs(quadrature_gradient-expected)<1.e-7
    assert (drift[:,0]*z[:,0]<0).all()  # Decreasing-time ODE moves outward, as exact Gaussian correction does.
    ck,_=correction(heads,z,.5,eps,weight=2.,checkpoint_model=True)
    torch.testing.assert_close(ck,drift)
    result=dict(passed=True,finite_difference_max_error=error,quadrature_gradient=quadrature_gradient,
                analytic_discrete_sde_gradient=expected,checkpoint_gradient_matches=True,
                gaussian_direction_correct=True,limitations='Validates discretized estimator; not full-future convergence or neural quality.')
    print(json.dumps(result,indent=2))
    from pathlib import Path
    Path('experiments/results/terminal_defect_20260908/fk_path/math_check.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
