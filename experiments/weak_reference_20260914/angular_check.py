"""Check chain rule, rotation invariants, and Gaussian coordinate conversions."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from experiments.weak_reference_20260914 import angular as a
from experiments.self_guidance_20260913.check import Analytic


def checks(real=False):
    torch.manual_seed(1408);x=torch.randn(5,2,4,4,dtype=torch.float64)
    perm=torch.randperm(x[0].numel());angle=.2
    torch.testing.assert_close(a.j_apply(a.j_apply(x,perm),perm),-x,rtol=0,atol=0)
    r=a.rotate(x,perm,angle)
    torch.testing.assert_close(r.square().sum(1).sum(),x.square().sum(1).sum(),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(a.rotate(r,perm,-angle),x,rtol=1e-14,atol=1e-14)
    # Independent autograd score of the transformed non-Gaussian potential.
    x.requires_grad_(True)
    potential=lambda z:-(z.square()/2+.01*z.pow(4)).sum()
    score=lambda z:-z-.04*z.pow(3)
    grad=torch.autograd.grad(.5*(potential(a.rotate(x,perm,angle))+potential(a.rotate(x,perm,-angle))),x)[0]
    pull=.5*(a.rotate(score(a.rotate(x,perm,angle)),perm,-angle)+a.rotate(score(a.rotate(x,perm,-angle)),perm,angle))
    torch.testing.assert_close(grad,pull,rtol=1e-12,atol=1e-12)
    # Spherical Gaussian with arbitrary nonzero mean, including t=0 limit.
    center=torch.randn_like(x);labels=torch.arange(len(x));errors=[]
    class Gaussian(Analytic):
        def field(self,z,t,kind):
            self.counts['full']+=1
            variance=1.7*t*t+(1-t)**2
            return center+(1.7*t-(1-t))/variance*(z-t*center)
    rt=Gaussian()
    for t in (0.,.3,.9):
        z=x.detach();full=rt.field(z,z.new_tensor(t),'full')
        delta=a.correction(rt,z,t,labels,full,perm,angle,center)
        errors.append(float(delta.abs().max()));assert errors[-1]<1e-12
    # General covariance, whiten -> rotate -> unwhiten, using exact transpose Jacobian.
    cov=np.array([[2.,.4],[.4,.2]]);eig,Q=np.linalg.eigh(cov);S=(Q*np.sqrt(eig))@Q.T
    R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    T=S@R@np.linalg.inv(S);P=np.linalg.inv(cov)
    error=float(np.max(np.abs(T.T@P@T-P)));assert error<1e-12
    result=dict(cpu_passed=True,gaussian_velocity_max_error=max(errors),
        whitened_gaussian_precision_error=error,autograd_chain_rule=True,norm_and_inverse=True)
    if real:
        from experiments.guidance_pasted_20260912 import common as c
        rt=c.runtime('sit_small');noise=torch.randn(2,4,32,32,device='cuda');labels=torch.tensor([2,61],device='cuda')
        config=dict(arm='check',kind='angular',alpha=1.25,omega=0,angle=.2,steps=64)
        base=a.sample(rt,noise,labels,config);ref=a.plain.sample(rt,noise,labels,dict(config,kind='baseline'))
        assert torch.equal(base['latents'],ref['latents'])
        out=a.sample(rt,noise,labels,dict(config,omega=1))
        assert out['counts']['full']==240 and not torch.equal(out['latents'],ref['latents'])
        result.update(model_passed=True,zero_native_bitwise=True,counts=out['counts'])
    path=Path('docs/data/weak_reference_20260914/angular_check.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',action='store_true');checks(p.parse_args().model)
