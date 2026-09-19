"""Check two analytical limits; no model queries or quality selection."""
from pathlib import Path
import hashlib
import json
import torch


def derivative(value,x):
    return torch.autograd.grad(value.sum(),x,create_graph=True,retain_graph=True)[0]


def divergence(value,x):
    return sum(derivative(value[:,j],x)[:,j] for j in range(x.shape[1]))


def main():
    torch.set_num_threads(4)
    gen=torch.Generator().manual_seed(2026121015)
    x=torch.randn((127,2),generator=gen,dtype=torch.float64).requires_grad_()
    means=x.new_tensor([[-1.,.5],[.8,-.4]])
    variance=x.new_tensor([.7,1.6]);weights=x.new_tensor([.35,.65])
    logits=weights.log()[None]-variance.log()[None]-(x[:,None]-means[None]).square().sum(-1)/(2*variance[None])
    logq=torch.logsumexp(logits,dim=1);q=logq.exp();score=derivative(logq,x)
    strong=torch.stack((torch.sin(x[:,0])+x[:,1],torch.cos(x[:,1])-.3*x[:,0]),dim=1)
    checks=[]
    for t,a in ((.13,.8),(.41,1.25),(.73,.8)):
        weak=x/t+(1-t)/t*score
        guided=(1+a)*strong-a*weak
        continuity=-divergence(q[:,None]*guided,x)
        diffusion=a*(1-t)/t
        sde_drift=(1+a)*strong-a*x/t
        fokker_planck=-divergence(q[:,None]*sde_drift,x)+diffusion*divergence(derivative(q,x),x)
        torch.testing.assert_close(continuity,fokker_planck,rtol=1e-11,atol=1e-12)
        checks.append(dict(t=t,a=a,max_pde_difference=float((continuity-fokker_planck).abs().max())))
    a=.8;endpoint_variance=4.**(-a);path=[]
    for t in (0.,.25,.5,.75,1.):
        vp=t*t+(1-t)**2;vw=4*t*t+(1-t)**2
        actual=vp**(1+a)/vw**a
        renoised=t*t*endpoint_variance+(1-t)**2
        path.append(dict(t=t,actual_rollout_variance=actual,renoised_endpoint_variance=renoised))
    assert abs(path[2]['actual_rollout_variance']-path[2]['renoised_endpoint_variance'])>.09
    out=Path('docs/data/guidance_shared_budget_20260912');out.mkdir(parents=True,exist_ok=True)
    result=dict(passed=True,pde_checks=checks,endpoint_vs_path=path,
        scope='Continuous exact-density identities at t>0; no empirical IG/CFG density claim or new FID result.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'sampler_law_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
