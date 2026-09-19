"""CPU checks of the changed objective and time law; no generation claims."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
import numpy as np
import torch
from . import k, recovery
from .training import active_time, anchor, install_training
from experiments.guidance_dynamic_50k_20260915 import training


def main():
    recovery.verify()
    torch.set_num_threads(2)
    for path in Path(__file__).parent.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
    path = k.WORK/'experiments/guidance_dynamic_recovery_20260915/pipeline_theory.py'
    ast.parse(path.read_text(),filename=str(path))
    report = {'cpu_only':True,'original_sources_verified':True,'ast':True}
    time_results = {}
    for model in ('sit_small','jit'):
        g = torch.Generator().manual_seed(121)
        state = g.get_state()
        def draw():
            t = torch.rand(32768,generator=g) if model=='sit_small' else torch.sigmoid(torch.randn(32768,generator=g)*.8-.8)
            return active_time(model,t,g)
        t = draw();after = g.get_state()
        g.set_state(state)
        assert torch.equal(t,draw()) and torch.equal(after,g.get_state())
        assert bool(((t>=0)&(t<.5)).all())
        threshold = .25 if model=='sit_small' else 1/(1+math.exp(.8))
        expected = .5 if model=='sit_small' else .5/(.5*(1+math.erf(1/math.sqrt(2))))
        observed = float((t<threshold).float().mean())
        assert abs(expected-observed)<.012
        time_results[model] = dict(replay_exact=True,cdf_expected=expected,cdf_observed=observed)
    report['time_law'] = time_results

    class Adapter:
        def __init__(self,model):self.name=model;self.cfg=k.settings(model)
        def autocast(self,training=False):return nullcontext()
        def patch(self,x):return x.flatten(2).transpose(1,2)
        def features(self,z,t,labels):return dict(context=self.patch(z),condition=t)
        def base(self,z,t,labels,kind):return .2*z+.1
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Linear(3,3,dtype=torch.float64)
        def forward(self,x,condition):return self.layer(x)

    raw_compute = training.compute_loss
    install_training()
    torch.manual_seed(932)
    head = Head()
    batch = dict(positive=torch.randn(8,3,4,4,dtype=torch.float64),
        noise=torch.randn(8,3,4,4,dtype=torch.float64),t=torch.tensor([.01,.1,.2,.249,.251,.3,.4,.49],dtype=torch.float64),
        labels=torch.arange(8),source=torch.zeros(8),coin=torch.zeros(8))
    objective_results = {}
    for model in ('sit_small','jit'):
        adapter = Adapter(model)
        spec = dict(loss='guided_weak',source='real',base='strong',deployed_objective=True)
        changed = training.compute_loss(adapter,head,batch,spec)
        changed_grad = torch.autograd.grad(changed,tuple(head.parameters()))
        z,target = training.pair(adapter,batch,spec)
        pred = head(adapter.patch(z),batch['t']).float()
        baseline = adapter.patch(adapter.base(z,batch['t'],batch['labels'],'strong')).float()
        a = anchor(model,batch['t'])[:,None,None]
        den = (1-batch['t'])[:,None,None] if model=='jit' else 1.
        implied_target = baseline+(baseline-target)/a
        expected = ((a/den)*(pred-implied_target)).square().mean()
        expected_grad = torch.autograd.grad(expected,tuple(head.parameters()))
        assert torch.allclose(changed,expected,rtol=1e-7,atol=1e-9)
        for u,v in zip(changed_grad,expected_grad):assert torch.allclose(u,v,rtol=1e-6,atol=1e-8)
        assert adapter.cfg['alpha']==k.settings(model)['alpha']
        spec.pop('deployed_objective')
        assert torch.equal(training.compute_loss(adapter,head,batch,spec),raw_compute(adapter,head,batch,spec))
        objective_results[model] = dict(loss_error=abs(float((changed-expected).detach())),
            gradient_max_error=max(float((u-v).abs().max()) for u,v in zip(changed_grad,expected_grad)),
            original_arm_unchanged=True,adapter_config_restored=True)
    report['objective'] = objective_results

    rng = np.random.default_rng(914)
    basis,_ = np.linalg.qr(rng.normal(size=(51,9)))
    project = lambda x:basis@(basis.T@x)
    strong,target = rng.normal(size=(2,51,4))
    projection_results = []
    for a in (.3,.8):
        ordinary = project(target)
        guided = ((1+a)*project(strong)-project(target))/a
        for b in (0.,.2,a,1.,2.):
            f0 = strong+b*(strong-ordinary);fg = strong+b*(strong-guided)
            actual = np.sum((f0-target)**2)-np.sum((fg-target)**2)
            expected = ((1+b)**2-(1-b/a)**2)*np.sum(project(strong-target)**2)
            assert abs(actual-expected)<1e-9
            projection_results.append(dict(a=a,b=b,identity_error=abs(actual-expected)))
    report['finite_capacity_risk_identity'] = projection_results
    # Independent finite differences of the normalized Gaussian product path.
    r=2.25;tau=.6;x=np.linspace(-4,4,2001);h=1e-4
    def gaussian_product(s):
        vp,vg=1+s,1.3+s
        precision=(1-r)/vp+r/vg
        variance=1/precision
        return np.exp(-x*x/(2*variance))/np.sqrt(2*np.pi*variance),variance
    q,vq=gaussian_product(tau)
    dt=(gaussian_product(tau+h)[0]-gaussian_product(tau-h)[0])/(2*h)
    lap=(x*x/vq**2-1/vq)*q
    slope=1/(1+tau)-1/(1.3+tau)
    rhs=-.5*r*(r-1)*q*slope**2*(x*x-vq)
    error=float(np.max(np.abs(dt-.5*lap-rhs)))
    assert error<1e-8
    report['normalized_product_heat_defect'] = dict(finite_difference_max_error=error,
        note='Algebra check only; not evidence of image quality or novelty.')
    report['passed'] = True
    path=k.ROOT/'deployed_handoff/cpu_checks.json'
    k.atomic(path,report)
    print(path)
    print(report)


if __name__=='__main__':main()
