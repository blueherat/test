"""CPU algebra audit. These checks provide no image-quality evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .objectives import contrast_loss, contrast_residual, covariance_target


def main():
    assert not torch.cuda.is_initialized()
    torch.set_default_dtype(torch.float64)
    atoms = torch.tensor([-1.7, -.4, .9, 2.3])
    prior = torch.tensor([[.10, .30, .45, .15], [.40, .10, .20, .30]])
    time = .37
    z = torch.linspace(-2, 3, 19, requires_grad=True)
    likelihood = torch.exp(-.5 * ((z[:, None] - time * atoms) / (1 - time))**2)
    joint = .5 * prior[None] * likelihood[:, None]
    posterior = joint / joint.sum((1, 2), keepdim=True)
    eta = posterior[:, 1].sum(1)
    mu = (posterior * atoms).sum(2) / posterior.sum(2)
    velocities = (mu - z[:, None]) / (1 - time)
    delta = velocities[:, 1] - velocities[:, 0]
    mixture = (velocities * posterior.sum(2)).sum(1)
    variance = eta * (1 - eta)
    source = torch.tensor([0., 1.])[None, :, None].expand_as(posterior)
    target = ((atoms[None] - z[:, None]) / (1 - time))[:, None].expand_as(posterior)
    actual_cov = (posterior * (source - eta[:, None, None]) * (target - mixture[:, None, None])).sum((1, 2))
    torch.testing.assert_close(actual_cov, variance * delta, atol=1e-12, rtol=1e-12)
    eta_grad = torch.autograd.grad(eta.sum(), z, retain_graph=True)[0]
    torch.testing.assert_close(actual_cov, (1-time)/time * eta_grad, atol=1e-12, rtol=1e-12)
    posterior_score = torch.autograd.grad(torch.log(eta / (1-eta)).sum(), z, retain_graph=True)[0]
    torch.testing.assert_close(delta, (1-time)/time * posterior_score, atol=1e-11, rtol=1e-11)

    # Population minimizer with deliberately inaccurate input-measurable baseline.
    baseline = mixture.detach() + .7 * torch.sin(z.detach())
    theta = delta.detach().clone().requires_grad_()
    expand = lambda value: value[:, None, None].expand_as(posterior).reshape(-1, 1)
    r = contrast_residual(expand(theta), target.reshape(-1,1), source.reshape(-1),
                          expand(eta).reshape(-1), expand(baseline))
    population_risk = (posterior.detach().reshape(-1) * r.square().flatten()).sum()
    derivative = torch.autograd.grad(population_risk, theta)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(derivative), atol=1e-12, rtol=0)

    # Both nuisance errors: exact bias, including the squared-error shrinkage.
    eta_hat = .9 * eta.detach() + .03
    a = eta.detach() - eta_hat
    b = mixture.detach() - baseline
    formula = (variance.detach() * delta.detach() + a * b) / (variance.detach() + a.square())
    centered = source - eta_hat[:,None,None]
    computed = (posterior.detach() * centered * (target.detach()-baseline[:,None,None])).sum((1,2))
    computed /= (posterior.detach() * centered.square()).sum((1,2))
    torch.testing.assert_close(computed, formula, atol=1e-12, rtol=1e-12)
    cov_hat = covariance_target(target.reshape(-1,1), source.reshape(-1),
                                expand(eta_hat).reshape(-1), expand(baseline)).flatten()
    cov_hat = (posterior.detach().reshape(-1) * cov_hat).reshape_as(posterior).sum((1,2))
    torch.testing.assert_close(cov_hat, variance.detach()*delta.detach()+a*b, atol=1e-12, rtol=1e-12)

    # The API must update only the correction head, not the nuisance estimates.
    pred = torch.zeros((4,3), requires_grad=True)
    noise_target = torch.arange(12.).reshape(4,3).requires_grad_()
    probability = torch.tensor([.2,.3,.6,.8], requires_grad=True)
    base = torch.ones((4,3), requires_grad=True)
    loss = contrast_loss(pred, noise_target, torch.tensor([0.,1.,0.,1.]), probability, base)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert probability.grad is base.grad is noise_target.grad is None

    # Independent quadrature verifies the source-information I-MMSE corollary.
    x = atoms.detach().numpy()
    weights = prior.numpy()
    u = np.linspace(-12,12,12001)
    def information(gamma):
        kernel = np.exp(-.5*(u[:,None]-np.sqrt(gamma)*x)**2)/np.sqrt(2*np.pi)
        density = kernel @ weights.T
        mix = density.mean(1)
        value = np.trapezoid((.5*density*np.log(density/mix[:,None])).sum(1),u)
        means = (kernel*x) @ weights.T / density
        prob = density[:,1]/(2*mix)
        expected = .5*np.trapezoid(mix*prob*(1-prob)*(means[:,1]-means[:,0])**2,u)
        return value, expected
    gamma, step = 1.6, 1e-4
    fd = (information(gamma+step)[0]-information(gamma-step)[0])/(2*step)
    exact = information(gamma)[1]
    assert abs(fd-exact)<2e-8, (fd,exact)
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parents[2]
    out = root/'docs/data/endpoint_contrast_20260914'
    out.mkdir(parents=True,exist_ok=True)
    result = dict(passed=True,cuda_initialized=False,image_quality_evaluated=False,
        conditional_covariance_identity=True,score_difference_identity=True,
        oracle_eta_arbitrary_baseline_unbiased=True,nuisance_bias_identity=True,
        covariance_bias_is_product=True,nuisance_gradients_stopped=True,
        i_mmse_source_corollary_error=float(abs(fd-exact)),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
    (out/'algebra_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
