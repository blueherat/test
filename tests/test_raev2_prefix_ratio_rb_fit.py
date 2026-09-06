"""Conditional-average objective retains one prior, not five priors."""
import numpy as np
import torch
from experiments.fit_raev2_prefix_ratio_rb import rb_objective_gradient
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic


def test_conditional_risk_and_gradient_match_independent_autograd():
    rng = np.random.default_rng(202609099)
    a = np.geomspace(.001, .9, 13)
    q = rng.normal(size=(13, 7))
    p = q[:, None]+a[:, None, None]*rng.normal(size=(13, 5, 7))
    w = rng.normal(size=7)
    value, gradient = rb_objective_gradient(w, p, q, a, .2)
    parameter = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    fp = torch.from_numpy(p)@parameter
    fq = (torch.from_numpy(q)@parameter)[:, None].expand_as(fp)
    signal = torch.from_numpy(a)[:, None].expand_as(fp)
    reference = paired_scaled_logistic(fp, fq, signal).mean()+.1*parameter.square().sum()
    grad, = torch.autograd.grad(reference, parameter)
    np.testing.assert_allclose(value, reference.item(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(gradient, grad.numpy(), atol=1e-10, rtol=1e-10)
