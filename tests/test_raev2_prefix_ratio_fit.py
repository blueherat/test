"""Analytic convex objective/gradient versus independent PyTorch autograd."""
import numpy as np
import torch
from experiments.fit_raev2_prefix_ratio import objective_gradient
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic


def test_convex_gradient_matches_scaled_logistic_autograd():
    rng = np.random.default_rng(202609098)
    signal = np.geomspace(.001, .9, 17)
    p = rng.normal(size=(17, 7))
    q = p + signal[:, None]*rng.normal(size=(17, 7))
    w = rng.normal(size=7)
    loss, gradient = objective_gradient(w, p, q, signal, .2)
    parameter = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    reference = paired_scaled_logistic(torch.from_numpy(p)@parameter, torch.from_numpy(q)@parameter,
                                       torch.from_numpy(signal)).mean() + .1*parameter.square().sum()
    grad, = torch.autograd.grad(reference, parameter)
    np.testing.assert_allclose(loss, reference.item(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(gradient, grad.numpy(), atol=1e-10, rtol=1e-10)
