"""Independent gradient check and effective-count/balanced-pairing checks."""
import numpy as np
import torch
from experiments.raev2_paired_ratio_loss import paired_scaled_logistic
from experiments.raev2_conditional_ratio_objective import ConditionalRatioRisk
from experiments.raev2_prefix_ratio64k_plan import PLAN, positive_indices


def test_cached_conditional_objective_against_autograd():
    rng = np.random.default_rng(202609106)
    a = np.geomspace(.001, .9, 17)
    q = rng.normal(size=(17, 9))
    p = q[:, None]+a[:, None, None]*rng.normal(size=(17, 5, 9))
    w = rng.normal(size=9)
    risk = ConditionalRatioRisk(p, q, a, .23)
    value, gradient = risk(w)
    param = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    ref = paired_scaled_logistic(torch.from_numpy(p)@param,
        (torch.from_numpy(q)@param)[:, None], torch.from_numpy(a)[:, None]).mean()+.23/2*param.square().sum()
    grad, = torch.autograd.grad(ref, param)
    np.testing.assert_allclose(value, ref.item(), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(gradient, grad.numpy(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(risk.losses(w).mean()+.23/2*(w@w), value, atol=1e-8, rtol=1e-8)


def test_pairing_preserves_class_balance_and_independent_count():
    ids = np.arange(64000)
    k = positive_indices(ids, 'train')
    assert k.shape == (64000, 5)
    for c in range(1000):
        np.testing.assert_array_equal(np.bincount(k[c::1000].ravel(), minlength=64), np.full(64, 5))
    validation = positive_indices(np.arange(8000), 'validation')
    np.testing.assert_array_equal(validation, np.tile(np.arange(8), (8000, 1)))
    assert PLAN['ridge'] == 2881/64000
