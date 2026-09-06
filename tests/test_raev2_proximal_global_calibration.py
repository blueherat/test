import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.raev2_proximal_calibration import DiagonalMoments, apply_proximal
from experiments.raev2_proximal_global_calibration import pool_global_fit


def test_global_pool_uses_uncentered_moments_without_translation():
    gen = torch.Generator().manual_seed(5)
    y = 4 + torch.randn((25, 3, 2, 2), generator=gen, dtype=torch.float64)
    r = .6*y - .2
    moments = DiagonalMoments()
    moments.update(y, r)
    fit = pool_global_fit(moments.fit())
    expected = (r*y).mean()/y.square().mean()
    torch.testing.assert_close(fit.slope, expected.expand(3, 2, 2))
    assert not fit.center.any() and not fit.offset.any()
    torch.testing.assert_close(apply_proximal(y+r, fit), (y+r)/(1+expected))


def test_empirical_actual_wasserstein_decreases_for_nonoptimal_coupling():
    # Non-Gaussian, nonzero-mean finite distributions; the training pairing is
    # deliberately not their optimal transport pairing.
    y = torch.tensor([[1., 2.], [-2., 4.], [3., -1.], [2., 5.]], dtype=torch.float64)
    p = 2.2*y + torch.tensor([[1., 0.], [0., 1.], [-1., 2.], [0., -1.]], dtype=torch.float64)
    paired_y = y[[0, 3, 2, 1]]
    moments = DiagonalMoments()
    moments.update(paired_y, p-paired_y)
    fit = pool_global_fit(moments.fit())
    new = apply_proximal(p, fit)
    def w2(x):
        cost = torch.cdist(x, y).square().numpy()
        rows, cols = linear_sum_assignment(cost)
        return np.mean(cost[rows, cols])
    assert fit.slope[0] > 0
    assert w2(new) < w2(p)


def test_inactive_cone_is_exact_identity():
    y = torch.tensor([[1., 3.], [-1., 2.]], dtype=torch.float64)
    moments = DiagonalMoments()
    moments.update(y, -.2*y)
    fit = pool_global_fit(moments.fit())
    assert not fit.slope.any()
    assert torch.equal(apply_proximal(.8*y, fit), .8*y)
