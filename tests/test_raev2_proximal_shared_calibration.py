import torch

from experiments.raev2_proximal_calibration import DiagonalMoments, apply_proximal
from experiments.raev2_proximal_shared_calibration import pool_channel_fit


def test_pool_matches_complete_data_with_unequal_position_means():
    gen = torch.Generator().manual_seed(17)
    x = torch.randn((23, 3, 2, 4), generator=gen, dtype=torch.float64)
    x += torch.arange(8, dtype=torch.float64).reshape(1, 1, 2, 4)
    r = .4*x + torch.randn(x.shape, generator=gen, dtype=torch.float64)
    r[:, 1] *= -1
    moments = DiagonalMoments()
    moments.update(x, r)
    fit = pool_channel_fit(moments.fit())
    mean_x, mean_r = x.mean((0, 2, 3)), r.mean((0, 2, 3))
    xc, rc = x-mean_x[None, :, None, None], r-mean_r[None, :, None, None]
    expected = ((xc*rc).mean((0, 2, 3))/xc.square().mean((0, 2, 3))).clamp_min(0)
    torch.testing.assert_close(fit.slope[:, 0, 0], expected)
    torch.testing.assert_close(fit.offset[:, 0, 0], mean_r)
    assert fit.count == 23


def test_channel_projection_keeps_finite_step_certificate():
    gen = torch.Generator().manual_seed(31)
    y = torch.randn((30, 2, 2, 3), generator=gen, dtype=torch.float64)
    residual = torch.stack((.3*y[:, 0]+.2, -.2*y[:, 1]), dim=1)
    moments = DiagonalMoments()
    moments.update(y, residual)
    fit = pool_channel_fit(moments.fit())
    h = fit.offset+fit.slope*(y-fit.center)
    corrected = apply_proximal(y+residual, fit)
    torch.testing.assert_close(corrected-y, (residual-h)/(1+fit.slope))
    assert ((corrected-y).square().mean() <= residual.square().mean()).item()
