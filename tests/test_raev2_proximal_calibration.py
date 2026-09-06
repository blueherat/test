import torch

from experiments.raev2_proximal_calibration import (
    DiagonalMoments, apply_proximal, evaluate_correction, heldout_step_metrics,
)


def test_cone_projection_removes_positive_linear_error_and_retains_negative():
    # Independent exact centered sign quadrature, with nonzero means/offsets.
    y = torch.tensor([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], dtype=torch.float64)
    center = torch.tensor([2., -3.], dtype=torch.float64)
    offset = torch.tensor([0.4, -0.2], dtype=torch.float64)
    residual = offset+y*torch.tensor([0.3, -0.6], dtype=torch.float64)
    target = y+center
    moments = DiagonalMoments()
    moments.update(target[:2], residual[:2])
    moments.update(target[2:], residual[2:])
    fit = moments.fit()
    torch.testing.assert_close(fit.center, center)
    torch.testing.assert_close(fit.offset, offset)
    torch.testing.assert_close(fit.slope, torch.tensor([0.3, 0.], dtype=torch.float64))
    h = evaluate_correction(target, fit)
    torch.testing.assert_close(residual.square().mean()-(residual-h).square().mean(), h.square().mean())
    corrected = apply_proximal(target+residual, fit)
    torch.testing.assert_close(corrected[:, 0], target[:, 0])
    torch.testing.assert_close(corrected[:, 1]-target[:, 1], -0.6*y[:, 1])


def test_next_state_oracle_identity_with_irreducible_posterior_noise():
    # Full independent quadrature for X,E~symmetric ±1, with Gaussian-oracle
    # conditional mean at t=.5: G=(1-t)/(t²+(1-t)²) z=z.
    x = torch.tensor([[-1.], [-1.], [1.], [1.]], dtype=torch.float64)
    noise = torch.tensor([[-1.], [1.], [-1.], [1.]], dtype=torch.float64)
    t, s = .5, .4
    z = (1-t)*x+t*noise
    true_next = (1-s)*x+s*noise
    predicted = z+(t-s)*(z-z)/t
    moments = DiagonalMoments()
    moments.update(true_next, predicted-true_next)
    fit = moments.fit()
    assert bool((fit.covariance < 0).all())
    assert torch.equal(fit.slope, torch.zeros_like(fit.slope))
    assert torch.equal(fit.offset, torch.zeros_like(fit.offset))
    assert torch.equal(apply_proximal(predicted, fit), predicted)


def test_unseen_risk_certificate_and_nonexpansivity():
    rng = torch.Generator().manual_seed(173)
    y = torch.randn(32, 2, 3, generator=rng)
    residual = .1+.4*y+torch.randn(y.shape, generator=rng)*.05
    moments = DiagonalMoments()
    moments.update(y, residual)
    fit = moments.fit()
    u, v = torch.randn(2, 16, 2, 3, generator=rng)
    pu, pv = apply_proximal(u, fit), apply_proximal(v, fit)
    assert bool(((pu-pv).flatten(1).norm(dim=1) <= (u-v).flatten(1).norm(dim=1)+1e-6).all())
    metrics = heldout_step_metrics(u, v, fit)
    assert bool((metrics["nonexpansive_slack"] >= -1e-6).all())


def test_batch_merging_and_degenerate_coordinate():
    rng = torch.Generator().manual_seed(5)
    y = torch.randn(19, 7, generator=rng, dtype=torch.float64)
    y[:, 2] = 3
    e = torch.randn(19, 7, generator=rng, dtype=torch.float64)
    whole, merged = DiagonalMoments(), DiagonalMoments()
    whole.update(y, e)
    for start in range(0, len(y), 3):
        merged.update(y[start:start+3], e[start:start+3])
    a, b = whole.fit(), merged.fit()
    for key in ("center", "offset", "slope", "variance", "covariance"):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=1e-14, rtol=1e-14)
    assert a.slope[2] == 0
