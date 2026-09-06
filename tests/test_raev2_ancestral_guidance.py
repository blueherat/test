import numpy as np
import torch

from experiments.raev2_ancestral_guidance import ancestral_step, coefficients


def test_conditional_channel_mean_and_variance():
    # Given clean x, reconstruct the joint Gaussian forward Markov channel.
    for t, s in ((1., .998), (.9, .7), (.5, .25), (.08, 0.)):
        r, q, std = coefficients(t, s)
        a, b = 1-t, 1-s
        assert np.isclose(r*a+q, b)
        assert np.isclose(r*r*t*t+std*std, s*s)
        assert np.isclose(r*t*t, (a/b)*s*s)


def test_partial_refresh_preserves_conditional_marginal():
    for eta in (0., .5, 1.):
        for t, s in ((1., .99), (.8, .6), (.2, .1), (.1, 0.)):
            r, q, std = coefficients(t, s, eta)
            assert np.isclose(r*(1-t)+q, 1-s)
            assert np.isclose((r*t)**2+std**2, s*s)


def test_zero_refresh_is_euler_and_endpoint_is_clean():
    gen = torch.Generator().manual_seed(17)
    x, g, n = [torch.randn(3, 4, generator=gen) for _ in range(3)]
    expected = x-(.8-.7)*((x-g)/.8)
    assert torch.equal(ancestral_step(x,g,.8,.7,n,eta=0.), expected)
    for eta in (0., .5, 1.):
        torch.testing.assert_close(ancestral_step(x,g,.08,0.,n,eta=eta),g)


def test_positive_reverse_covariance_for_nondegenerate_gaussian():
    # Plug-in posterior mean omits clean posterior variance; expose the error.
    t, s, v = .8, .6, 2.
    r, q, std = coefficients(t,s)
    vt = (1-t)**2*v+t*t
    posterior_gain = (1-t)*v/vt
    mean_variance = (r+q*posterior_gain)**2*vt+std*std
    omitted = q*q*v*t*t/vt
    assert omitted > 0
    assert np.isclose(mean_variance+omitted, (1-s)**2*v+s*s)
