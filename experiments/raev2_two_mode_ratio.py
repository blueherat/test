"""Difference of two Gaussian posterior means, in fixed spatial eigenspaces."""

import math
from scipy.optimize import brentq


def euler_terminal_variance(prior_variance, grid):
    """Actual finite Euler endpoint variance of the scalar Gaussian denoiser."""
    variance=1.
    for t,s in zip(grid[:-1],grid[1:]):
        a=1-t
        gain=a*prior_variance/(t*t+a*a*prior_variance)
        factor=1-(t-s)*(1-gain)/max(t,.05)
        variance*=factor*factor
    return variance


def fit_euler_prior_variance(endpoint_variance, grid):
    """Invert the monotone finite sampler variance map, without image metrics."""
    if endpoint_variance<=0:raise ValueError('positive variance required')
    lo,hi=1e-12,max(endpoint_variance,1.)
    while euler_terminal_variance(hi,grid)<endpoint_variance:
        hi*=2
        if hi>1e12:raise ValueError('target outside finite-grid Gaussian range')
    root=brentq(lambda logv:euler_terminal_variance(math.exp(logv),grid)-endpoint_variance,
                math.log(lo),math.log(hi),xtol=1e-13)
    return math.exp(root)


def posterior_difference(t, real_variance, model_variance):
    a=1-t
    vp=t*t+a*a*real_variance
    vq=t*t+a*a*model_variance
    return a*t*t*(real_variance-model_variance)/(vp*vq)


def two_mode_correction(state, t, calibration):
    p,q=calibration['real'],calibration['generated']
    dc=state.mean((-2,-1),keepdim=True)
    ac=state-dc
    return (posterior_difference(t,p['dc_variance'],q['dc_variance'])*dc+
            posterior_difference(t,p['ac_variance'],q['ac_variance'])*ac)
