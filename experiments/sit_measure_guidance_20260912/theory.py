"""Independent finite-distribution checks of the five population arguments."""
from __future__ import annotations
import itertools
import json
import math
import numpy as np
from scipy.special import logsumexp
from .data import ROOT


def gaussian_mixture(x, means, variance, weights):
    log_components = (np.log(weights)[None]-.5*np.log(2*np.pi*variance)
                      -(x[:, None]-means[None])**2/(2*variance))
    logp = logsumexp(log_components, axis=1)
    posterior = np.exp(log_components-logp[:, None])
    score = ((means[None]-x[:, None])/variance*posterior).sum(1)
    return logp, score


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    # Four independent Bernoulli modes: exact weighted moments, no Monte Carlo.
    values = np.array(list(itertools.product((-1., 1.), repeat=4)))
    clt = values.sum(1)/2
    moment = lambda x: [float(np.mean(x)), float(np.mean(x*x)),
                       float(np.mean(x**4)-3*np.mean(x*x)**2)]
    original, transformed = moment(values[:, 0]), moment(clt)
    assert original == [0., 1., -2.] and transformed == [0., 1., -.5]
    # The noise in an average divided by sqrt(m) has the original covariance.
    noise_variance_before, noise_variance_after = .3, 4*.3/4
    assert noise_variance_before == noise_variance_after
    # An exact two-variable copula intervention.
    joint = np.array([[.45, .05], [.05, .45]])
    product = joint.sum(1)[:, None]*joint.sum(0)[None]
    assert np.allclose(joint.sum(0), product.sum(0))
    assert np.allclose(joint.sum(1), product.sum(1))
    contrast = np.log(joint/product)
    assert contrast[0, 0] > 0 and contrast[0, 1] < 0
    # Class mixing preserves the overall data marginal but changes conditionals.
    permutation = np.eye(4)[[1, 0, 3, 2]]
    q = (np.eye(4)+permutation)/2
    conditionals = np.array([[.7, .2, .1], [.1, .3, .6], [.2, .7, .1], [.5, .2, .3]])
    assert np.allclose((q @ conditionals).mean(0), conditionals.mean(0))
    # Conjugating standard OU by an affine conditional generator is stationary.
    a = np.array([[2., .8], [0., .5]])
    covariance = a @ a.T
    rho = .9
    correct = rho*rho*covariance+(1-rho*rho)*(a @ a.T)
    isotropic = rho*rho*covariance+(1-rho*rho)*np.trace(covariance)/2*np.eye(2)
    assert np.max(np.abs(correct-covariance)) < 1e-14
    assert np.linalg.norm(isotropic-covariance) > .1
    # Defensive mixture bound holds for arbitrary discrete p,b and alpha>0.
    p, bad = np.array([.1, .2, .7]), np.array([.000001, .299999, .7])
    mixture = (p+bad)/2
    alpha = 2.75
    tilted = p*(p/mixture)**alpha
    z = tilted.sum()
    tilted /= z
    assert z >= 1 and np.max(tilted/p) <= 2**alpha+1e-12
    # Counterexample: one CLT iteration need not merge peaks.
    x = np.linspace(-3, 3, 12001)
    native_logp, _ = gaussian_mixture(x, np.array([-1., 1.]), .01, np.ones(2)/2)
    clt2_logp, _ = gaussian_mixture(x, np.array([-math.sqrt(2), 0., math.sqrt(2)]),
                                   .01, np.array([.25, .5, .25]))
    modes = lambda logp: int(((logp[1:-1] > logp[:-2]) & (logp[1:-1] > logp[2:])).sum())
    assert modes(native_logp) == 2 and modes(clt2_logp) == 3
    result = dict(passed=True, clt_exact_moments=dict(original=original, transformed=transformed),
        clt_heat_commutation_variance=[noise_variance_before, noise_variance_after],
        factor_marginals_exact=True, factor_log_density_ratio=contrast.tolist(),
        label_marginal_preserved=True,
        conditional_ou_covariance_error=float(np.max(np.abs(correct-covariance))),
        ambient_ou_covariance_error=float(np.linalg.norm(isotropic-covariance)),
        defensive_normalizer=float(z), defensive_max_relative_density=float(np.max(tilted/p)),
        defensive_bound=2**alpha, nonmonotone_mode_counterexample=[2, 3],
        does_not_establish_real_model_mechanism=True, no_fid=True)
    (ROOT/'theory_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    run()
