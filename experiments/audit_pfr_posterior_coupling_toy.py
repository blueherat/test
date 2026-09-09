"""Finite ancestral posterior coupling: Gaussian identity and mixture bias.

Existing reverse-martingale identity; no novelty or quality claim.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm


def posterior(z, t, weights, means, variances):
    a = 1 - t
    variance = a * a * variances + t * t
    logits = np.log(weights) - .5 * (np.log(variance) + (np.asarray(z)[..., None] - a * means) ** 2 / variance)
    probabilities = np.exp(logits - logsumexp(logits, axis=-1, keepdims=True))
    component_means = means + a * variances / variance * (np.asarray(z)[..., None] - a * means)
    component_variances = variances * t * t / variance
    return probabilities, component_means, component_variances


def denoise(z, t, params):
    w, m, _ = posterior(z, t, *params)
    return np.sum(w * m, axis=-1)


def coefficients(t, r):
    a, b = 1 - t, 1 - r
    aa = a / b * r * r / (t * t)
    bb = b - aa * a
    variance = r * r * (1 - a * a * r * r / (b * b * t * t))
    assert variance >= 0
    return aa, bb, variance


def main():
    started = time.perf_counter()
    nodes, weights = roots_hermitenorm(128)
    weights /= np.sqrt(2 * np.pi)
    rng = np.random.default_rng(202609432)
    gaussian_errors, rows = [], []
    for mean in [-1., 0., 2.]:
        for variance in [.25, 1., 4., 16.]:
            params = (np.array([1.]), np.array([mean]), np.array([variance]))
            for t in [.6, .8, .95, 1.]:
                r = t - 1 / 32
                aa, bb, vv = coefficients(t, r)
                for z in [-2., 0., 2.]:
                    now = denoise(z, t, params)
                    center = aa * z + bb * now
                    noise = rng.normal(size=16)
                    paired = .5 * (denoise(center + np.sqrt(vv) * noise, r, params) +
                                    denoise(center - np.sqrt(vv) * noise, r, params))
                    gaussian_errors.extend(np.abs(paired - now).tolist())
    assert max(gaussian_errors) < 1e-12
    params = (np.array([.3, .7]), np.array([-2., 1.]), np.array([.25, .5]))
    for t in [.6, .8, .95, 1.]:
        for z in [-2., 0., 2.]:
            now = denoise(z, t, params)
            pw, pm, pv = posterior(z, t, *params)
            for h in [1 / 32, 1 / 64, 1 / 128]:
                r = t - h
                aa, bb, vv = coefficients(t, r)
                component_centers = aa * z + bb * pm
                component_var = vv + bb * bb * pv
                # Exact reverse transition is a posterior Gaussian mixture.
                exact = np.sum(pw * np.sum(weights[:, None] * denoise(
                    component_centers[None, :] + nodes[:, None] * np.sqrt(component_var)[None, :], r, params), axis=0))
                assert abs(exact - now) < 1e-10
                center = aa * z + bb * now
                approximate = np.sum(weights * denoise(center + np.sqrt(vv) * nodes, r, params))
                noise = rng.normal(size=4096)
                estimates = .5 * (denoise(center + np.sqrt(vv) * noise, r, params) +
                                   denoise(center - np.sqrt(vv) * noise, r, params))
                rows.append(dict(t=t, z=z, h=h, now=float(now), exact_residual=float(now-exact),
                                 closure_bias=float(now-approximate),
                                 single_antithetic_pair_sd=float(estimates.std(ddof=1)),
                                 mc_mean_error=float(estimates.mean()-approximate)))
    result = dict(complete=True, gaussian_max_error=max(gaussian_errors), gaussian_checks=len(gaussian_errors),
                  mixture_rows=rows, seconds=time.perf_counter()-started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Exact 1D Gaussian/Gaussian-mixture denoisers only; no learned-model error or quality claims.')
    out = Path('experiments/results/terminal_defect_20260908/pfr_posterior_coupling_toy.json')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    for h in [1/32, 1/64, 1/128]:
        selected = [r for r in rows if r['h']==h]
        print(dict(h=h, maximum_closure_bias=max(abs(r['closure_bias']) for r in selected),
                   median_pair_sd=float(np.median([r['single_antithetic_pair_sd'] for r in selected]))))
    print(dict(complete=True, gaussian_max_error=result['gaussian_max_error'], seconds=result['seconds']))


if __name__ == '__main__':
    main()
