#!/usr/bin/env python3
"""Independent analytic-map / quadrature check and compact Round 4 archive.

Does not import or rerun the audited ODE solver script. No model or GPU calls.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.integrate import quad


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/moser_finite_source_v1')
PACKAGE = ROOT / 'docs/data/raev2_moser_finite_source_20260906'


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    started, cpu = time.perf_counter(), time.process_time()
    summary = json.loads((SOURCE / 'summary.json').read_text())
    request = json.loads((SOURCE / 'request.json').read_text())
    assert summary['complete'] and not summary['raev2_fid_measured']
    identities = {name: identity(SOURCE / name) for name in ('request.json', 'summary.json', 'trajectories.npz')}
    assert identities['request.json'] == summary['request']
    assert identities['trajectories.npz'] == summary['outputs'][0]
    for name in ('script', 'protocol'):
        assert identity(request[name]['path']) == request[name]
        identities['audited_' + name] = request[name]
    identities['review_script'] = identity(__file__)
    with np.load(SOURCE / 'trajectories.npz', allow_pickle=False) as saved:
        times, initial = saved['time'], saved['initial']
        exact, projected = saved['exact'], saved['projected']
    assert len(initial) == request['quadrature_nodes'] == 1024
    assert exact.shape == projected.shape == (1024, 5)
    a = request['a']
    assert a == 0.5 and np.array_equal(times, np.linspace(0, 1, 5))
    # CDF inverse is unique: derivative of x+a*t*sin(x) is >= 1-a > 0.
    lo, hi = np.zeros_like(exact), np.full_like(exact, 2 * np.pi)
    for _ in range(60):
        middle = (lo + hi) / 2
        below = middle + a * times[None, :] * np.sin(middle) < initial[:, None]
        lo, hi = np.where(below, middle, lo), np.where(below, hi, middle)
    cdf_inverse = (lo + hi) / 2
    # This atan2 form preserves the whole [0,2*pi] branch, including x0 > pi.
    tangent_map = 2 * np.arctan2(np.exp(-a * times[None, :]) * np.sin(initial[:, None] / 2),
                                 np.cos(initial[:, None] / 2))
    exact_error = float(np.max(np.abs(cdf_inverse - exact)))
    projected_error = float(np.max(np.abs(tangent_map - projected)))
    assert exact_error < 2e-8 and projected_error < 2e-8
    moment_checks = []
    for t in times:
        moments, bounds = [], []
        for n in (1, 2):
            def integrand(x):
                mapped = 2 * math.atan2(math.exp(-a * t) * math.sin(x / 2), math.cos(x / 2))
                return math.cos(n * mapped) / (2 * math.pi)
            value, err = quad(integrand, 0, 2 * math.pi, epsabs=1e-12, epsrel=1e-12)
            assert abs(value - math.tanh(a * t / 2) ** n) < 1e-12
            moments.append(value)
            bounds.append(err)
        gram, gram_err = quad(lambda x: math.sin(x)**2 * (1+a*t*math.cos(x))/(2*math.pi),
                              0, 2*math.pi, epsabs=1e-12, epsrel=1e-12)
        assert abs(gram - 0.5) < 1e-12
        moment_checks.append(dict(time=float(t), quadrature_fourier_moments=moments,
                                  quadrature_error_estimates=bounds, gradient_gram=gram,
                                  gram_quadrature_error_estimate=gram_err))
    source, source_err = quad(lambda x: a*math.cos(x)**2/(2*math.pi), 0, 2*math.pi,
                              epsabs=1e-12, epsrel=1e-12)
    assert abs(source - 0.25) < 1e-12
    assert np.allclose(moment_checks[-1]['quadrature_fourier_moments'],
                       summary['finite_galerkin']['endpoint_fourier_moments'], rtol=0, atol=2e-8)
    assert np.allclose([np.cos(cdf_inverse[:, -1]).mean(), np.cos(2*cdf_inverse[:, -1]).mean()],
                       [0.25, 0], rtol=0, atol=1e-12)
    # J(k)=k^2/4-k/4, so dJ/dk=k/2-1/4 and minimizer is k=1/2.
    variations = []
    for k in (0.0, 0.5, 1.0):
        delta = 1e-5
        objective = lambda coefficient: coefficient**2/4 - coefficient/4
        numeric = (objective(k+delta)-objective(k-delta))/(2*delta)
        expected = k/2-0.25
        assert abs(numeric-expected) < 1e-10
        variations.append(dict(k=k, finite_difference=numeric, analytic_first_variation=expected))
    for row in summary['empirical_objective']['rows']:
        assert row['exact_gradient_gram'] == 0 and row['source'] == -2
        assert row['objective'] == 2*row['k']
    for record in identities.values():
        assert identity(record['path']) == record
    result = dict(
        complete=True, round=4, created_utc=datetime.now(timezone.utc).isoformat(), sources=identities,
        scope='Independent verification of the same completed Round 4; no new experiment round.',
        all_1024_trajectories_checked_at_all_5_times=True,
        independent_cdf_inverse_max_abs_difference=exact_error,
        independent_tangent_half_map_max_abs_difference=projected_error,
        quadrature=moment_checks, finite_source=source, finite_source_quadrature_error_estimate=source_err,
        scalar_first_variation_checks=variations,
        mathematical_review=dict(
            weak_sign='delta J(phi)=integral rho grad(u).grad(phi)-integral(p-q)phi; '
                      'periodic integration by parts yields -div(rho grad u)=p-q.',
            exact_moser='rho*v=-a*sin(x)/(2*pi); -partial_x(rho*v)=a*cos(x)/(2*pi)=p-q.',
            projected_map='tan(x_t/2)=exp(-a*t)*tan(x_0/2). Its pushforward of uniform has '
                          'Fourier moments tanh(a*t/2)^n; it is not the density mixture.',
            finite_projection='The cos test residual vanishes under prescribed rho, but the actual '
                              'projected trajectory law differs; finite projection cannot inherit exact full-path transport.',
            empirical_nullspace='Exact sin(0)=sin(pi)=0 gives Gram=0, source=cos(pi)-cos(0)=-2, '
                                'J(k)=2*k with no lower bound. FP64 sin(pi) residual is not an exact nullspace.',
            issues_found=[]),
        limitations=['The smooth positive S1 population obeys Moser assumptions; the empirical Dirac '
                     'example deliberately does not and only tests transfer to finite empirical loss.',
                     'Quadrature and FP64 ODE error checks are numerical verification, not interval-certified bounds.',
                     'This provides no theorem or measured quality gain for RAEv2 and no FID measurement.'],
        new_gpu_calls=0, model_calls=0, trained_parameters=0, raev2_fid_measured=False,
        wall_seconds_before_review_write=time.perf_counter()-started,
        cpu_seconds_before_review_write=time.process_time()-cpu)
    write(SOURCE / 'independent_review.json', result)
    PACKAGE.mkdir(exist_ok=False)
    files = {}
    for name in ('request.json', 'summary.json', 'trajectories.npz', 'independent_review.json'):
        src = SOURCE / name
        files[name] = identity(src)
        shutil.copyfile(src, PACKAGE / name)
        assert identity(PACKAGE / name)['sha256'] == files[name]['sha256']
    readme = ('# Final Round 4: finite-source Moser audit\n\n'
              'Four original artifacts are copied byte-for-byte. This includes all 1024 trajectories '
              'at five fixed times, original source/protocol hashes, measured CPU costs, and an '
              'independent CDF-inverse / tangent-half-map / quadrature review. No model or GPU '
              'sampling, training or image FID was performed.\n\n'
              'The positive smooth population example validates exact target-compatible Moser flow; '
              'the finite Galerkin example disproves inheritance of the full density path from one '
              'projected weak equation. The separate empirical Dirac example demonstrates a null '
              'gradient Gram with nonzero source; it is not a counterexample to the smooth theorem.\n\n'
              'See the [frozen protocol](../../RAEV2_MOSER_FINITE_SOURCE_PROTOCOL_20260906_ZH.md).\n')
    (PACKAGE / 'README.md').write_text(readme)
    manifest = dict(version=1, round=4, created_utc=datetime.now(timezone.utc).isoformat(),
                    reviewer_and_exporter=identity(__file__), files=files,
                    copied_file_count=len(files), copied_bytes=sum(row['size_bytes'] for row in files.values()),
                    readme=identity(PACKAGE / 'README.md'), no_omitted_numerical_output=True)
    write(PACKAGE / 'manifest.json', manifest)
    print(json.dumps(dict(review=identity(SOURCE / 'independent_review.json'),
                          manifest=identity(PACKAGE / 'manifest.json'),
                          exact_map_error=exact_error, projected_map_error=projected_error), indent=2))


if __name__ == '__main__':
    main()
