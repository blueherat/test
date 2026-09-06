#!/usr/bin/env python3
"""Fixed S1 checks of finite-source transport; no model or quality sampling."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/moser_finite_source_v1')


def identity(path):
    path = Path(path)
    b = path.read_bytes()
    return dict(path=str(path), size_bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started, cpu = time.perf_counter(), time.process_time()
    protocol = ROOT / 'docs/RAEV2_MOSER_FINITE_SOURCE_PROTOCOL_20260906_ZH.md'
    request = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   script=identity(__file__), protocol=identity(protocol),
                   a=0.5, quadrature_nodes=1024, solver='DOP853', rtol=1e-10,
                   atol=1e-12, coefficients=[0, -1, -10, -100],
                   no_model_calls=True, no_fid=True,
                   selection='Analytic derivation preceded fixed numerical verification; not a blind test.')
    write(args.output_dir / 'request.json', request)
    a, n = request['a'], request['quadrature_nodes']
    x0 = (np.arange(n, dtype=np.float64) + 0.5) * (2 * np.pi / n)
    x_eval = np.linspace(0.0, 1.0, 5)
    exact = solve_ivp(lambda t, x: -a * np.sin(x) / (1 + t * a * np.cos(x)),
                      (0.0, 1.0), x0, method='DOP853', rtol=1e-10, atol=1e-12,
                      t_eval=x_eval)
    projected = solve_ivp(lambda t, x: -a * np.sin(x), (0.0, 1.0), x0,
                          method='DOP853', rtol=1e-10, atol=1e-12, t_eval=x_eval)
    if not exact.success or not projected.success:
        raise RuntimeError('Fixed solver failed; do not change parameters silently')
    invariant_error = np.max(np.abs(exact.y + a * exact.t[None, :] * np.sin(exact.y) - x0[:, None]))
    target = np.array([a / 2, 0.0])
    exact_moments = np.array([np.cos(exact.y[:, -1]).mean(), np.cos(2 * exact.y[:, -1]).mean()])
    projected_moments = np.array([np.cos(projected.y[:, -1]).mean(), np.cos(2 * projected.y[:, -1]).mean()])
    analytic_projected = np.array([np.tanh(a / 2), np.tanh(a / 2) ** 2])
    grams = np.array([(np.sin(x0) ** 2 * (1 + t * a * np.cos(x0))).mean() for t in x_eval])
    finite_source = float((np.cos(x0) * a * np.cos(x0)).mean())
    coefficients = finite_source / grams
    empirical = [dict(k=k, exact_gradient_gram=0.0, source=-2.0, objective=2.0*k)
                 for k in request['coefficients']]
    # At 0 and pi the analytic sine is exactly zero; retain FP64 arithmetic residual separately.
    sine_roundoff = float(np.sin(np.pi) ** 2)
    assert invariant_error < 2e-8
    assert np.max(np.abs(exact_moments - target)) < 2e-8
    assert np.max(np.abs(projected_moments - analytic_projected)) < 2e-8
    assert np.max(np.abs(coefficients - a)) < 1e-12
    assert abs(projected_moments[1]) > 0.05
    np.savez(args.output_dir / 'trajectories.npz', time=x_eval, initial=x0,
             exact=exact.y, projected=projected.y)
    result = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  request=identity(args.output_dir / 'request.json'),
                  exact_moser=dict(rhs_calls=exact.nfev, min_path_density=(1-a)/(2*np.pi),
                                   cdf_invariant_max_error=float(invariant_error),
                                   endpoint_fourier_moments=exact_moments.tolist(),
                                   target_fourier_moments=target.tolist()),
                  finite_galerkin=dict(rhs_calls=projected.nfev, gradient_grams=grams.tolist(),
                                      finite_source=finite_source, coefficients=coefficients.tolist(),
                                      endpoint_fourier_moments=projected_moments.tolist(),
                                      analytic_endpoint_moments=analytic_projected.tolist(),
                                      discrepancy=(projected_moments-target).tolist(),
                                      mechanism='Zero residual in one projected feature on prescribed rho does not ensure actual rho follows that path.'),
                  empirical_objective=dict(rows=empirical, analytic_loss_unbounded_below=True,
                                           fp64_sine_pi_squared=sine_roundoff,
                                           mechanism='Finite empirical gradient Gram may have nullspace carrying nonzero source.'),
                  outputs=[identity(args.output_dir / 'trajectories.npz')],
                  new_gpu_calls=0, model_calls=0, trained_parameters=0, raev2_fid_measured=False,
                  conclusion='Exact target-compatible flow succeeds in analytic positive-density case; finite sample-only implementation lacks inherited guarantee. No RAEv2 training authorized by this check.',
                  wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu)
    write(args.output_dir / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['outputs', 'request']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
