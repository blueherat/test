"""Check exact IG state-carrying identities and their limits on known flows.

These are mathematical examples, not image-quality or novelty evidence.
No neural-model assets or sampling requests are modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm


OUT = Path(__file__).resolve().parents[1] / 'docs/data/ig_carrier_equalities_20260909'


def solve(field, x, t, end):
    result = solve_ivp(field, (t, end), x, method='DOP853', rtol=2e-12, atol=2e-13)
    assert result.success, result.message
    return result.y[:, -1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(202609091)
    states = rng.normal(size=(8, 2))
    diagonal = np.array([-.7, -1.1])
    b = np.array([.6, -.35])
    shear = .2

    def psi(u):
        return np.array([u[0], u[1] + shear*u[0]**2])

    def inverse(x):
        return np.array([x[0], x[1] - shear*x[0]**2])

    def jac(u):
        return np.array([[1., 0.], [2*shear*u[0], 1.]])

    def strong(t, x):
        u = inverse(x)
        return jac(u) @ (diagonal*u)

    def gap(t, x):
        return jac(inverse(x)) @ (np.exp(diagonal*t)*b)

    def strong_flow(x, t, end):
        return psi(np.exp(diagonal*(end-t))*inverse(x))

    def strong_jacobian(x, t, end):
        u = inverse(x)
        e = np.diag(np.exp(diagonal*(end-t)))
        return jac(e @ u) @ e @ np.linalg.inv(jac(u))

    rows = []
    for state_id, x in enumerate(states):
        for t in [0., .3]:
            for horizon in [.2, .7]:
                end = t+horizon
                base = strong_flow(x, t, end)
                carried = strong_jacobian(x, t, end) @ gap(t, x)
                fresh = gap(end, base)
                gap_error = float(np.linalg.norm(carried-fresh))
                assert gap_error < 2e-14
                for alpha in [.25, .5, 1., 2.]:
                    calibrated = psi(inverse(x)+alpha*horizon*np.exp(diagonal*t)*b)
                    target = solve(lambda time, z: strong(time, z)+alpha*gap(time, z), x, t, end)
                    continuation = strong_flow(calibrated, t, end)
                    endpoint_error = float(np.linalg.norm(continuation-target))
                    assert endpoint_error < 3e-10
                    raw_calibrated = x+alpha*horizon*gap(t, x)
                    raw_error = float(np.linalg.norm(strong_flow(raw_calibrated, t, end)-target))
                    rows.append(dict(state_id=state_id, t=t, horizon=horizon, alpha=alpha,
                                     gap_norm=float(np.linalg.norm(gap(t, x))),
                                     gap_transport_error=gap_error,
                                     exact_carrier_endpoint_error=endpoint_error,
                                     linear_write_endpoint_error=raw_error))
    assert min(row['gap_norm'] for row in rows) > .1
    assert max(row['linear_write_endpoint_error'] for row in rows) > .01
    with (OUT/'nonlinear_carrier_checks.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader(); writer.writerows(rows)

    # The generic carrier exists, but its first-order direction is a future integral.
    A = np.diag([-.7, -1.8])
    B = np.array([[.25, .8], [-.35, -.15]])
    horizon = .8
    M, quadrature_error = quad_vec(lambda t: expm(-A*t) @ B @ expm(A*t),
                                  0., horizon, epsabs=1e-13, epsrel=1e-13)
    scalar = float(np.sum(M*B)/np.sum(B*B))
    residual = float(np.linalg.norm(M-scalar*B)/np.linalg.norm(M))
    assert residual > .1
    errors = []
    for alpha in [.1, .05, .025, .0125]:
        exact = expm(-A*horizon) @ expm((A+alpha*B)*horizon)
        approx = np.eye(2)+alpha*M
        matrix_error = float(np.linalg.norm(exact-approx))
        solved = np.column_stack([solve(lambda t, z: (A+alpha*B) @ z,
                                        np.eye(2)[:, k], 0., horizon) for k in range(2)])
        endpoint_error = float(np.linalg.norm(expm(A*horizon) @ exact-solved))
        assert endpoint_error < 1e-10
        errors.append(dict(alpha=alpha, first_order_carrier_error=matrix_error,
                           exact_conjugacy_endpoint_error=endpoint_error))
    orders = [float(np.log2(errors[k]['first_order_carrier_error']/
                            errors[k+1]['first_order_carrier_error'])) for k in range(3)]
    assert min(orders) > 1.98
    noncommuting = dict(A=A.tolist(), B=B.tolist(), horizon=horizon,
        innovation_matrix=(B @ A-A @ B).tolist(), pulled_back_integral=M.tolist(),
        quadrature_error=float(quadrature_error), best_scalar_of_current_gap=scalar,
        relative_residual_after_best_scalar=residual, checks=errors, remainder_orders=orders,
        scope='Obstructs a scalar multiple of the current gap as a single initial write. '
              'Does not assert impossibility of reproducing ordinary IG with an IG schedule.')
    write_json(OUT/'noncommuting_example.json', noncommuting)

    # Equal current heads/gaps do not determine later guidance effects.
    future = []
    x = np.array([.2, -.4])
    for sign in [-1., 1.]:
        end = solve(lambda t, z: sign*t*b, x, 0., 1.)
        expected = x+sign*.5*b
        assert np.max(np.abs(end-expected)) < 2e-12
        future.append(dict(sign=sign, gap_at_time_zero=[0., 0.],
                           endpoint=end.tolist(), initial_carrier=(expected-x).tolist()))
    write_json(OUT/'same_current_gap_different_future.json', future)

    # Exact Gaussian FM fields with the same noise convention; two data worlds.
    def gaussian_velocity(std, t):
        variance = (1-t)**2+(std*t)**2
        return ((1+std**2)*t-1)/variance

    alpha = .25
    ig_std = float(solve(lambda t, z: ((1+alpha)*gaussian_velocity(1., t)
                                      -alpha*gaussian_velocity(2., t))*z,
                         np.array([1.]), 0., 1.)[0])
    assert abs(ig_std-2**(-alpha)) < 2e-11
    worlds = []
    for target in [.9, 1.1]:
        baseline_error = (1-target)**2
        weak_error = (2-target)**2
        guided_error = (ig_std-target)**2
        assert baseline_error < weak_error
        worlds.append(dict(target_std=target, strong_std=1., weak_std=2.,
                           ig_std=ig_std, alpha=alpha, strong_squared_w2=baseline_error,
                           weak_squared_w2=weak_error, ig_squared_w2=guided_error,
                           ig_improves=guided_error < baseline_error))
    assert [world['ig_improves'] for world in worlds] == [True, False]
    write_json(OUT/'quality_nonidentification.json', dict(worlds=worlds,
        scope='Both worlds rate the same strong model better than the same weak model. '
              'The same positive IG changes quality in opposite directions. '
              'One-dimensional population Gaussian W2, not image FID.'))

    audit = dict(passed=True, nonlinear_checks=len(rows),
        max_gap_transport_error=max(row['gap_transport_error'] for row in rows),
        max_exact_carrier_endpoint_error=max(row['exact_carrier_endpoint_error'] for row in rows),
        min_nonzero_gap_norm=min(row['gap_norm'] for row in rows),
        first_order_min_remainder_order=min(orders),
        scalar_current_gap_relative_residual=residual,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Independent numerical checks of derived identities and exact counterexamples; '
              'no neural-model, new-method, image-quality, or novelty validation.',
        research_goal_achieved=False)
    write_json(OUT/'audit.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
