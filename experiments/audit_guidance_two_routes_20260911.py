"""CPU-only mathematical audits for the two guidance research routes.

The examples are exact low-dimensional flows or linear distribution transforms.
They do not implement the authors' image samplers and do not estimate image FID.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/data/guidance_two_routes_20260911'


def csv_out(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_out(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def metric_roots(matrix):
    eigenvalues, vectors = np.linalg.eigh(matrix)
    assert np.min(eigenvalues) > 0
    return (vectors * np.sqrt(eigenvalues)) @ vectors.T, (vectors / np.sqrt(eigenvalues)) @ vectors.T


def level_set_candidate(x, preserve_matrix, improve_matrix, radius):
    """Two exact ellipsoid rotations; choose the lower other quadratic loss."""
    root, inv_root = metric_roots(preserve_matrix)

    def state(angle):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        return inv_root @ rotation @ root @ x

    candidates = []
    for sign in [-1., 1.]:
        theta = brentq(lambda a: np.linalg.norm(state(sign * a) - x) - radius, 0., .2, xtol=1e-14)
        z = state(sign * theta)
        candidates.append(z)
    z = min(candidates, key=lambda v: v @ improve_matrix @ v)
    preserved = abs(float(z @ preserve_matrix @ z - x @ preserve_matrix @ x))
    assert preserved < 1e-12
    assert abs(np.linalg.norm(z - x) - radius) < 1e-12
    return z, preserved


def local_future_relation():
    au = np.array([[-.2, 1.5], [-.4, .1]])
    ac = np.array([[.3, .2], [.8, -.15]])
    gap_matrix = ac - au
    # Vector-field bracket [u,c] = Dc*u - Du*c, fixing the convention.
    bracket_matrix = ac @ au - au @ ac
    x = np.array([.7, -.4])
    local_metric = gap_matrix.T @ gap_matrix
    local_gradient = local_metric @ x
    rows, interventions = [], []
    radius = .01
    for horizon in [.5, .25, .125, .0625, .03125, .015625, .0078125, .00390625]:
        u, c = expm(horizon * au), expm(horizon * ac)
        relative = np.linalg.solve(u, c)
        effective = np.real_if_close(logm(relative) / horizon)
        assert not np.iscomplexobj(effective)
        bch_approx = gap_matrix + .5 * horizon * bracket_matrix
        zeroth_error = float(np.linalg.norm(effective - gap_matrix))
        first_error = float(np.linalg.norm(effective - bch_approx))
        future_matrix = (c - u) / horizon
        future_metric = future_matrix.T @ future_matrix
        future_gradient = future_metric @ x
        gradient_rows = np.stack([local_gradient / np.linalg.norm(local_gradient), future_gradient / np.linalg.norm(future_gradient)])
        singular_values = np.linalg.svd(gradient_rows, compute_uv=False)
        cosine = float(gradient_rows[0] @ gradient_rows[1])
        rows.append(dict(
            horizon=horizon,
            effective_generator_local_error=zeroth_error,
            effective_generator_bch_error=first_error,
            gradient_cosine=cosine,
            separation_sine=float(np.sqrt(max(0., 1. - cosine ** 2))),
            sigma_min_over_max=float(singular_values[-1] / singular_values[0]),
            local_future_normalized_gradient_difference=float(np.linalg.norm(future_gradient - local_gradient)),
        ))
        for name, preserve, improve in [
            ('future_only', local_metric, future_metric),
            ('local_only', future_metric, local_metric),
        ]:
            z, equality_error = level_set_candidate(x, preserve, improve, radius)
            delta_local = .5 * float(z @ local_metric @ z - x @ local_metric @ x)
            delta_future = .5 * float(z @ future_metric @ z - x @ future_metric @ x)
            assert (delta_future if name == 'future_only' else delta_local) < 0
            interventions.append(dict(horizon=horizon, intervention=name, state_displacement=float(np.linalg.norm(z - x)), delta_local_loss=delta_local, delta_future_loss_normalized=delta_future, preserved_loss_abs_error=.5 * equality_error))
    # Check asymptotic orders on the final four intervals, independently of a
    # particular fitted coefficient.
    local_orders, bch_orders = [], []
    for previous, current in zip(rows[-4:-1], rows[-3:]):
        local_orders.append(float(np.log2(previous['effective_generator_local_error'] / current['effective_generator_local_error'])))
        bch_orders.append(float(np.log2(previous['effective_generator_bch_error'] / current['effective_generator_bch_error'])))
    assert min(local_orders) > .95 and max(local_orders) < 1.05
    assert min(bch_orders) > 1.95 and max(bch_orders) < 2.05
    csv_out('local_future_identifiability.csv', rows)
    csv_out('matched_finite_interventions.csv', interventions)
    result = dict(
        unconditional_matrix=au.tolist(), conditional_matrix=ac.tolist(), x=x.tolist(),
        local_generator=gap_matrix.tolist(), bracket_matrix=bracket_matrix.tolist(),
        local_error_orders=local_orders, bracket_corrected_error_orders=bch_orders,
        note='Exact linear flows. Future loss is normalized by H squared. Local loss is raw gap energy, not the full history-dependent CTRL sliding variable.',
    )
    json_out('local_future_audit.json', result)
    return result


def selective_contraction():
    """Same error contraction, different preservation of target variation."""
    alpha = .5
    rows = []
    for beta in [1., .8, .5]:
        for iterations in range(13):
            # Initial X ~ N(0,I2). Target is delta_{x1=1} x N(0,1).
            # T(x)=(1+alpha*(x1-1), beta*x2); exact Gaussian W2^2.
            normal_mean = -(alpha ** iterations)
            normal_std = alpha ** iterations
            tangent_std = beta ** iterations
            normal_error = normal_mean ** 2 + normal_std ** 2
            tangent_w2 = (1. - tangent_std) ** 2
            rows.append(dict(
                alpha=alpha, beta=beta, iterations=iterations,
                local_error_squared=normal_error,
                # U=id and C_H=(1+exp(-H)*(x1-1),x2), H=log(2),
                # generated by u=0, c=(-(x1-1),0).
                future_error_squared=.25 * normal_error,
                retained_tangent_std=tangent_std,
                target_wasserstein_squared=normal_error + tangent_w2,
                excess_tangent_wasserstein_squared=tangent_w2,
                log_abs_calibration_jacobian=iterations * np.log(alpha * beta),
                log_abs_tangent_jacobian=iterations * np.log(beta),
            ))
    for k in range(13):
        group = [r for r in rows if r['iterations'] == k]
        assert len({r['local_error_squared'] for r in group}) == 1
        assert len({r['future_error_squared'] for r in group}) == 1
    selected = [r for r in rows if r['iterations'] == 4]
    assert selected[0]['target_wasserstein_squared'] < selected[1]['target_wasserstein_squared'] < selected[2]['target_wasserstein_squared']
    csv_out('selective_contraction.csv', rows)
    json_out('selective_contraction_scope.json', dict(
        initial='N(0,I_2)', target='delta(x_1-1) times N(x_2;0,1)',
        transformation='T(x)=(1+alpha*(x1-1), beta*x2)',
        local_error='g=(-(x1-1),0)', future_error='R=C_H-U=(-0.5*(x1-1),0), H=log(2)',
        fields='u=0, c=(-(x1-1),0); U=id and C_H=(1+exp(-H)*(x1-1),x2)',
        caveat='An exact linear control example, not a canonical Gaussian FM pair or the author image samplers. beta=1 is the exact ideal forward-backward map here; beta<1 adds an independent tangent contraction. The target is deliberately singular; a positive-thickness target also requires preserving its normal variance.',
        manifold_tangent_identity='A differentiable map fixing every point of M must satisfy DT(x)v=v for v in T_xM.',
        global_contraction_bound='For a shared r-Lipschitz map, E||T^K(X)-T^K(Xprime)||^2 <= r^(2K) E||X-Xprime||^2. No new noise and the same map are required.',
    ))
    return selected


def map_density_relation():
    # Linear exact forward-backward map: log volume and integrated divergence.
    au = np.array([[-.2, 1.5], [-.4, .1]])
    ac = np.array([[.3, .2], [.8, -.15]])
    rows = []
    for h in [.05, .2, .5]:
        relative = np.linalg.solve(expm(h * au), expm(h * ac))
        sign, logdet = np.linalg.slogdet(relative)
        predicted = h * np.trace(ac - au)
        assert sign > 0 and abs(logdet - predicted) < 1e-12
        rows.append(dict(horizon=h, logdet_relative_map=float(logdet), integrated_divergence_difference=float(predicted), absolute_error=float(abs(logdet - predicted))))
    csv_out('map_density_identity.csv', rows)
    return max(row['absolute_error'] for row in rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    relations = local_future_relation()
    selected = selective_contraction()
    volume_error = map_density_relation()
    json_out('audit.json', dict(
        all_checks_passed=True,
        script=str(Path(__file__).resolve()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bch=relations,
        selected_contraction_examples=selected,
        volume_identity_max_error=volume_error,
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='audit.json'},
    ))
    print(json.dumps(dict(passed=True, output=str(OUT), local_generator_orders=relations['local_error_orders'], bch_orders=relations['bracket_corrected_error_orders'], contraction_comparison=selected, volume_identity_error=volume_error),indent=2))


if __name__ == '__main__':
    main()
