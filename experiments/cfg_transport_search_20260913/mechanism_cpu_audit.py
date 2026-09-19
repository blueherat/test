"""CPU audit of supervised transport calibration and a density-only counterexample.

Run: python experiments/cfg_transport_search_20260913/mechanism_cpu_audit.py
These are conditional FM velocity-risk diagnostics, NOT image-quality experiments.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def project_out(vec, basis):
    orth = []
    for col in basis:
        v = col.copy()
        for _ in range(2):
            for u in orth:
                v -= np.sum(v * u, axis=1, keepdims=True) * u
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        orth.append(np.divide(v, norms, out=np.zeros_like(v), where=norms > 1e-11))
    out = vec.copy()
    for _ in range(2):
        for u in orth:
            out -= np.sum(out * u, axis=1, keepdims=True) * u
    return out


def matrix_flow(field, t, horizon, dim, rtol=1e-11):
    sol = solve_ivp(
        lambda s, f: (field(s) @ f.reshape(dim, dim)).ravel(),
        (t, t + horizon), np.eye(dim).ravel(), method="DOP853",
        rtol=rtol, atol=rtol * 0.01,
    )
    assert sol.success
    return sol.y[:, -1].reshape(dim, dim)


def run_case(bias_scale, n, cal_seed, eval_seed):
    dim, t, horizon, gamma = 6, 0.45, 0.15, 1.0
    cond_var = np.array([0.4, 0.8, 1.3, 2.0, 3.2, 5.0])
    mean_var = np.array([0.8, 0.5, 1.1, 0.4, 1.5, 0.7])
    error = bias_scale * np.array([
        [0, 0.7, 0, 0, 0, 0], [0, 0, -0.9, 0, 0, 0],
        [0, 0, 0, 0.8, 0, 0], [0, 0, 0, 0, -0.6, 0],
        [0, 0, 0, 0, 0, 0.9], [-0.7, 0, 0, 0, 0, 0],
    ])

    def oracle(s, variance):
        vt = (1 - s) ** 2 + s * s * variance
        return np.diag((s * variance - (1 - s)) / vt)

    A = lambda s: oracle(s, cond_var) + error
    U = lambda s: oracle(s, cond_var + mean_var) + error
    B = lambda s: A(s) + gamma * (A(s) - U(s))
    ca = matrix_flow(A, t, horizon, dim)
    hb = matrix_flow(B, t, horizon, dim)
    relative = np.linalg.solve(ca, hb)
    ca_fine = matrix_flow(A, t, horizon, dim, rtol=1e-13)
    hb_fine = matrix_flow(B, t, horizon, dim, rtol=1e-13)
    relative_fine = np.linalg.solve(ca_fine, hb_fine)

    def bank(seed):
        rng = np.random.default_rng(seed)
        clean = rng.normal(size=(n, dim)) * np.sqrt(cond_var)
        eps = rng.normal(size=(n, dim))
        z = t * clean + (1 - t) * eps
        v = z @ A(t).T
        g = z @ (A(t) - U(t)).T
        clean_mean = z + (1 - t) * v
        parallel = (np.sum(g * clean_mean, axis=1, keepdims=True)
                    / np.maximum(np.sum(clean_mean ** 2, axis=1, keepdims=True), 1e-20))
        apg = g - parallel * clean_mean
        # The bias-corrected one-step Euler mixed loop is in span(g, secant).
        secant = -horizon * gamma * g @ A(t + horizon).T
        mixed = z @ (relative - np.eye(dim)).T / horizon
        q = project_out(mixed, [g, apg, secant])
        euler_mixed = gamma * g + secant
        euler_left = project_out(euler_mixed, [g, apg, secant])
        return dict(z=z, target=clean-eps, pred=v,
                    truth=z @ oracle(t, cond_var).T,
                    q=q, mixed=mixed, g=g,
                    euler_remainder=np.linalg.norm(euler_left, axis=1))

    cal, ev = bank(cal_seed), bank(eval_seed)
    norm = np.sqrt(np.mean(np.sum(cal['q'] ** 2, axis=1)))
    for data in [cal, ev]:
        data['q'] = data['q'] / max(norm, 1e-12)
    residual = cal['target'] - cal['pred']
    coefficient = np.sum(cal['q'] * residual) / np.sum(cal['q'] ** 2)
    oracle_coefficient = (np.sum(cal['q'] * (cal['truth']-cal['pred']))
                          / np.sum(cal['q'] ** 2))
    fit = ev['pred'] + coefficient * ev['q']
    opposite = ev['pred'] - coefficient * ev['q']
    mean_sq = lambda x: float(np.mean(np.sum(x*x, axis=1)))
    oracle_error_before = mean_sq(ev['pred'] - ev['truth'])
    oracle_error_after = mean_sq(fit - ev['truth'])
    return dict(
        bias_scale=bias_scale, calibration_seed=cal_seed, evaluation_seed=eval_seed,
        n_calibration=n, n_evaluation=n, coefficient=float(coefficient),
        coefficient_using_exact_mean=float(oracle_coefficient),
        heldout_observed_fm_risk_before=mean_sq(ev['pred']-ev['target']),
        heldout_observed_fm_risk_after=mean_sq(fit-ev['target']),
        heldout_true_velocity_error_before=oracle_error_before,
        heldout_true_velocity_error_after=oracle_error_after,
        heldout_true_velocity_error_wrong_sign=mean_sq(opposite-ev['truth']),
        residual_fraction_rms=float(norm / np.sqrt(mean_sq(cal['mixed']))),
        max_one_step_euler_residual=float(np.max(cal['euler_remainder'])),
        relative_map_solver_change=float(np.linalg.norm(relative-relative_fine)),
        interpretation='velocity risk only; q is deterministic in z,c,t; no quality claim',
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100000)
    parser.add_argument('--output', type=Path, default=Path(
        '/home/zhoushunyu/data/eqvae/experiments/cfg_transport_search_20260913/mechanism_cpu'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [run_case(bias, args.n, 173, seed)
               for bias in [0.0, 0.4] for seed in [811, 812]]
    density = dict(target='N(0, I_6)', operation='x -> 0.5*x',
                   exact_expected_log_density_gain=6*(1-0.5**2)/2,
                   exact_wasserstein_squared_from_target=6*(1-0.5)**2,
                   output_variance_per_dimension=0.25,
                   inference='conditional density ascent can accept destructive contraction')
    output = dict(cases=results, density_counterexample=density,
                  joint_distribution='M~N(0,diag(mean_var)), X|M~N(M,diag(cond_var)); condition M=0',
                  dynamics='canonical Gaussian FM with a known shared additive linear field error',
                  caution='normalization and coefficient use only calibration data; eval seeds independent')
    path = args.output / 'checks.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
