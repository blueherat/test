"""CPU check of the actual Heun auxiliary / Euler main lifting expansion.

The fields and analytic derivatives are independent of neural-model sampling.
This checks local numerical differences, not image quality.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.check_lifting_modified_flow_20260909 import (
    AS, AW, BS, BW, strong, weak, gap, gap_jacobian,
)

OUT = Path(__file__).resolve().parents[1] / 'docs/data/lifting_modified_flow_20260909'


def flow_numerical(field, x, grid, method):
    z = x.copy()
    for t, u in zip(grid[:-1], grid[1:]):
        h = u - t
        first = field(t, z)
        if method == 'heun':
            second = field(u, z + h * first)
            z = z + .5 * h * (first + second)
        else:
            assert method == 'euler'
            z = z + h * first
    return z


def lift_numerical(x, grid, alpha, m):
    z = x.copy()
    for _ in range(m):
        target = flow_numerical(strong, z, grid, 'heun')
        inverse = flow_numerical(weak, target, grid[::-1], 'heun')
        z = z + (alpha / m) * (inverse - z)
    return flow_numerical(strong, z, grid, 'euler')


def material_derivative_difference(t, x, alpha):
    strong_factor = 1 - np.tanh(AS @ x + BS * t) ** 2
    weak_factor = 1 - np.tanh(AW @ x + BW * t) ** 2
    js = strong_factor[:, None] * AS
    jd = gap_jacobian(t, x)
    dt_gap = strong_factor * BS - weak_factor * BW
    s = strong(t, x)
    d = gap(t, x)
    return alpha * (dt_gap + js @ d + jd @ s) + alpha**2 * (jd @ d)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(202609950)
    states = rng.uniform(-1, 1, (8, 2))
    times = rng.uniform(.15, .65, 8)
    profiles = {
        'one_step': [1.],
        'four_equal': [.25] * 4,
        'four_unequal': [.1, .2, .3, .4],
    }
    alphas = [.25, .5, 1., 1.5, 2.]
    hs = [.08, .04, .02, .01, .005]
    rows = []
    for direction in (1, -1):
        for index, (x, t) in enumerate(zip(states, times)):
            jdd = gap_jacobian(t, x) @ gap(t, x)
            for alpha in alphas:
                material_difference = material_derivative_difference(t, x, alpha)
                for m in (1, 2):
                    for name, fractions in profiles.items():
                        q = float(np.square(fractions).sum())
                        coefficient = .5 * ((alpha - alpha**2 / m) * jdd + q * material_difference)
                        for h_abs in hs:
                            h = direction * h_abs
                            grid = t + h * np.r_[0., np.cumsum(fractions)]
                            lifting = lift_numerical(x, grid, alpha, m)
                            ordinary = flow_numerical(lambda u, z: strong(u, z) + alpha * gap(u, z),
                                                      x, grid, 'euler')
                            difference = lifting - ordinary
                            rows.append(dict(direction=direction, state=index, alpha=alpha, m=m,
                                profile=name, q=q, h=h_abs,
                                predicted_norm=float(np.linalg.norm(coefficient)),
                                raw_error=float(np.linalg.norm(difference)),
                                corrected_error=float(np.linalg.norm(difference - h*h*coefficient))))
    with (OUT / 'discrete_checks.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summaries = []
    for direction in (1, -1):
        for alpha in alphas:
            for m in (1, 2):
                for name in profiles:
                    subset = [r for r in rows if (r['direction'], r['alpha'], r['m'], r['profile']) ==
                              (direction, alpha, m, name)]
                    errors = {h: np.mean([r['corrected_error'] for r in subset if r['h'] == h]) for h in hs}
                    order = float(np.log2(errors[.01] / errors[.005]))
                    summaries.append(dict(direction=direction, alpha=alpha, m=m, profile=name,
                                          corrected_order=order, passed=order > 2.85))
    audit = dict(passed=all(r['passed'] for r in summaries), seed=202609950, trials=len(rows),
                 states=states.tolist(), times=times.tolist(), profiles=profiles,
                 minimum_corrected_order=min(r['corrected_order'] for r in summaries), summaries=summaries,
                 formula='h^2/2*((alpha-alpha^2/m)*J_D D + q*(A_F-A_S)), A_v=partial_t v+J_v v',
                 scope='Heun auxiliary, Euler main; smooth local operator only; not neural image quality',
                 research_goal_achieved=False)
    (OUT / 'discrete_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('passed', 'trials', 'minimum_corrected_order')}), flush=True)
    assert audit['passed'], summaries


if __name__ == '__main__':
    main()
