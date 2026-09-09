"""Posthoc mechanism control: fit sampler endpoints, never target quality."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from experiments.audit_ag_source_condition_bridge import logdensity_score


def sample(initial, delta, early_only, n=200):
    def velocity(z, t, gamma):
        _, strong = logdensity_score(z, .35, t)
        _, weak = logdensity_score(z, 1.2, t)
        return -t * (strong + gamma * (strong - weak))

    z = initial.copy()
    # Split exactly at sigma=2 so the piecewise strength has no RK ambiguity.
    for high, low, steps in [(3., 2., n // 3), (2., 0., n - n // 3)]:
        gamma = .78 + (delta if high == 3. or not early_only else 0.)
        grid = np.linspace(high, low, steps + 1)
        for t, s in zip(grid[:-1], grid[1:]):
            h = s - t
            k1 = velocity(z, t, gamma)
            k2 = velocity(z + h*k1/2, t+h/2, gamma)
            k3 = velocity(z + h*k2/2, t+h/2, gamma)
            k4 = velocity(z + h*k3, s, gamma)
            z += h*(k1+2*k2+2*k3+k4)/6
    return z


def main():
    start = time.perf_counter()
    out = Path(__file__).resolve().parents[1] / 'experiments/results/terminal_defect_20260908'
    saved = np.load(out / 'ag_source_calibration_toy.npz')
    initial = saved['initial']
    fit = np.arange(len(initial)) % 8 == 0
    held = ~fit
    base = sample(initial, 0., False)
    parity = float(np.max(np.abs(base - saved['ordinary_ag'])))
    assert parity < 1e-5, parity
    rows = []
    for arm in ['weak_reference', 'mixture_reference', 'weak_posterior_relaxed']:
        target = saved[arm]
        for early in [False, True]:
            def objective(delta):
                return float(np.mean((sample(initial[fit], delta, early) - target[fit])**2))
            result = minimize_scalar(objective, bounds=(0., 3.), method='bounded',
                                     options={'xatol': 1e-5})
            assert result.success
            endpoint = sample(initial, result.x, early)
            fine = sample(initial, result.x, early, 400)
            numerical = float(np.max(np.abs(endpoint - fine)))
            assert numerical < 1e-4, numerical
            before = float(np.mean((base[held] - target[held])**2))
            after = float(np.mean((endpoint[held] - target[held])**2))
            row = dict(arm=arm, control='early_strength' if early else 'global_strength',
                       delta=float(result.x), fit_mse=float(result.fun), heldout_mse=after,
                       heldout_original_displacement_mse=before,
                       heldout_displacement_explained=1-after/before,
                       numerical_200_400_max=numerical,
                       boundary_solution=bool(result.x < .001 or result.x > 2.999))
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(complete=True, posthoc=True, rows=rows,
                  baseline_endpoint_parity_max=parity, seconds=time.perf_counter()-start,
                  protocol='Fit one scalar to calibrated endpoints on every eighth quantile; evaluate remaining quantiles. No true distribution or quality used for fitting. Controls retain AG throughout. This tests endpoint redundancy in one toy family, not novelty or neural quality.')
    (out / 'ag_calibration_strength_control.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
