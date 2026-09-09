"""Exact Gaussian velocity: distinguish time revision from model-error repair.

No learned model, FID, or claim of identifying the RAE failure mechanism.
The 1D linear flow is integrated both as a log-scale quadrature and as an ODE.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import quad, solve_ivp


def slope(t, variance):
    return ((variance + 1) * t - 1) / ((1-t)**2 + variance*t*t)


def main():
    rows = []
    gamma = .35
    for variance in (.25, 1., 4., 16.):
        native_log = quad(lambda t: slope(t, variance), 0, 1,
                          epsabs=1e-12, epsrel=1e-12)[0]
        assert abs(native_log - .5*np.log(variance)) < 1e-11
        for h in (1/32, 1/64):
            def revision(t):
                tau = t + min(h, max(0., .5-t))
                return slope(t, variance)-slope(tau, variance) if t < .5 else 0.
            delta = gamma * quad(revision, 0, .5, points=[.5-h],
                                 epsabs=1e-12, epsrel=1e-12)[0]
            std = np.exp(native_log + delta)
            # Independent integration of d z / dt rather than log z.
            z = 1.
            for left, right in ((0., .5-h), (.5-h, .5), (.5, 1.)):
                sol = solve_ivp(lambda t, x: (slope(t, variance)+gamma*revision(t))*x,
                                (left, right), [z], method='DOP853',
                                rtol=1e-11, atol=1e-13)
                assert sol.success
                z = sol.y[0, -1]
            assert abs(z-std) < 1e-9
            row = dict(target_variance=variance, h=h, gamma=gamma,
                       log_std_shift=delta, revised_variance=std**2,
                       native_w2_squared=0., revised_w2_squared=(std-np.sqrt(variance))**2,
                       independent_ode_std_error=abs(z-std))
            rows.append(row)
            print(row, flush=True)
    out = Path('experiments/results/terminal_defect_20260908/pfr_exact_gaussian_amplitude.csv')
    with out.open('x') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
