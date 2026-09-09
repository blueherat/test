"""CPU check of endpoint-preserving clock changes versus raw time secants."""
import csv
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from audit_pfr_exact_gaussian_amplitude import slope


def g(t):
    return np.sin(2*np.pi*t)**2 if t < .5 else 0.


def gp(t):
    return 2*np.pi*np.sin(4*np.pi*t) if t < .5 else 0.


def main():
    gamma = .35
    rows = []
    for variance in (.25, 1., 4., 16.):
        for h in (1/32, 1/64, 1/128):
            # phi=t-gamma*h*g has positive derivative and fixes both endpoints.
            assert gamma*h*2*np.pi < 1
            def raw(t):
                return gamma*(slope(t, variance)-slope(t+h*g(t), variance))
            def compensated(t):
                return raw(t)-gamma*h*gp(t)*slope(t, variance)
            def exact(t):
                phi=t-gamma*h*g(t)
                return (1-gamma*h*gp(t))*slope(phi, variance)-slope(t, variance)
            shifts = {name: quad(fn, 0, .5, epsabs=1e-12, epsrel=1e-12)[0]
                      for name, fn in [('raw',raw),('compensated',compensated),('exact_clock',exact)]}
            assert abs(shifts['exact_clock']) < 1e-11
            for name, delta in shifts.items():
                rows.append(dict(target_variance=variance,h=h,arm=name,
                                 log_std_shift=delta,
                                 w2_squared=variance*np.expm1(delta)**2))
    out=Path('experiments/results/terminal_defect_20260908/pfr_clock_compensator.csv')
    with out.open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for variance in (.25,1.,4.,16.):
        r=[x for x in rows if x['target_variance']==variance]
        print(variance, {a:[x['log_std_shift'] for x in r if x['arm']==a]
                         for a in ('raw','compensated','exact_clock')},flush=True)


if __name__=='__main__':main()
