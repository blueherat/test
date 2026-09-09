"""Polarization audit from paired risks; no surrogate quality claim."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    df = pd.read_csv(a.validation)
    heads = {name: part.sort_values('sample').set_index('sample')
             for name, part in df.groupby('head')}
    rows = []
    for left, right in [('data', 'original'), ('teacher', 'original'), ('teacher', 'data')]:
        u, v = heads[left], heads[right]
        assert u.index.equals(v.index) and len(u) == 1000
        assert np.array_equal(u.noise_time, v.noise_time)
        # MSE_X(U)-MSE_X(V)-MSE_F(U)+MSE_F(V)
        # = 2 <(U-V)/t_floor, (F-X)/t_floor>, per coordinate.
        inner = 0.5 * (u.data_velocity_mse - v.data_velocity_mse
                       - u.teacher_velocity_mse + v.teacher_velocity_mse)
        assert np.isfinite(inner).all()
        rows.append(dict(left=left, right=right, samples=len(inner),
                         error_alignment_mean=float(inner.mean()),
                         descriptive_paired_se=float(inner.std(ddof=1)/np.sqrt(len(inner))),
                         positive_fraction=float((inner > 0).mean())))
    out = pd.DataFrame(rows)
    out.to_csv(a.output, index=False, mode='x')
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
