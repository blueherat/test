#!/usr/bin/env python3
"""Finite-dimensional algebra checks; not a diffusion/image quality experiment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    a = np.array([[2., 0.], [0., 1.]])
    b = np.array([[0., -1.], [3., 0.]])
    u, singular, vt = np.linalg.svd(b.T @ a)
    rotation = u @ vt
    value = float(np.linalg.norm(a - b @ rotation, 'fro')**2)
    lower_bound = float((a * a).sum() + (b * b).sum() - 2 * singular.sum())
    assert np.allclose(rotation @ rotation.T, np.eye(2), atol=1e-14, rtol=0)
    assert abs(value - lower_bound) < 1e-14
    # A legal predictable three-view factor: each row block is an isometry.
    factors = [np.eye(2), rotation, np.array([[1., 0.], [0., -1.]])]
    stacked = np.concatenate(factors, axis=0)
    joint_cov = stacked @ stacked.T
    assert np.linalg.eigvalsh(joint_cov).min() > -1e-14
    for i in range(3):
        assert np.array_equal(joint_cov[2*i:2*i+2, 2*i:2*i+2], np.eye(2))
    # Three separately "perfect" pairings can be incompatible as a joint law.
    impossible = np.array([[1.,1.,-1.],[1.,1.,1.],[-1.,1.,1.]])
    eigenvalues = np.linalg.eigvalsh(impossible)
    assert eigenvalues.min() < 0
    # X1=Z, Y1=rho*Z+sqrt(1-rho^2)*E; X2=X1, Y2=-Y1.
    terminal = [{'rho': rho, 'first_cost_exact': 2-2*rho, 'terminal_cost_exact': 2+2*rho,
                 'each_endpoint_mean': 0, 'each_endpoint_variance': 1}
                for rho in (1,0,-1)]
    assert terminal[0]['first_cost_exact'] == 0 and terminal[0]['terminal_cost_exact'] == 4
    assert terminal[2]['first_cost_exact'] == 4 and terminal[2]['terminal_cost_exact'] == 0
    result = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'complete': True,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Analytic Gaussian moments plus a fixed 2x2 Procrustes/covariance calculation; no Monte Carlo or learned model.',
              'procrustes': {'A': a.tolist(), 'B': b.tolist(), 'Q': rotation.tolist(),
                             'orthogonality_max_abs': float(np.abs(rotation@rotation.T-np.eye(2)).max()),
                             'achieved_cost': value, 'nuclear_norm_lower_bound': lower_bound,
                             'independent_innovation_cost': float((a*a).sum()+(b*b).sum()),
                             'shared_unrotated_cost': float(((a-b)**2).sum())},
              'incompatible_pairwise_correlations': {'matrix': impossible.tolist(), 'eigenvalues': eigenvalues.tolist()},
              'current_noise_adaptive_sign_counterexample': {'transformed_noise': 'sign(Z)*Z=abs(Z)',
                                                            'mean_exact': math.sqrt(2/math.pi), 'second_moment_exact': 1,
                                                            'variance_exact': 1-2/math.pi, 'orthogonal_scalar_square': 1},
              'greedy_local_cost_terminal_failure': terminal,
              'gpu_calls': 0, 'model_calls': 0, 'generated_images': 0,
              'cpu_wall_seconds_before_write': time.perf_counter()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
