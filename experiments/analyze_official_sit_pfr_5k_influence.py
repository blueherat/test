"""Local paired FID sensitivity, conditional on fixed class allocation.

This is an approximate noise diagnostic, not a calibrated confidence interval.
Requires the separate pixel/source/FID audit to have completed first.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def symmetric(a):
    return (a + a.T) / 2


def reference_root(cov):
    ev, vectors = np.linalg.eigh(symmetric(cov))
    assert ev.min() > 0, "Positive definite reference required; no regularization."
    return (vectors * np.sqrt(ev)) @ vectors.T


def functional(x, ref_mean, ref_cov, root, weights=None, derivative=False):
    n = len(x)
    if weights is None:
        weights = np.full(n, 1 / n)
    assert np.all(weights > 0) and abs(weights.sum() - 1) < 1e-12
    alpha = n / (n - 1)
    mean = weights @ x
    centered = x - mean
    cov = alpha * (centered.T @ (weights[:, None] * centered))
    ev, vectors = np.linalg.eigh(symmetric(root @ cov @ root))
    assert ev.min() > 0, "Singular covariance: local inverse square root unsupported."
    fid = float(np.square(mean - ref_mean).sum() + np.trace(cov) +
                np.trace(ref_cov) - 2 * np.sqrt(ev).sum())
    if not derivative:
        return fid
    invroot = (vectors / np.sqrt(ev)) @ vectors.T
    a = symmetric(np.eye(x.shape[1]) - root @ invroot @ root)
    influence = (2 * (centered @ (mean - ref_mean)) +
                 alpha * np.einsum('ij,ij->i', centered @ a, centered) -
                 np.sum(a * cov))
    assert abs(weights @ influence) < 1e-8
    return fid, influence, float(ev.min()), float(ev.max() / ev.min())


def paired_se(delta, labels):
    """Fixed equal class counts: sum_c (n_c/n)^2 s_c^2/n_c."""
    variance = 0.0
    for label in np.unique(labels):
        values = delta[labels == label]
        assert len(values) > 1
        variance += (len(values) / len(delta)) ** 2 * values.var(ddof=1) / len(values)
    return float(np.sqrt(variance))


def self_check():
    rng = np.random.default_rng(202609430)
    x = rng.normal(size=(40, 6)) @ rng.normal(size=(6, 6)) + .3
    m = rng.normal(size=6)
    matrix = rng.normal(size=(6, 6))
    cov = matrix @ matrix.T + np.eye(6)
    root = reference_root(cov)
    fid, influence, _, _ = functional(x, m, cov, root, derivative=True)
    direction = rng.normal(size=40)
    direction -= direction.mean()
    direction /= np.max(np.abs(direction))
    prediction = float(direction @ influence / len(x))
    checks = []
    for eps in [1e-3, 5e-4]:
        plus = functional(x, m, cov, root, (1 + eps * direction) / len(x))
        minus = functional(x, m, cov, root, (1 - eps * direction) / len(x))
        measured = (plus - minus) / (2 * eps)
        assert abs(measured - prediction) < 1e-6
        checks.append(dict(eps=eps, measured=measured, predicted=prediction))
    labels = np.arange(40) % 8
    assert paired_se(np.zeros(40), labels) == 0
    # Constant within-class influences cannot fluctuate under fixed class counts.
    assert paired_se(labels.astype(float), labels) == 0
    return dict(fid=fid, derivative_checks=checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-check-only', action='store_true')
    args = parser.parse_args()
    check = self_check()
    if args.self_check_only:
        print(json.dumps(check, indent=2))
        return
    started = time.monotonic()
    output = Path('experiments/results/terminal_defect_20260908')
    with (output / 'official_sit_pfr_5k.csv').open() as f:
        verified = {row['arm']: row for row in csv.DictReader(f)}
    root = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_5k_20260908')
    refpath = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
    assert hashlib.sha256(refpath.read_bytes()).hexdigest() == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    ref = np.load(refpath)
    m, cov = ref['mu'].astype(float), ref['sigma'].astype(float)
    rr = reference_root(cov)
    stats, influences = {}, {}
    for arm in ['pfr', 'ordinary115']:
        path, = (root / arm / 'quality/features').glob('*.features.pt')
        x = torch.load(path, map_location='cpu', weights_only=True).numpy().astype(float)
        assert x.shape == (5000, 2048) and np.isfinite(x).all()
        fid, influence, minimum, condition = functional(x, m, cov, rr, derivative=True)
        assert abs(fid - float(verified[arm]['independent_fid'])) < 1e-7
        stats[arm] = dict(fid=fid, minimum_root_argument_eigenvalue=minimum,
                          root_argument_condition_number=condition,
                          feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        influences[arm] = influence
        print(arm, stats[arm], flush=True)
    labels = np.arange(5000) % 1000
    delta = influences['pfr'] - influences['ordinary115']
    se = paired_se(delta, labels)
    result = dict(complete=True, scope='Local delta approximation; no calibrated CI or p-value; fixed reference and class counts.',
                  arms=stats, fid_difference_pfr_minus_control=stats['pfr']['fid'] - stats['ordinary115']['fid'],
                  approximate_paired_noise_se=se, self_check=check,
                  seconds=time.monotonic() - started,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (output / 'official_sit_pfr_5k_influence.json').open('x') as f:
        json.dump(result, f, indent=2)
    np.savez(output / 'official_sit_pfr_5k_influence.npz', labels=labels, **influences)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
