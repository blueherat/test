"""Fit the one preregistered spatial matrix and its diagonal control by moments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.raev2_spatial_guidance_covariance import PLAN
from experiments.summarize_raev2_guidance_20260907 import DATA, sha


def class_summary(loss, labels):
    sums = np.bincount(labels, weights=loss, minlength=1000)
    counts = np.bincount(labels, minlength=1000)
    assert np.all(counts == 8)
    class_mean = sums/counts
    mean = float(class_mean.mean())
    se = float(class_mean.std(ddof=1)/np.sqrt(1000))
    return {'mean_full_dimension_nll_change': mean, 'per_coordinate_nll_change': mean/262144,
            'class_standard_error': se, 'upper_two_standard_errors': mean+2*se,
            'passes': bool(mean+2*se < 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    root = args.folder.resolve()
    execution = json.loads((root/'preparation.json').read_text())
    assert execution['extraction_complete'] and execution['plan'] == PLAN
    out = root/'fit'
    out.mkdir(exist_ok=False)
    began = time.perf_counter()
    banks = {}
    for split in ['train', 'validation']:
        path = root/'moments'/split/'moments.npz'
        assert sha(path) == execution['splits'][split]['moments_sha256']
        banks[split] = np.load(path)
        assert np.array_equal(banks[split]['ids'], np.arange(PLAN[split]))
    train, validation = banks['train'], banks['validation']
    r_train = train['residual'][train['active']]
    assert r_train.shape[1] == PLAN['fitted_dimension'] and len(r_train) > PLAN['fitted_dimension']
    covariance = r_train.T @ r_train/len(r_train)
    covariance = (covariance+covariance.T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() <= 0:
        failure = {'complete': True, 'plan': PLAN, 'entry_condition_passed': False,
                   'reason': 'unregularized empirical covariance is not finite SPD',
                   'minimum_eigenvalue': float(eigenvalues.min())}
        (out/'calibration.json').write_text(json.dumps(failure, indent=2)+'\n')
        print(json.dumps(failure), flush=True)
        return
    root_full = (eigenvectors*np.sqrt(eigenvalues))@eigenvectors.T
    diagonal = np.diag(covariance).copy()
    root_diagonal = np.diag(np.sqrt(diagonal))
    residual = validation['residual']
    rotated = residual@eigenvectors
    base_energy = np.square(residual).sum(axis=1)
    full_loss = np.where(validation['active'], .5*(np.log(eigenvalues).sum()
                        + (np.square(rotated)/eigenvalues).sum(axis=1)-base_energy), 0.)
    diagonal_loss = np.where(validation['active'], .5*(np.log(diagonal).sum()
                            + (np.square(residual)/diagonal).sum(axis=1)-base_energy), 0.)
    assert np.isfinite(full_loss).all() and np.isfinite(diagonal_loss).all()
    checks = {name: class_summary(loss, validation['labels']) for name, loss in [
        ('full_minus_global', full_loss), ('diagonal_minus_global', diagonal_loss),
        ('full_minus_diagonal', full_loss-diagonal_loss)]}
    gate = checks['full_minus_global']['passes'] and checks['full_minus_diagonal']['passes']
    np.savez(out/'covariance.npz', covariance=covariance, root_full=root_full,
             root_diagonal=root_diagonal, eigenvalues=eigenvalues, eigenvectors=eigenvectors,
             training_residual_mean=r_train.mean(axis=0))
    np.savez(out/'validation_records.npz', ids=validation['ids'], labels=validation['labels'],
             full_minus_global=full_loss, diagonal_minus_global=diagonal_loss,
             full_minus_diagonal=full_loss-diagonal_loss)
    global_path = DATA/'directional_variance_moments/calibration.json'
    global_calibration = json.loads(global_path.read_text())
    kappa = global_calibration['kappa']
    rows = []
    for index in range(100):
        mask = validation['query_indices'] == index
        assert mask.sum() == 80
        rows.append({'query_index': index, 'samples': int(mask.sum()),
                     'full_minus_global': float(full_loss[mask].mean()),
                     'full_minus_diagonal': float((full_loss-diagonal_loss)[mask].mean())})
    report = {'complete': True, 'goal_achieved': False, 'plan': PLAN,
        'entry_condition_passed': bool(gate), 'validation': checks,
        'train_active': len(r_train), 'validation_active': int(validation['active'].sum()),
        'kappa': kappa, 'dimension': len(eigenvalues),
        'spectrum': {'minimum': float(eigenvalues.min()), 'maximum': float(eigenvalues.max()),
            'trace': float(eigenvalues.sum()), 'condition_number': float(eigenvalues.max()/eigenvalues.min()),
            'effective_rank': float(eigenvalues.sum()**2/np.square(eigenvalues).sum()),
            'largest_eigenvalue_trace_fraction': float(eigenvalues.max()/eigenvalues.sum()),
            'mean_residual_energy_fraction': float(np.square(r_train.mean(axis=0)).sum()/eigenvalues.sum()),
            'trace_ratio_to_existing_global': float((262144-256+kappa+eigenvalues.sum())/(262144-1+kappa))},
        'validation_all_times': rows,
        'spherical_head_sha256': PLAN['spherical_head_sha256'],
        'global_calibration_sha256': sha(global_path), 'covariance_sha256': sha(out/'covariance.npz'),
        'validation_records_sha256': sha(out/'validation_records.npz'),
        'preparation_snapshot_sha256': sha(root/'preparation.json'),
        'source_sha256': sha(Path(__file__).resolve()), 'cpu_seconds': time.perf_counter()-began,
        'no_fid_used_in_fit': True, 'no_optimizer_grid_or_parameter_search': True}
    (out/'calibration.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'validation_all_times'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
