"""Fit one covariance eigenvalue ratio by its exact sample-mean optimum."""
import json
from pathlib import Path
import time
import numpy as np
from experiments.raev2_directional_variance import PLAN
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    root = DATA/'directional_variance_moments'
    execution_path = root/'execution.json'
    execution = json.loads(execution_path.read_text())
    assert execution['complete'] and execution['plan'] == PLAN
    assert sha(DATA/'conditional_variance_fit/head.pt') == PLAN['spherical_head_sha256']
    for split in ['train', 'validation']:
        assert sha(root/split/'moments.npz') == execution['splits'][split]['moments_sha256']
    train, validation = [np.load(root/split/'moments.npz') for split in ['train', 'validation']]
    began = time.perf_counter()
    active = train['active']
    assert active.any() and np.isfinite(train['beta']).all() and (train['beta'] >= 0).all()
    kappa = float(train['beta'][active].mean())
    assert np.isfinite(kappa) and kappa > 0
    influence = ((train['beta']-kappa)*active).reshape(64, 1000).sum(0)*1000/active.sum()
    parameter_se = float(influence.std(ddof=1)/np.sqrt(1000))
    losses = np.where(validation['active'], .5*(np.log(kappa)+validation['beta']*(1/kappa-1)), 0.)
    assert np.isfinite(losses).all()
    classes = losses.reshape(8, 1000).mean(0)
    mean, se = float(classes.mean()), float(classes.std(ddof=1)/np.sqrt(1000))
    np.savez(root/'validation_records.npz', ids=validation['ids'], loss=losses, class_mean_loss=classes)
    rows = []
    for index in range(100):
        mask = validation['query_indices'] == index
        assert mask.sum() == 80
        rows.append({'query_index': index, 'samples': int(mask.sum()), 'active': int(validation['active'][mask].sum()),
                     'mean_beta': float(validation['beta'][mask].mean()), 'mean_full_dimension_nll_change': float(losses[mask].mean())})
    record = {'complete': True, 'goal_achieved': False, 'plan': PLAN, 'kappa': kappa,
              'kappa_class_sandwich_standard_error': parameter_se,
              'train_active': int(active.sum()), 'validation_active': int(validation['active'].sum()),
              'validation': {'full_dimension_nll_change': mean, 'per_coordinate_nll_change': mean/262144,
                  'class_standard_error': se, 'upper_two_standard_errors': mean+2*se,
                  'entry_condition_passed': bool(mean+2*se < 0)},
              'validation_all_times': rows, 'added_trace_fraction_when_active': (kappa-1)/262144,
              'source_sha256': sha(Path(__file__).resolve()), 'feature_execution_sha256': sha(execution_path),
              'validation_records_sha256': sha(root/'validation_records.npz'),
              'spherical_head_sha256': PLAN['spherical_head_sha256'], 'cpu_seconds': time.perf_counter()-began,
              'no_optimizer_or_fid_evaluation': True, 'no_further_fit_or_time_ratio_if_validation_fails': True}
    (root/'calibration.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/directional_variance_calibration.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'validation_all_times'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
