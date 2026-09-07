"""One fixed convex conditional Gaussian variance fit; no FID access."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from experiments.raev2_conditional_variance import PLAN, ConditionalVarianceRisk
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def normalize(features, mean=None, scale=None):
    x = np.array(features, dtype=np.float64, copy=True)
    if mean is None:
        mean = x.mean(0)
        x -= mean
        scale = float(np.sqrt(np.mean(x*x)))
    else:
        x -= mean
    assert np.isfinite(scale) and scale > 0
    x /= scale
    return np.column_stack((x, np.ones(len(x)))), mean, scale


def main():
    out = DATA/'conditional_variance_fit'
    out.mkdir(exist_ok=False)
    paths = [Path(__file__).resolve(), ROOT/'experiments/raev2_conditional_variance.py']
    sources = {str(p): sha(p) for p in paths}
    root = DATA/'conditional_variance_features'
    execution_path = root/'execution.json'
    execution = json.loads(execution_path.read_text())
    assert execution['complete'] and execution['all_used_input_hashes_verified'] and execution['plan'] == PLAN
    for split in ['train', 'validation']:
        for name, digest in execution['splits'][split]['files'].items():
            assert sha(root/split/name) == digest
    calibration_path = DATA/'guided_reverse_variance/calibration.json'
    calibration_sha = sha(calibration_path)
    assert calibration_sha == execution['variance_calibration_sha256']
    calibration = json.loads(calibration_path.read_text())
    mse0 = np.asarray([row['mse'] for row in calibration['rows']], dtype=np.float64)
    assert len(mse0) == 100 and (mse0 > 0).all()
    (out/'frozen_source').mkdir()
    for p in paths:
        (out/'frozen_source'/p.name).write_bytes(p.read_bytes())
    (out/'plan.json').write_text(json.dumps(PLAN, indent=2)+'\n')
    began = time.perf_counter()
    meta = np.load(root/'train/metadata.npz')
    assert np.array_equal(meta['ids'], np.arange(64000))
    x, mean, scale = normalize(np.load(root/'train/features.npy', mmap_mode='r'))
    assert x.shape == (64000, 2881)
    assert abs(np.mean(np.sum(x*x, axis=1))-2881) < 1e-8
    risk = ConditionalVarianceRisk(x, meta['residual_mse']/mse0[meta['query_indices']], PLAN['ridge'])
    iterations = []
    def progress(weight):
        iterations.append(None)
        if len(iterations) % 10 == 0:
            value, gradient = risk(weight)
            print(json.dumps({'iteration': len(iterations), 'objective': value,
                              'gradient_maxabs': float(np.abs(gradient).max()),
                              'seconds': time.perf_counter()-began}), flush=True)
    opt = PLAN['optimizer']
    result = minimize(risk, np.zeros(2881), jac=True, method=opt['method'], callback=progress,
                      options={k: opt[k] for k in ['maxiter', 'gtol', 'ftol', 'maxls']})
    objective, gradient = risk(result.x)
    training_loss = float(risk.losses(result.x).mean())
    converged = bool(np.abs(gradient).max() < 1e-5)
    meta = np.load(root/'validation/metadata.npz')
    assert np.array_equal(meta['ids'], np.arange(8000)) and np.array_equal(meta['labels'], meta['ids']%1000)
    vx, _, _ = normalize(np.load(root/'validation/features.npy', mmap_mode='r'), mean, scale)
    y = meta['residual_mse']/mse0[meta['query_indices']]
    losses = ConditionalVarianceRisk(vx, y, 0).losses(result.x)
    h = vx@result.x
    assert np.isfinite(h).all() and np.isfinite(np.exp(h)).all()
    classes = losses.reshape(8, 1000).mean(0)
    value, se = float(classes.mean()), float(classes.std(ddof=1)/np.sqrt(1000))
    time_rows = []
    for index in range(100):
        mask = meta['query_indices'] == index
        assert mask.sum() == 80
        time_rows.append({'query_index': index, 'time': calibration['rows'][index]['time'],
                          'samples': int(mask.sum()), 'mean_nll_change': float(losses[mask].mean()),
                          'mean_variance_ratio': float(np.exp(h[mask]).mean()),
                          'mean_observed_residual_ratio': float(y[mask].mean())})
    validation = {'samples': 8000, 'classes': 1000, 'mean_nll_change': value,
                  'class_standard_error': se, 'upper_two_standard_errors': value+2*se,
                  'entry_condition_passed': bool(converged and value+2*se < 0),
                  'variance_ratio_quantiles': dict(zip(['min', 'p01', 'p50', 'p99', 'max'],
                      np.quantile(np.exp(h), [0, .01, .5, .99, 1]).tolist())),
                  'descriptive_uncertainty_not_untouched_historical_holdout': True,
                  'not_a_fid_generalization_claim': True}
    np.savez(out/'validation_records.npz', ids=meta['ids'], times=meta['times'], query_indices=meta['query_indices'],
             residual_ratio=y, log_variance_ratio=h, nll_change=losses, class_mean_loss=classes)
    checkpoint = out/'head.pt'
    torch.save({'weight': torch.from_numpy(result.x[:-1]), 'bias': float(result.x[-1]),
                'feature_mean': torch.from_numpy(mean), 'feature_scale': scale,
                'plan': PLAN, 'validation': validation, 'calibration_sha256': calibration_sha}, checkpoint)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    record = {'complete': True, 'plan': PLAN, 'sources': sources,
              'feature_execution_sha256': sha(execution_path), 'calibration_sha256': calibration_sha,
              'training_nll_change': training_loss, 'regularized_objective': objective,
              'gradient_max_abs': float(np.abs(gradient).max()), 'optimizer_converged': converged,
              'optimizer_message': str(result.message), 'iterations': result.nit,
              'coefficient_norm': float(np.linalg.norm(result.x)), 'validation': validation,
              'validation_all_times': time_rows, 'cpu_elapsed_seconds': time.perf_counter()-began,
              'checkpoint_sha256': sha(checkpoint), 'validation_records_sha256': sha(out/'validation_records.npz'),
              'no_fid_evaluated': True, 'goal_achieved': False}
    (out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/conditional_variance_fit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'validation_all_times'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
