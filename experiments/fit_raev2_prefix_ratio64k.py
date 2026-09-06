"""One fixed 2881-parameter conditional ratio solve on 64K actual states."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from experiments.raev2_conditional_ratio_objective import ConditionalRatioRisk
from experiments.raev2_prefix_ratio64k_plan import PLAN
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def augmented_features(p, q, mean=None, scale=None):
    p, q = p.astype(np.float64), q.astype(np.float64)
    if mean is None:
        mean = .5*(p.mean((0, 1))+q.mean(0))
        p -= mean; q -= mean
        scale = float(np.sqrt(.5*(np.mean(p*p)+np.mean(q*q))))
    else:
        p -= mean; q -= mean
    assert np.isfinite(scale) and scale > 0
    p /= scale; q /= scale
    pp = np.concatenate((p, np.ones((*p.shape[:-1], 1))), axis=-1)
    qq = np.concatenate((q, np.ones((len(q), 1))), axis=-1)
    return pp, qq, mean, scale


def main():
    out = DATA/'prefix_ratio64k_fit'
    out.mkdir(exist_ok=False)
    source_paths = [Path(__file__).resolve(), ROOT/'experiments/raev2_conditional_ratio_objective.py',
                    ROOT/'experiments/raev2_prefix_ratio64k_plan.py']
    sources = {str(p): sha(p) for p in source_paths}
    feature_root = DATA/'prefix_ratio64k_features'
    execution_path = feature_root/'execution.json'
    execution = json.loads(execution_path.read_text())
    assert execution['complete'] and execution['all_input_hashes_verified']
    assert execution['plan'] == PLAN
    for split in ['train', 'validation']:
        for name, digest in execution['splits'][split]['files'].items():
            assert sha(feature_root/split/name) == digest
    (out/'plan.json').write_text(json.dumps(PLAN, indent=2)+'\n')
    (out/'frozen_source').mkdir()
    for p in source_paths:
        (out/'frozen_source'/p.name).write_bytes(p.read_bytes())
    began = time.perf_counter()
    folder = feature_root/'train'
    meta = np.load(folder/'metadata.npz')
    assert np.array_equal(meta['ids'], np.arange(64000))
    pp, qq, mean, scale = augmented_features(np.load(folder/'positive.npy', mmap_mode='r'),
                                           np.load(folder/'negative.npy', mmap_mode='r'))
    assert pp.shape == (64000, 5, 2881) and qq.shape == (64000, 2881)
    energy = .5*(np.mean(np.sum(pp**2, axis=-1))+np.mean(np.sum(qq**2, axis=-1)))
    assert abs(energy-2881) < 1e-8
    signal = (np.float32(1)-meta['times']).astype(np.float64)
    risk = ConditionalRatioRisk(pp, qq, signal, PLAN['ridge'])
    iterations = []
    def progress(weight):
        iterations.append(None)
        if len(iterations) % 10 == 0:
            value, gradient = risk(weight)
            print(json.dumps({'iteration': len(iterations), 'objective': value,
                              'gradient_maxabs': float(np.abs(gradient).max()),
                              'cpu_elapsed_seconds': time.perf_counter()-began}), flush=True)
    opt = PLAN['optimizer']
    result = minimize(risk, np.zeros(2881), jac=True, method=opt['method'], callback=progress,
                      options={k: opt[k] for k in ['maxiter', 'gtol', 'ftol', 'maxls']})
    objective, gradient = risk(result.x)
    training_risk = float(risk.losses(result.x).mean())
    converged = bool(np.abs(gradient).max() < 1e-5)
    del risk, pp, qq
    folder = feature_root/'validation'
    meta = np.load(folder/'metadata.npz')
    assert np.array_equal(meta['ids'], np.arange(8000))
    vp, vq, _, _ = augmented_features(np.load(folder/'positive.npy', mmap_mode='r'),
                                    np.load(folder/'negative.npy', mmap_mode='r'), mean, scale)
    vsignal = (np.float32(1)-meta['times']).astype(np.float64)
    losses = ConditionalRatioRisk(vp, vq, vsignal, 0).losses(result.x)
    classes = losses.reshape(8, 1000).mean(0)
    value, se = float(classes.mean()), float(classes.std(ddof=1)/np.sqrt(1000))
    batch_values = losses.reshape(-1, 8).mean(1)
    validation = {'independent_noise_states': 8000, 'positive_choices_per_state': 8,
                  'classes': 1000, 'mean_scaled_logistic': value, 'class_standard_error': se,
                  'upper_two_standard_errors': value+2*se,
                  'native_batch_standard_error_diagnostic': float(batch_values.std(ddof=1)/np.sqrt(len(batch_values))),
                  'entry_condition_passed': bool(converged and value+2*se < 0),
                  'not_a_fid_or_input_gradient_generalization_claim': True}
    np.savez(out/'validation_records.npz', ids=meta['ids'], times=meta['times'], loss=losses, class_mean_loss=classes)
    checkpoint = out/'head.pt'
    torch.save({'weight': torch.from_numpy(result.x[:-1]), 'bias': float(result.x[-1]),
                'feature_mean': torch.from_numpy(mean), 'feature_scale': scale,
                'plan': PLAN, 'validation': validation}, checkpoint)
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    record = {'complete': True, 'plan': PLAN, 'sources': sources,
              'feature_execution_sha256': sha(execution_path),
              'training_conditional_risk': training_risk, 'regularized_objective': objective,
              'gradient_max_abs': float(np.abs(gradient).max()), 'optimizer_converged': converged,
              'optimizer_message': str(result.message), 'iterations': result.nit,
              'coefficient_norm': float(np.linalg.norm(result.x)), 'validation': validation,
              'cpu_elapsed_seconds': time.perf_counter()-began,
              'checkpoint_sha256': sha(checkpoint), 'validation_records_sha256': sha(out/'validation_records.npz'),
              'no_fid_evaluated': True}
    (out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_fit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
