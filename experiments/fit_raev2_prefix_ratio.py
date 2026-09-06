"""One convex regularized noisy ratio fit on frozen native prefix features."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def objective_gradient(w, p, q, signal, ridge):
    fp, fq = p@w, q@w
    x, y = signal*fp/2, signal*fq/2
    linear = np.mean((q-p)/(4*signal[:, None]), axis=0)
    lp = np.logaddexp(x, -x)-np.log(2)
    lq = np.logaddexp(y, -y)-np.log(2)
    objective = w@linear + np.mean((lp+lq)/(2*signal**2)) + ridge*(w@w)/2
    gradient = (linear + p.T@(np.tanh(x)/(4*signal))/len(p)
                + q.T@(np.tanh(y)/(4*signal))/len(p) + ridge*w)
    return float(objective), gradient


def unregularized_losses(w, p, q, signal):
    fp, fq = p@w, q@w
    lp = np.logaddexp(signal*fp/2, -signal*fp/2)-np.log(2)
    lq = np.logaddexp(signal*fq/2, -signal*fq/2)-np.log(2)
    return ((q-p)@w)/(4*signal) + (lp+lq)/(2*signal**2)


def main():
    out = DATA / 'prefix_ratio_fit'
    out.mkdir(exist_ok=False)
    feature_root = DATA / 'prefix_ratio_features'
    execution_path = feature_root / 'execution.json'
    execution = json.loads(execution_path.read_text())
    assert execution['complete']
    train_path, valid_path = [feature_root/s/'features.npz' for s in ('train', 'validation')]
    for split, path in [('train', train_path), ('validation', valid_path)]:
        assert sha(path) == execution['splits'][split]['features_sha256']
    data = np.load(train_path)
    raw = data['features'].astype(np.float64)
    assert raw.shape == (5000, 2, 2880) and np.array_equal(data['ids'], np.arange(5000))
    mean = raw.mean((0, 1))
    scale = float(np.sqrt(np.mean((raw-mean)**2)))
    assert scale > 0
    normalized = (raw-mean)/scale
    features = np.concatenate((normalized, np.ones((5000, 2, 1))), axis=-1)
    n, _, dim = features.shape
    # E_data ||phi_aug||^2 = dim. A N(0,I/dim) coefficient prior therefore
    # has unit expected f^2 on this fixed training feature distribution.
    assert abs(np.mean(np.sum(features**2, axis=-1))-dim) < 1e-8
    ridge = dim/n
    signal = (np.float32(1)-data['times']).astype(np.float64)
    plan = {'feature_execution_sha256': sha(execution_path), 'source_sha256': sha(Path(__file__).resolve()),
            'feature_dim': dim-1, 'train_pairs': n, 'parameters_including_bias': dim,
            'normalization': 'training feature mean and one global RMS scale; no per-channel or time fitting',
            'prior': 'generalized Bayes N(0,I/dim); unit expected f^2 before observing labels',
            'ridge': ridge, 'ridge_rule': 'dim/n in averaged paired scaled logistic loss',
            'prior_scale_is_a_fixed_modeling_assumption_not_a_unique_theorem': True,
            'optimizer': 'one L-BFGS-B solve from zero; maxiter500, gtol1e-8, ftol1e-13',
            'validation_entry': 'unregularized held-out loss mean + two class SE < 0',
            'sampling_if_entry_passed': {'strength': 1., 'formula': 'native G + t^2 grad_z f(prefix(z,t,c))',
                'time_window': 'all original 100 queries', 'seed': 202609071, 'samples': 1000},
            'no_fid_fitting_no_layer_or_regularizer_search': True}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio_fit_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    began = time.perf_counter()
    result = minimize(objective_gradient, np.zeros(dim), args=(features[:, 0], features[:, 1], signal, ridge),
                      jac=True, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-13, 'maxls': 30})
    objective, gradient = objective_gradient(result.x, features[:, 0], features[:, 1], signal, ridge)
    assert np.max(np.abs(gradient)) < 1e-5, ('convex solve has not converged', result.message, np.max(np.abs(gradient)))
    train_loss = unregularized_losses(result.x, features[:, 0], features[:, 1], signal)
    val = np.load(valid_path)
    assert np.array_equal(val['ids'], np.arange(1000))
    vfeatures = np.concatenate(((val['features'].astype(np.float64)-mean)/scale, np.ones((1000, 2, 1))), axis=-1)
    vsignal = (np.float32(1)-val['times']).astype(np.float64)
    losses = unregularized_losses(result.x, vfeatures[:, 0], vfeatures[:, 1], vsignal)
    risk, se = float(losses.mean()), float(losses.std(ddof=1)/np.sqrt(1000))
    validation = {'samples': 1000, 'mean_scaled_logistic': risk, 'class_standard_error': se,
                  'upper_two_standard_errors': risk+2*se, 'entry_condition_passed': risk+2*se < 0}
    checkpoint = out/'head.pt'
    torch.save({'weight': torch.from_numpy(result.x[:-1]), 'bias': float(result.x[-1]),
                'feature_mean': torch.from_numpy(mean), 'feature_scale': scale, 'plan': plan,
                'validation': validation}, checkpoint)
    np.savez(out/'validation_records.npz', ids=val['ids'], times=val['times'], loss=losses)
    record = {'complete': True, 'plan': plan, 'validation': validation,
              'training_unregularized_loss': float(train_loss.mean()), 'regularized_objective': objective,
              'gradient_max_abs': float(np.max(np.abs(gradient))), 'iterations': result.nit,
              'optimizer_message': str(result.message), 'coefficient_norm': float(np.linalg.norm(result.x)),
              'cpu_elapsed_seconds': time.perf_counter()-began, 'checkpoint_sha256': sha(checkpoint),
              'validation_records_sha256': sha(out/'validation_records.npz'), 'no_fid_evaluated': True}
    (out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio_fit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
