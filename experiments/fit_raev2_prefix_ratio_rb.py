"""Same convex head, exactly averaging the five real choices per noise state."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
import torch
from experiments.fit_raev2_prefix_ratio import objective_gradient, unregularized_losses
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def rb_objective_gradient(w, positives, negative, signal, ridge):
    pairs = [objective_gradient(w, positives[:, k], negative, signal, ridge) for k in range(positives.shape[1])]
    return float(np.mean([p[0] for p in pairs])), np.mean([p[1] for p in pairs], axis=0)


def rb_losses(w, positives, negative, signal):
    return np.mean([unregularized_losses(w, positives[:, k], negative, signal) for k in range(positives.shape[1])], axis=0)


def main():
    out = DATA/'prefix_ratio_rb_fit'
    out.mkdir(exist_ok=False)
    root = DATA/'prefix_ratio_rb_features'
    execution_path = root/'execution.json'
    execution = json.loads(execution_path.read_text())
    assert execution['complete']
    assert sha(root/'features.npz') == execution['features_sha256']
    train = np.load(root/'features.npz')
    p, q = train['positive'].astype(np.float64), train['negative'].astype(np.float64)
    assert p.shape == (5000, 5, 2880) and q.shape == (5000, 2880)
    mean = .5*(p.mean((0, 1)) + q.mean(0))
    scale = float(np.sqrt(.5*(np.mean((p-mean)**2) + np.mean((q-mean)**2))))
    pp = np.concatenate(((p-mean)/scale, np.ones((5000, 5, 1))), axis=-1)
    qq = np.concatenate(((q-mean)/scale, np.ones((5000, 1))), axis=-1)
    dim, n = pp.shape[-1], len(pp)
    ridge = dim/n  # 5000 independent states, never 25000 paired combinations.
    energy = .5*(np.mean(np.sum(pp**2, axis=-1)) + np.mean(np.sum(qq**2, axis=-1)))
    assert abs(energy-dim) < 1e-8
    signal = (np.float32(1)-train['times']).astype(np.float64)
    plan = {'feature_execution_sha256': sha(execution_path), 'source_sha256': sha(Path(__file__).resolve()),
            'independent_states': n, 'real_alternatives': 5, 'parameters_including_bias': dim,
            'ridge': ridge, 'ridge_rule': 'dim/5000, unchanged from the prior head',
            'normalization': 'one empirical mixture mean and one global RMS scale, averaging the five positive choices',
            'prior': 'unchanged N(0,I/dim), unit expected f^2 on the feature mixture',
            'objective': 'conditional average of the same paired scaled logistic risk',
            'optimizer': 'one L-BFGS-B from zero, maxiter500 gtol1e-8 ftol1e-13',
            'validation_entry': 'same original independent 1000 pairs; mean risk + 2 class SE < 0',
            'sampling_if_entry_passed': {'strength': 1., 'times': 'all original 100', 'seed': 202609071, 'samples': 1000},
            'no_layer_temperature_time_or_regularizer_search': True}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio_rb_fit_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    began = time.perf_counter()
    old = torch.load(DATA/'prefix_ratio_fit/head.pt', map_location='cpu', weights_only=False)
    old_w = old['weight'].numpy()
    transferred = np.r_[scale/old['feature_scale']*old_w,
                         old['bias']+(mean-old['feature_mean'].numpy())@old_w/old['feature_scale']]
    old_conditional_risk = float(rb_losses(transferred, pp, qq, signal).mean())
    result = minimize(rb_objective_gradient, np.zeros(dim), args=(pp, qq, signal, ridge), jac=True,
                      method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-13, 'maxls': 30})
    objective, gradient = rb_objective_gradient(result.x, pp, qq, signal, ridge)
    assert np.max(np.abs(gradient)) < 1e-5, ('convex solve not converged', result.message)
    training_risk = float(rb_losses(result.x, pp, qq, signal).mean())
    validation_path = DATA/'prefix_ratio_features/validation/features.npz'
    valid = np.load(validation_path)
    vf = np.concatenate(((valid['features'].astype(np.float64)-mean)/scale, np.ones((1000, 2, 1))), axis=-1)
    vsignal = (np.float32(1)-valid['times']).astype(np.float64)
    losses = unregularized_losses(result.x, vf[:, 0], vf[:, 1], vsignal)
    risk, se = float(losses.mean()), float(losses.std(ddof=1)/np.sqrt(1000))
    validation = {'samples': 1000, 'mean_scaled_logistic': risk, 'class_standard_error': se,
                  'upper_two_standard_errors': risk+2*se, 'entry_condition_passed': risk+2*se < 0}
    checkpoint = out/'head.pt'
    torch.save({'weight': torch.from_numpy(result.x[:-1]), 'bias': float(result.x[-1]),
                'feature_mean': torch.from_numpy(mean), 'feature_scale': scale,
                'plan': plan, 'validation': validation}, checkpoint)
    np.savez(out/'validation_records.npz', ids=valid['ids'], times=valid['times'], loss=losses)
    record = {'complete': True, 'plan': plan, 'validation': validation,
              'old_single_pair_head_on_full_conditional_training_risk': old_conditional_risk,
              'training_conditional_risk': training_risk, 'regularized_objective': objective,
              'gradient_max_abs': float(np.max(np.abs(gradient))), 'iterations': result.nit,
              'optimizer_message': str(result.message), 'coefficient_norm': float(np.linalg.norm(result.x)),
              'cpu_elapsed_seconds': time.perf_counter()-began, 'checkpoint_sha256': sha(checkpoint),
              'validation_input_sha256': sha(validation_path), 'no_fid_evaluated': True}
    (out/'execution.json').write_text(json.dumps(record, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio_rb_fit.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
