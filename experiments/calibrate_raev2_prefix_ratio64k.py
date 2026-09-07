"""One global likelihood calibration diagnostic for the fixed ridge head.

Even held-out classes estimate one scalar; odd classes only assess it. This
does not change the running unit-strength1K/5K or automatically sample a new
variant. The ridge stationarity identity motivates the test; no FID is read.
"""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
import torch
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def projected_logits(folder, fitted):
    p = np.load(folder/'positive.npy', mmap_mode='r')
    q = np.load(folder/'negative.npy', mmap_mode='r')
    weight = fitted['weight'].numpy().astype(np.float64)/fitted['feature_scale']
    offset = fitted['bias']-fitted['feature_mean'].numpy()@weight
    fp = np.empty(p.shape[:2], dtype=np.float64)
    fq = np.empty(len(q), dtype=np.float64)
    for begin in range(0, len(q), 256):
        end = min(len(q), begin+256)
        fp[begin:end] = np.asarray(p[begin:end], dtype=np.float64)@weight+offset
        fq[begin:end] = np.asarray(q[begin:end], dtype=np.float64)@weight+offset
    return fp, fq


def per_state(p, q, signal, alpha):
    ap = signal[:, None]*alpha*p/2
    aq = signal*alpha*q/2
    tp, tq = np.tanh(ap), np.tanh(aq)
    linear = (q-p.mean(1))/(4*signal)
    slope = linear+((p*tp).mean(1)+q*tq)/(4*signal)
    curvature = ((p*p*(1-tp*tp)).mean(1)+q*q*(1-tq*tq))/8
    logcosh_p = np.logaddexp(ap, -ap)-np.log(2)
    logcosh_q = np.logaddexp(aq, -aq)-np.log(2)
    loss = alpha*linear+(logcosh_p.mean(1)+logcosh_q)/(2*signal**2)
    return loss, slope, curvature


def main():
    out = DATA/'prefix_ratio64k_probability_calibration'
    out.mkdir(exist_ok=False)
    head_path = DATA/'prefix_ratio64k_fit/head.pt'
    features = DATA/'prefix_ratio64k_features'
    fit = json.loads((DATA/'prefix_ratio64k_fit/execution.json').read_text())
    feature_execution = json.loads((features/'execution.json').read_text())
    assert fit['complete'] and fit['validation']['entry_condition_passed'] and feature_execution['complete']
    assert sha(head_path) == fit['checkpoint_sha256']
    sources = {str(path): sha(path) for path in [Path(__file__).resolve(), head_path,
        DATA/'prefix_ratio64k_fit/execution.json', features/'execution.json']}
    plan = {'created_unix': time.time(), 'round': 2, 'global_fitted_scalars': 1,
            'fit_classes': 'even ImageNet class ids in original8K heldout',
            'check_classes': 'odd ImageNet class ids in original8K heldout',
            'prior_holdout_used_for_original_entry_not_a_new_data_draw': True,
            'fit_on_scaled_logistic_likelihood_not_fid': True,
            'exact_training_identity': 'd/dalpha R_train(alpha*w)|1 = -ridge*||augmented_w||^2',
            'entry_rule': 'one-sided shrinkage correction: alpha-2*SE>1 and odd-class loss-change+2*SE<0',
            'root_solver': 'brentq on alpha>=0; expanding bracket is numerical optimization, no sampling',
            'running_unit_strength1k_and5k_remain_unchanged': True,
            'no_automatic_sampling': True, 'sources': sources}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    started = time.perf_counter()
    fitted = torch.load(head_path, map_location='cpu', weights_only=False)
    results = {}
    for split in ['train', 'validation']:
        folder = features/split
        for name, digest in feature_execution['splits'][split]['files'].items():
            assert sha(folder/name) == digest
        p, q = projected_logits(folder, fitted)
        metadata = np.load(folder/'metadata.npz')
        signal = (np.float32(1)-metadata['times']).astype(np.float64)
        loss, slope, curvature = per_state(p, q, signal, 1.)
        expected = fit['training_conditional_risk'] if split == 'train' else fit['validation']['mean_scaled_logistic']
        assert abs(loss.mean()-expected) < 1e-8
        results[split] = {'loss_at_one': float(loss.mean()), 'score_at_one': float(slope.mean()),
                          'curvature_at_one': float(curvature.mean())}
        if split == 'train':
            predicted = -fit['plan']['ridge']*fit['coefficient_norm']**2
            # Optimization stopped at a small, nonzero vector gradient.
            bound = fit['gradient_max_abs']*np.sqrt(2881)*fit['coefficient_norm']
            assert abs(slope.mean()-predicted) <= bound+1e-10
            results[split].update(ridge_predicted_score=predicted, stationarity_error_bound=bound)
            continue
        selector = metadata['ids']%2 == 0
        assert selector.sum() == 4000
        _, at_zero, _ = per_state(p, q, signal, 0.)
        if at_zero[selector].mean() >= 0:
            alpha = 0.
        else:
            upper = 1.
            while per_state(p, q, signal, upper)[1][selector].mean() < 0:
                upper *= 2
                assert upper < 1024, 'no finite optimum; no parameter fallback'
            alpha = brentq(lambda a: per_state(p, q, signal, a)[1][selector].mean(), 0., upper, xtol=1e-12)
        after, score, hessian = per_state(p, q, signal, alpha)
        class_score = score[selector].reshape(8, 500).mean(0)
        se = float(class_score.std(ddof=1)/np.sqrt(500)/hessian[selector].mean())
        check_change = (after-loss)[~selector].reshape(8, 500).mean(0)
        change, change_se = float(check_change.mean()), float(check_change.std(ddof=1)/np.sqrt(500))
        result = {'complete': True, 'plan': plan, 'alpha': float(alpha), 'alpha_class_cluster_se': se,
                  'stationarity': float(score[selector].mean()), 'odd_class_loss_change': change,
                  'odd_class_loss_change_class_se': change_se,
                  'entry_condition_passed': bool(alpha-2*se > 1 and change+2*change_se < 0),
                  'checkpoint_sha256': fit['checkpoint_sha256'], 'diagnostics': results,
                  'cpu_elapsed_seconds': time.perf_counter()-started,
                  'no_fid_evaluated_or_guidance_modified': True}
    for path, digest in sources.items():
        assert sha(Path(path)) == digest
    (out/'execution.json').write_text(json.dumps(result, indent=2)+'\n')
    (ROOT/'experiments/results/raev2_guidance_20260907/prefix_ratio64k_probability_calibration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
