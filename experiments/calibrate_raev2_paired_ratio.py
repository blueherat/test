"""One global noisy-logit temperature by convex held-out logistic likelihood.

Even classes estimate the scalar; odd classes only check calibration gain.
No Inception features, images or FID enter this calculation.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    path = DATA / 'paired_ratio_fit/validation_records.npz'
    records = np.load(path)['records']
    request_path = DATA / 'paired_ratio_screen1k/shard0/request.json'
    request = json.loads(request_path.read_text())
    times = np.array(request['time_grid'], dtype=np.float32)[records[:, 1].astype(int)]
    signal = (np.float32(1) - times).astype(np.float64)
    p, q = records[:, 3], records[:, 4]
    fit = records[:, 0].astype(int) % 2 == 0
    def score(alpha):
        return ((q - p) + p * np.tanh(signal * alpha * p / 2) +
                q * np.tanh(signal * alpha * q / 2)) / (4 * signal)
    def hessian(alpha):
        return (p*p * (1 - np.tanh(signal * alpha * p / 2)**2) +
                q*q * (1 - np.tanh(signal * alpha * q / 2)**2)) / 8
    def loss(alpha):
        lp = np.logaddexp(signal * alpha * p / 2, -signal * alpha * p / 2) - np.log(2)
        lq = np.logaddexp(signal * alpha * q / 2, -signal * alpha * q / 2) - np.log(2)
        return alpha * (q - p) / (4 * signal) + (lp + lq) / (2 * signal**2)
    assert np.max(np.abs(loss(1) - records[:, 2])) < 1e-8
    assert score(0)[fit].mean() < 0
    upper = 1.
    while score(upper)[fit].mean() < 0:
        upper *= 2
        assert upper < 1024, 'no finite calibration optimum found'
    alpha = brentq(lambda x: score(x)[fit].mean(), 0., upper, xtol=1e-12)
    cluster_scores = score(alpha)[fit].reshape(5, 500).mean(0)
    se = float(cluster_scores.std(ddof=1) / np.sqrt(500) / hessian(alpha)[fit].mean())
    heldout_change = (loss(alpha) - loss(1))[~fit].reshape(5, 500).mean(0)
    change_mean = float(heldout_change.mean())
    change_se = float(heldout_change.std(ddof=1) / np.sqrt(500))
    result = {'complete': True, 'alpha': alpha, 'alpha_class_cluster_se': se,
              'heldout_loss_change': change_mean, 'heldout_loss_change_class_cluster_se': change_se,
              'entry_condition_passed': bool(alpha - 2 * se > 1 and change_mean + 2 * change_se < 0),
              'fit_classes': 'even ImageNet class ids', 'check_classes': 'odd ImageNet class ids',
              'source_is_prior_classifier_holdout_not_a_new_data_draw': True,
              'uncalibrated_1k_had_already_been_observed': True,
              'fid_used_for_parameter_fit': False, 'time_dependent_parameters': 0,
              'global_fitted_scalars': 1, 'sampling_time_window_unchanged': True,
              'stationarity': float(score(alpha)[fit].mean()),
              'score_at_one': float(score(1)[fit].mean()),
              'fit_loss_after_calibration': float(loss(alpha)[fit].mean()),
              'checkpoint_sha256': sha(DATA / 'paired_ratio_fit/critic.pt'),
              'validation_records_sha256': sha(path), 'native_time_request_sha256': sha(request_path),
              'source_sha256': sha(Path(__file__).resolve()),
              'limitation': 'proper scoring within a one-dimensional function family does not guarantee an exact density ratio or FID gain'}
    out = DATA / 'paired_ratio_calibration.json'
    assert not out.exists()
    out.write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'experiments/results/raev2_guidance_20260907/paired_ratio_calibration.json').write_text(out.read_text())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
