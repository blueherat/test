"""Archive both fixed ratio screens and reconstruct their FID independently."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.audit_raev2_screen_fid_20260907 import load_moments, resolve
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    baseline = json.loads((DATA / 'ancestral_screen1k/metrics.json').read_text())[0]
    control = json.loads((DATA / 'ancestral_screen1k/official/summary.json').read_text())
    cost = control['trajectory_seconds_sum'] + control['decode_seconds_sum']
    reference = Path(resolve('imagenet_256_fid_stats'))
    assert sha(reference) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    mr, cr = load_moments(str(reference))
    rows = []
    for mode in ('paired_ratio', 'paired_ratio_calibrated'):
        folder = DATA / (mode + '_screen1k')
        execution = json.loads((folder / 'execution.json').read_text())
        assert execution['complete']
        for source, digest in execution['sources'].items():
            assert sha(folder / 'frozen_source' / Path(source).name) == digest
        metric = json.loads((folder / 'metrics.json').read_text())[0]
        summary = json.loads((folder / mode / 'summary.json').read_text())
        assert summary['complete'] and summary['count'] == 1000 and summary['seed'] == 202609071
        assert summary['sample_sha256'] == metric['sample_sha256'] == sha(Path(metric['sample_path']))
        assert summary['paired_noise_labels_sha256'] == control['paired_noise_labels_sha256']
        assert summary['sample_model_calls'] == summary['sample_ratio_backward_calls'] == 100000
        features = folder / 'official_feature_cache' / f"{mode}-{metric['sample_sha256'][:16]}-inception.features.pt"
        x = torch.load(features, map_location='cpu', weights_only=True).double().numpy()
        assert x.shape == (1000, 2048) and np.isfinite(x).all()
        mean = x.mean(0)
        centered = x - mean
        spectrum = np.linalg.eigvalsh(centered @ cr @ centered.T / 999)
        mean_term = float(np.sum((mean - mr)**2))
        covariance_term = float(np.sum(centered**2)/999 + np.trace(cr) - 2*np.sqrt(np.maximum(spectrum, 0)).sum())
        independent = mean_term + covariance_term
        assert abs(independent - metric['fid']) < 1e-3
        total = summary['trajectory_seconds_sum'] + summary['decode_seconds_sum']
        row = {**metric, **summary, 'relative_fid_improvement_percent': 100*(1-metric['fid']/baseline['fid']),
               'independent_fid': independent, 'feature_sha256': sha(features),
               'mean_term': mean_term, 'covariance_term': covariance_term,
               'inference_gpu_seconds': total, 'inference_cost_ratio': total/cost,
               'execution_sha256': sha(folder / 'execution.json')}
        rows.append(row)
        print(mode, row['fid'], row['relative_fid_improvement_percent'], flush=True)
    fit = json.loads((DATA / 'paired_ratio_fit/execution.json').read_text())
    record = {'complete': True, 'goal_achieved': False, 'three_percent_quality_passed': False,
              'rows': rows, 'baseline_fid': baseline['fid'], 'reference_sha256': sha(reference),
              'independent_formula': 'FP64 N-by-N Gram eigenvalues, Bessel sample covariance',
              'training_gpu_seconds_once': fit['training_gpu_seconds'],
              'training_cost_excluded_from_inference_ratios_but_disclosed_separately': True,
              'calibration': json.loads((DATA / 'paired_ratio_calibration.json').read_text()),
              'conclusion': 'Neither frozen raw ratio nor held-out probability calibration yields 3%; no further temperature or time-window trials.'}
    assert all(row['relative_fid_improvement_percent'] < 3 for row in rows)
    path = ROOT / 'experiments/results/raev2_guidance_20260907/paired_ratio_screens.json'
    path.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
