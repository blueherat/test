"""Extend the prior independent FID audit to five newly completed arms."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from experiments.audit_raev2_screen_fid_20260907 import DATA, ROOT, sha, load_moments, resolve


def main():
    previous_path = ROOT / 'experiments/results/raev2_guidance_20260907/fid_audit.json'
    previous = json.loads(previous_path.read_text())
    ref = Path(resolve('imagenet_256_fid_stats'))
    assert sha(ref) == previous['reference_sha256']
    mr, sr = load_moments(str(ref))
    base = next(r for r in previous['rows'] if r['mode'] == 'official')
    base_trace, real_trace = base['sample_covariance_trace'], float(np.trace(sr))
    base_affinity = (base_trace + real_trace - base['covariance_term']) / (2 * math.sqrt(base_trace * real_trace))
    rows = []
    for study in ('critic_screen1k', 'two_mode_screen1k', 'semantic_complement_screen1k'):
        folder = DATA / study
        assert json.loads((folder / 'execution.json').read_text())['complete']
        for metric in json.loads((folder / 'metrics.json').read_text()):
            name = metric['branch']
            path = folder / 'official_feature_cache' / f"{name}-{metric['sample_sha256'][:16]}-inception.features.pt"
            x = torch.load(path, map_location='cpu', weights_only=True).double().numpy()
            mean = x.mean(0)
            centered = x - mean
            eigenvalues = np.linalg.eigvalsh(centered @ sr @ centered.T / (len(x) - 1))
            mean_term = float(np.sum((mean - mr)**2))
            trace = float(np.sum(centered**2) / (len(x) - 1))
            cov = trace + real_trace - 2 * np.sqrt(np.maximum(eigenvalues, 0)).sum()
            fid = mean_term + float(cov)
            assert abs(fid - metric['fid']) < 1e-3
            affinity = (trace + real_trace - cov) / (2 * math.sqrt(trace * real_trace))
            row = {'study': study, 'mode': name, 'fid': fid, 'official_fid': metric['fid'],
                   'feature_sha256': sha(path), 'mean_term': mean_term, 'covariance_term': float(cov),
                   'sample_covariance_trace': trace,
                   'decomposition': {
                       'mean_change': mean_term - base['mean_term'],
                       'trace_effect_at_baseline_shape': trace - base_trace - 2 * math.sqrt(real_trace) * base_affinity * (math.sqrt(trace) - math.sqrt(base_trace)),
                       'shape_effect_at_new_trace': -2 * math.sqrt(trace * real_trace) * (affinity - base_affinity)}}
            rows.append(row)
            print(json.dumps(row), flush=True)
    record = {'complete': True, 'previous_audit_sha256': sha(previous_path),
              'previous_count': len(previous['rows']), 'new_count': len(rows),
              'previous_rows_not_recomputed': True, 'reference_sha256': sha(ref),
              'new_max_abs_error': max(abs(r['fid'] - r['official_fid']) for r in rows),
              'formula': 'independent N-by-N Gram eigenvalues; Bessel sample covariance',
              'decomposition_is_descriptive_not_causal': True, 'new_rows': rows}
    out = ROOT / 'experiments/results/raev2_guidance_20260907/fid_audit_extended.json'
    out.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
