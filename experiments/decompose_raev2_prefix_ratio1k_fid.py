"""Exact mean/trace/shape accounting from the two independently audited FIDs."""
import json
import math
from pathlib import Path
from experiments.summarize_raev2_guidance_20260907 import ROOT, sha


def main():
    folder = ROOT/'experiments/results/raev2_guidance_20260907'
    baseline_path = folder/'fid_audit.json'
    candidate_path = folder/'prefix_ratio64k_screen1k_audit.json'
    baseline_audit = json.loads(baseline_path.read_text())
    candidate_audit = json.loads(candidate_path.read_text())
    assert baseline_audit['complete'] and candidate_audit['complete']
    assert baseline_audit['reference_sha256'] == candidate_audit['reference_sha256']
    baseline = next(row for row in baseline_audit['rows'] if row['mode'] == 'official')
    candidate = candidate_audit['independent_reconstruction']
    t0, t1 = baseline['sample_covariance_trace'], candidate['sample_covariance_trace']
    trace_ref = baseline_audit['reference_covariance_trace']
    b0 = (t0+trace_ref-baseline['covariance_term'])/(2*math.sqrt(t0))
    b1 = (t1+trace_ref-candidate['covariance_term'])/(2*math.sqrt(t1))
    mean = candidate['mean_term']-baseline['mean_term']
    trace = t1-t0-2*b0*(math.sqrt(t1)-math.sqrt(t0))
    shape = -2*math.sqrt(t1)*(b1-b0)
    total = candidate['fid']-baseline['fid']
    assert abs(mean+trace+shape-total) < 1e-10
    record = {'complete': True, 'samples': 1000, 'mean_term_delta': mean,
              'trace_effect_at_baseline_shape': trace, 'normalized_shape_effect_at_candidate_trace': shape,
              'total_fid_delta': total,
              'decomposition_order': 'first mean, then trace at baseline normalized shape, then normalized shape at candidate trace',
              'descriptive_not_causal_and_not_5k_prediction': True,
              'no_parameters_fitted_no_images_generated': True,
              'source_audits': {str(p): sha(p) for p in [baseline_path, candidate_path]},
              'source_sha256': sha(Path(__file__).resolve())}
    (folder/'prefix_ratio64k_1k_fid_decomposition.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
