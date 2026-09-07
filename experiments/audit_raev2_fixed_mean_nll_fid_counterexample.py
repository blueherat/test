"""Exact scalar counterexample: fixed-mean Gaussian KL and FID order differently.

No fitting, generated images, GPU, FID selection or new quality candidate.
"""
import hashlib
import json
import math
from pathlib import Path


def main():
    source = Path(__file__).resolve()
    root = source.parents[1]
    # P=N(0,1); all candidate means remain 1. E_P[(X-1)^2]=2.
    target_mean, target_variance, fixed_mean = 0.0, 1.0, 1.0
    mse = target_variance + (target_mean - fixed_mean) ** 2
    rows = []
    for name, variance in [('variance_matches_target', 1.0), ('fixed_mean_nll_optimum', mse)]:
        kl = .5 * (math.log(variance / target_variance) + mse / variance - 1)
        fid = (target_mean - fixed_mean) ** 2 + (math.sqrt(target_variance) - math.sqrt(variance)) ** 2
        rows.append({'name': name, 'mean': fixed_mean, 'variance': variance,
                     'kl_p_to_q': kl, 'fid_squared_wasserstein': fid})
    before, after = rows
    assert math.isclose(before['kl_p_to_q'], .5, abs_tol=1e-14)
    assert math.isclose(after['kl_p_to_q'], .5 * math.log(2), abs_tol=1e-14)
    assert math.isclose(after['fid_squared_wasserstein'], 4 - 2 * math.sqrt(2), abs_tol=1e-14)
    assert after['kl_p_to_q'] < before['kl_p_to_q']
    assert after['fid_squared_wasserstein'] > before['fid_squared_wasserstein']
    output = {'complete': True, 'goal_achieved': False, 'analytic_counterexample_not_raev2_quality': True,
              'target': {'mean': target_mean, 'variance': target_variance}, 'rows': rows,
              'kl_change': after['kl_p_to_q'] - before['kl_p_to_q'],
              'fid_change': after['fid_squared_wasserstein'] - before['fid_squared_wasserstein'],
              'claim': 'Even exact fixed-mean Gaussian KL improvement need not improve FID with identity decoding',
              'does_not_identify_cause_of_actual_raev2_results': True,
              'new_quality_candidates': 0, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    path = root / 'experiments/results/raev2_guidance_20260907/fixed_mean_nll_fid_counterexample.json'
    path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
