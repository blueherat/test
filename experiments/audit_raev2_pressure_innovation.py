#!/usr/bin/env python3
"""Round 2: fixed CPU-only moment audit; no model, sampling or FID calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')
PROTOCOL = ROOT / 'docs/RAEV2_PRESSURE_INNOVATION_PROTOCOL_20260906_ZH.md'
REVIEW = ROOT / 'docs/RAEV2_REMAINING_MECHANISM_REVIEW_20260906_ZH.md'
PREFIXES = ('candidate_tau0.0', 'candidate_tau0.5', 'candidate_tau1.0', 'control_tau0.0')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def artifact(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data)}


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def signs(values):
    return {'negative': int(np.sum(values < 0)), 'zero': int(np.sum(values == 0)),
            'positive': int(np.sum(values > 0))}


def equal_fp32(values, name):
    if not np.array_equal(values, values.astype(np.float32).astype(np.float64)):
        raise ValueError(f'{name} is not exact FP32-to-decimal-to-FP64 roundtrip')


def audit(output):
    wall, cpu = time.perf_counter(), time.process_time()
    validate = RESTART / 'paired_bridge_v1/validate'
    paths = {
        'protocol': PROTOCOL, 'mechanism_review': REVIEW, 'script': Path(__file__),
        'teacher_csv': validate / 'teacher_regression.csv',
        'moments_npz': validate / 'moments.npz',
        'producer_request': validate / 'request.json',
        'archived_producer': validate / 'sources/run_raev2_paired_bridge_pilot.py',
        'archived_bridge': validate / 'sources/raev2_paired_bridge.py',
    }
    # The request is committed before opening or calculating numerical records.
    output.mkdir(parents=True, exist_ok=False)
    request = {'round': 2, 'protocol': 'raev2_pressure_innovation_fixed_cache_v1',
               'created_utc': stamp(), 'sources': {k: artifact(p) for k, p in paths.items()},
               'scope': 'All 100 fixed times, 100 records/time; equal-time descriptive means. '
                        'No time/class choice, IID CI, training, model calls or FID.',
               'output_dir': str(output.resolve()), 'python': sys.version,
               'numpy': np.__version__}
    write_json(output / 'request.json', request)
    request_record = artifact(output / 'request.json')

    original_request = json.loads(paths['producer_request'].read_text())
    for key, original in (
        ('archived_producer', 'experiments/run_raev2_paired_bridge_pilot.py'),
        ('archived_bridge', 'experiments/raev2_paired_bridge.py'),
    ):
        if request['sources'][key]['sha256'] != original_request['sources'][original]['sha256']:
            raise ValueError('producer request/source identity mismatch')

    with np.load(paths['moments_npz'], allow_pickle=False) as archive:
        expected = {'counts', 'target_sum', 'official_sum', 'candidate_sum', 'control_sum',
                    'official_squared_delta_sum', 'candidate_squared_delta_sum',
                    'control_squared_delta_sum'}
        if set(archive.files) != expected:
            raise ValueError('unexpected moment schema')
        arrays = {key: archive[key] for key in archive.files}
    counts = arrays['counts']
    if counts.shape != (100,) or counts.dtype != np.int64 or not np.all(counts == 100):
        raise ValueError('expected precisely 100 records at each of 100 times')
    for name, values in arrays.items():
        if name == 'counts':
            continue
        shape = (100, 2048) if name in {'target_sum', 'official_sum', 'candidate_sum', 'control_sum'} else (100,)
        if values.shape != shape or values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError(f'invalid moment array {name}')
    for name in ('target', 'official', 'candidate', 'control'):
        if np.any(arrays[name + '_sum'][:, 1024:] < 0):
            raise ValueError('negative raw second moment')

    with paths['teacher_csv'].open(newline='') as stream:
        data = list(csv.DictReader(stream))
    if len(data) != 10000:
        raise ValueError('expected exactly 10000 records')
    steps = np.array([int(r['step_index']) for r in data])
    if set(steps) != set(range(100)) or not np.array_equal(np.bincount(steps), counts):
        raise ValueError('time coverage does not match moments')
    ids = np.array([int(r['sample_id']) for r in data])
    labels = np.array([int(r['label']) for r in data])
    if set(ids) != set(range(1000)) or set(labels) != set(range(1000)):
        raise ValueError('expected complete 1000-image/class coverage')
    if any(n != 10 for n in Counter(ids).values()):
        raise ValueError('each source must occur at its ten assigned times')
    label_map = {}
    for sample_id, label in zip(ids.tolist(), labels.tolist()):
        if sample_id in label_map and label_map[sample_id] != label:
            raise ValueError('inconsistent label for source')
        label_map[sample_id] = label
    if len(set(label_map.values())) != 1000:
        raise ValueError('class/source assignment not one-to-one')
    t = np.array([float(r['t']) for r in data])
    s = np.array([float(r['s']) for r in data])
    beta = np.array([float(r['beta']) for r in data])
    energy = np.array([[float(r[p + '_target_energy']) for p in PREFIXES] for r in data])
    if not all(np.isfinite(a).all() for a in (t, s, beta, energy)) or np.any(energy < 0):
        raise ValueError('nonfinite or negative input')
    if not np.all((s >= 0) & (s < t) & (t <= 1) & (beta > 0)):
        raise ValueError('invalid time pair')
    if not np.array_equal(energy, np.repeat(energy[:, :1], 4, axis=1)):
        raise ValueError('four target energies are not exactly equal')
    for name, values in [('t', t), ('s', s), ('beta', beta), ('target_energy', energy)]:
        equal_fp32(values, name)
    beta_rebuilt = ((t.astype(np.float32) - s.astype(np.float32)) /
                    np.maximum(t.astype(np.float32), np.float32(.05))).astype(np.float64)
    if not np.array_equal(beta, beta_rebuilt):
        raise ValueError('stored beta does not match native FP32 pair formula')
    converted = beta * beta * energy[:, 0]
    # This compares only a hypothetical extra FP32 unit conversion, not the
    # unavailable original R reduction or its historical CUDA accumulation error.
    converted32 = (beta.astype(np.float32) ** 2 * energy[:, 0].astype(np.float32)).astype(np.float64)
    conversion_difference = np.abs(converted - converted32)
    nonzero = converted > 0

    target = arrays['target_sum'] / counts[:, None]
    official = arrays['official_sum'] / counts[:, None]
    channel_delta = target[:, 1024:] - official[:, 1024:]
    channel_mean_residual = target[:, :1024] - official[:, :1024]
    rows = []
    for k in range(100):
        use = steps == k
        if len(set(ids[use])) != 100 or len(set(labels[use])) != 100:
            raise ValueError('duplicate source/class within a time')
        for values in (t, s, beta):
            if len(set(values[use])) != 1:
                raise ValueError('time group has heterogeneous time/beta')
        y2 = float(np.mean(official[k, 1024:]))
        w2 = float(np.mean(target[k, 1024:]))
        d2 = float(np.mean(channel_delta[k]))
        e2 = math.fsum(converted[use].tolist()) / int(counts[k])
        c2 = d2 - e2
        rows.append({'step_index': k, 't': float(t[use][0]), 's': float(s[use][0]),
                     'beta': float(beta[use][0]), 'count': int(counts[k]),
                     'Y_second_moment': y2, 'W_second_moment': w2,
                     'd2_W2_minus_Y2': d2, 'e2_R2_reconstructed': e2,
                     'cross_twice_d2_minus_e2': c2,
                     'cross_YR_reconstructed': .5*c2,
                     'channel_spatial_mean_residual_squared_mean': float(np.mean(channel_mean_residual[k]**2)),
                     'channel_d2_negative_count': int(np.sum(channel_delta[k] < 0)),
                     'channel_d2_zero_count': int(np.sum(channel_delta[k] == 0)),
                     'channel_d2_positive_count': int(np.sum(channel_delta[k] > 0))})
    with (output / 'per_time.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'channel_moments.npz',
                        step_index=np.arange(100, dtype=np.int64),
                        channel_second_moment_difference=channel_delta,
                        channel_spatial_mean_residual=channel_mean_residual)

    # Fixed Gaussian example specified before these data were inspected.
    base_variance, residual_variance = 1., .5
    innovation_variance = base_variance + residual_variance
    fid = (math.sqrt(innovation_variance) - math.sqrt(base_variance))**2
    assert fid > 0 and innovation_variance == 1.5
    analytic = {
        'kind': 'Exact Gaussian algebra, not an RAE or image FID experiment',
        'X_and_epsilon': 'independent N(0,1)', 't': .5, 's': 0., 'beta': 1.,
        'Z': '(X+epsilon)/2', 'Full_equals_exact_posterior_mean': 'F=M=Z',
        'IG': 'G=sqrt(2)*Z', 'native_Y_variance': base_variance,
        'true_W_X_variance': 1., 'X_minus_F_variance': residual_variance,
        'covariance_Z_X_minus_F': 0.,
        'independence_reason': 'Joint Gaussian and zero covariance; multiplying the centered '
                               'Gaussian residual by an independent sign preserves its law and independence.',
        'symmetric_innovation_target_variance': innovation_variance,
        'identity_decoder_1d_Gaussian_FID_before': 0.,
        'identity_decoder_1d_Gaussian_FID_after': fid,
        'population_counterexample': 'Even exact Full and exact conditional variance can duplicate '
                                      'a variance repair already accomplished by IG. '
                                      'This disproves a universal improvement claim, not all RAE utility.'}
    write_json(output / 'gaussian_counterexample.json', analytic)

    cols = {name: np.array([r[name] for r in rows]) for name in
            ('d2_W2_minus_Y2', 'e2_R2_reconstructed', 'cross_twice_d2_minus_e2',
             'channel_spatial_mean_residual_squared_mean', 'Y_second_moment', 'W_second_moment')}
    means = {name: math.fsum(values.tolist()) / 100 for name, values in cols.items()}
    sign_counts = {name: signs(cols[name]) for name in
                   ('d2_W2_minus_Y2', 'e2_R2_reconstructed', 'cross_twice_d2_minus_e2')}
    negative_d2 = sign_counts['d2_W2_minus_Y2']['negative']
    negative_defect = sign_counts['cross_twice_d2_minus_e2']['negative']
    decision = (f'No new training budget for pressure-only or symmetric innovation. '
                f'In this fixed cohort, true second-moment change d2 is negative at {negative_d2}/100 times '
                f'and d2-e2 is negative at {negative_defect}/100 times. '
                'These full-cohort observations do not certify m0=0 or justify treating all residual '
                'energy as missing variance. They are descriptive evidence, not an IID population test. '
                'The fixed Gaussian example independently rules out a universal guarantee for the '
                'symmetric variant; actual IG missing covariance and a bounded finite Stein class remain unestablished.')
    for key, path in paths.items():
        if artifact(path) != request['sources'][key]:
            raise ValueError(f'source changed during audit: {key}')
    produced = {name: artifact(output/name) for name in
                ('per_time.csv', 'channel_moments.npz', 'gaussian_counterexample.json')}
    summary = {
        'complete': True, 'round': 2, 'created_utc': stamp(), 'request': request_record,
        'scope': 'CPU-only mechanism decision on preexisting teacher data; not FID, '
                 'not a new sampler and not completion of the 5% quality goal.',
        'rows': 10000, 'time_points': 100, 'records_per_time': 100,
        'unique_sources': 1000, 'unique_labels': 1000,
        'checks': {'moment_schema_shapes_dtypes_finite': True, 'counts_csv_match': True,
                   'four_target_energies_exact_equal': True,
                   'native_beta_fp32_formula_exact_match': True,
                   'stored_scalars_exact_fp32_roundtrip': True,
                   'source_and_producer_request_identities': True,
                   'source_sha_unchanged_through_finish': True},
        'equal_time_means': means, 'exact_sign_counts_no_tolerance_or_selection': sign_counts,
        'all_102400_channel_d2_signs': signs(channel_delta),
        'rounding_boundary': {
            'formula': 'per row: FP64(beta)*FP64(beta)*stored_FP32_target_energy; then equal record mean',
            'producer': 'R=FP32(W-Y), target=FP32(R/beta), target_energy=FP32 mean(target^2). '
                        'W/Y moments were accumulated from their FP32 states in FP64.',
            'maximum_abs_difference_if_extra_FP32_unit_conversion_used': float(conversion_difference.max()),
            'maximum_relative_difference_if_extra_FP32_unit_conversion_used':
                float(np.max(conversion_difference[nonzero]/converted[nonzero])) if np.any(nonzero) else 0.,
            'historical_error_bound': 'Original R/normalized-target tensors and exact reduction tree were not '
                                      'saved. No certified numerical bound on the prior division/square/reduction '
                                      'or W-Y subtraction error can be recovered here. The comparison above '
                                      'only measures an additional conversion arithmetic choice; it is not such a bound.',
            'cross_interpretation': 'd2-e2 reconstructs 2E[Y.R]/D up to those stored FP32 arithmetic '
                                    'boundaries; it is not a direct dot product of original tensors.'},
        'limitations': ['Fixed heterogeneous class/time cohort; same source/noise reused across ten times. '
                        'No IID CI, significance test or independence claim.',
                        'No per-channel R^2 stored, therefore no per-channel cross or conditional covariance.',
                        'Near-zero aggregate cross would only pass a necessary condition, not prove m0=0.',
                        'Current validation did not save Full F; symmetric Full-innovation covariance '
                        'cannot be reconstructed from this R_G cache.',
                        'Exact population moment identities do not alone turn finite-cohort values '
                        'into population refutations or generated-image quality guarantees.'],
        'decision': decision, 'outputs': produced,
        'calls': {'GPU': 0, 'main_model': 0, 'auxiliary_model': 0, 'decoder': 0,
                  'training': 0, 'new_samples': 0, 'FID': 0},
        'wall_seconds_before_summary_write': time.perf_counter()-wall,
        'cpu_seconds_before_summary_write': time.process_time()-cpu}
    write_json(output / 'summary.json', summary)
    print(json.dumps({'round': 2, 'complete': True, 'equal_time_means': means,
                      'sign_counts': sign_counts, 'decision': decision,
                      'wall_seconds': summary['wall_seconds_before_summary_write'],
                      'cpu_seconds': summary['cpu_seconds_before_summary_write'],
                      'summary': artifact(output/'summary.json')}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=RESTART/'pressure_innovation_v1')
    args = parser.parse_args()
    audit(args.output_dir)


if __name__ == '__main__':
    main()
