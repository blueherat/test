#!/usr/bin/env python3
"""Independent CPU recheck of the completed Round 2 cache audit.

Does not import the audited script. Source teacher energies are accumulated in
50-digit Decimal arithmetic; saved FP64 moments are recomputed separately.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESTART = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906')


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    wall, cpu = time.perf_counter(), time.process_time()
    original = RESTART / 'pressure_innovation_v1'
    output = RESTART / 'pressure_innovation_review_v1'
    output.mkdir(exist_ok=False)
    request = json.loads((original / 'request.json').read_text())
    summary = json.loads((original / 'summary.json').read_text())
    identities = {'review_script': identity(__file__), 'audit_request': identity(original / 'request.json'),
                  'audit_summary': identity(original / 'summary.json')}
    assert summary['complete'] and summary['round'] == 2
    assert identities['audit_request'] == summary['request']
    for name, expected in request['sources'].items():
        identities[name] = identity(expected['path'])
        assert identities[name] == expected, name
    for name, expected in summary['outputs'].items():
        identities[name] = identity(original / name)
        assert identities[name] == expected, name

    with Path(request['sources']['teacher_csv']['path']).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    by_time = defaultdict(list)
    ids, labels = Counter(), Counter()
    for row in rows:
        by_time[int(row['step_index'])].append(row)
        ids[int(row['sample_id'])] += 1
        labels[int(row['label'])] += 1
    assert len(rows) == 10000 and set(by_time) == set(range(100))
    assert all(len(group) == 100 for group in by_time.values())
    assert set(ids) == set(labels) == set(range(1000))
    assert set(ids.values()) == set(labels.values()) == {10}
    prefixes = ('candidate_tau0.0', 'candidate_tau0.5', 'candidate_tau1.0', 'control_tau0.0')
    energy = []
    with localcontext() as ctx:
        ctx.prec = 50
        for k in range(100):
            total = Decimal(0)
            for row in by_time[k]:
                targets = [Decimal(row[p + '_target_energy']) for p in prefixes]
                assert len(set(targets)) == 1
                beta = Decimal(row['beta'])
                total += beta * beta * targets[0]
            energy.append(float(total / Decimal(100)))
    energy = np.array(energy)
    with np.load(request['sources']['moments_npz']['path'], allow_pickle=False) as cache:
        counts = cache['counts']
        assert np.array_equal(counts, np.full(100, 100, dtype=np.int64))
        target = cache['target_sum'] / 100
        official = cache['official_sum'] / 100
    channel_delta = target[:, 1024:] - official[:, 1024:]
    mean_delta = target[:, :1024] - official[:, :1024]
    d2 = np.add.reduce(channel_delta, axis=1) / 1024
    cross = d2 - energy
    mean_square = np.add.reduce(mean_delta * mean_delta, axis=1) / 1024
    rebuilt = dict(d2_W2_minus_Y2=d2, e2_R2_reconstructed=energy,
                   cross_twice_d2_minus_e2=cross,
                   channel_spatial_mean_residual_squared_mean=mean_square,
                   Y_second_moment=np.add.reduce(official[:, 1024:], axis=1) / 1024,
                   W_second_moment=np.add.reduce(target[:, 1024:], axis=1) / 1024)
    with (original / 'per_time.csv').open(newline='') as stream:
        reported = list(csv.DictReader(stream))
    assert [int(row['step_index']) for row in reported] == list(range(100))
    maximum_errors = {}
    for name, values in rebuilt.items():
        saved = np.array([float(row[name]) for row in reported])
        maximum_errors[name] = float(np.max(np.abs(values - saved)))
        assert np.allclose(values, saved, rtol=2e-14, atol=1e-18), name
        assert np.isclose(values.mean(), summary['equal_time_means'][name], rtol=2e-14, atol=1e-18), name
    with np.load(original / 'channel_moments.npz', allow_pickle=False) as saved:
        assert np.array_equal(saved['step_index'], np.arange(100))
        assert np.array_equal(saved['channel_second_moment_difference'], channel_delta)
        assert np.array_equal(saved['channel_spatial_mean_residual'], mean_delta)
    signs = lambda values: dict(negative=int((values < 0).sum()), zero=int((values == 0).sum()),
                               positive=int((values > 0).sum()))
    for name in ('d2_W2_minus_Y2', 'e2_R2_reconstructed', 'cross_twice_d2_minus_e2'):
        assert signs(rebuilt[name]) == summary['exact_sign_counts_no_tolerance_or_selection'][name]
    assert signs(channel_delta) == summary['all_102400_channel_d2_signs']
    with localcontext() as ctx:
        ctx.prec = 50
        # Var(Z)=1/2, Cov(Z,X-Z)=0; hence independent Gaussian components.
        posterior_variance = Decimal(1) / 2
        repaired_variance = Decimal(2) * posterior_variance
        innovation_variance = repaired_variance + posterior_variance
        gaussian_fid = (innovation_variance.sqrt() - repaired_variance.sqrt()) ** 2
    gaussian = json.loads((original / 'gaussian_counterexample.json').read_text())
    assert np.isclose(float(gaussian_fid), gaussian['identity_decoder_1d_Gaussian_FID_after'], rtol=1e-14)
    assert gaussian['identity_decoder_1d_Gaussian_FID_before'] == 0
    for record in identities.values():
        assert identity(record['path']) == record
    result = dict(complete=True, round=2, created_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Independent CPU review; neither another round nor a new quality experiment.',
                  sources=identities, rows=10000, times=100, channels_per_time=1024,
                  decimal_precision=50, comparison_tolerance=dict(rtol=2e-14, atol=1e-18),
                  maximum_per_time_absolute_errors=maximum_errors,
                  equal_time_means={name: float(values.mean()) for name, values in rebuilt.items()},
                  channel_second_moment_arrays_bitwise_equal=True,
                  channel_spatial_mean_arrays_bitwise_equal=True,
                  gaussian_fid_decimal=str(gaussian_fid),
                  checks=dict(all_source_identities=True, full_time_coverage=True,
                              all_four_teacher_labels_equal=True, source_moment_reconstruction=True,
                              per_time_values=True, aggregate_values=True, all_sign_counts=True,
                              fixed_gaussian_counterexample=True),
                  limitations=['Source FP32 residual tensors are absent; this independently checks stored '
                               'scalars and moments, not the original CUDA residual arithmetic.',
                               'Source moments are aggregate statistics; original W/Y states are absent.',
                               'No population test, conditional mean/covariance estimate or FID claim.'],
                  calls=dict(GPU=0, main_model=0, auxiliary_model=0, decoder=0, training=0, FID=0),
                  wall_seconds_before_write=time.perf_counter()-wall,
                  cpu_seconds_before_write=time.process_time()-cpu)
    dest = output / 'independent_review.json'
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(result=identity(dest), checks=result['checks'],
                          maximum_errors=maximum_errors, gaussian_fid_decimal=str(gaussian_fid)), indent=2))


if __name__ == '__main__':
    main()
