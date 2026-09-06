#!/usr/bin/env python3
"""CPU-only, read-only audit and descriptive summary of the fixed bridge pilot.

The original experimental artifacts are never modified.  The output directory
must be new.  This is not a quality evaluation or a population KL certificate.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/paired_bridge_v1')
PROTOCOL = 'raev2_paired_bridge_fixed_mechanism_pilot_v1'
MODES = ('pilot', 'train', 'validate', 'rollout')
ARMS = ('official', 'candidate', 'control')
TEACHERS = ('candidate_tau0.0', 'candidate_tau0.5', 'candidate_tau1.0', 'control_tau0.0')
TERMS = ('target_energy', 'prediction_energy', 'cross', 'risk_gain')
EXPECTED_CALLS = {
    'pilot': {'baseline': (4, 32), 'candidate': (10, 320), 'control': (10, 320)},
    'train': {'baseline': (8192, 65536), 'candidate': (2048, 65536), 'control': (2048, 65536)},
    'validate': {'baseline': (1300, 10400), 'candidate': (2000, 50000), 'control': (1200, 30000)},
    'rollout': {'baseline': (1200, 9600), 'candidate': (800, 6400), 'control': (800, 6400)},
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as dest:
        json.dump(value, dest, indent=2, ensure_ascii=False, allow_nan=False)
        dest.write('\n')


def read_csv(path):
    with Path(path).open(newline='') as source:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(source)]


def write_csv(path, rows):
    with Path(path).open('x', newline='') as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(value):
    value = np.asarray(value, dtype=np.float64)
    require(value.size > 0 and np.isfinite(value).all(), 'invalid descriptive array')
    quantiles = np.quantile(value, [0, .05, .25, .5, .75, .95, 1])
    return dict(count=int(value.size), mean=float(value.mean()),
                quantiles=dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'),
                                   map(float, quantiles))))


class Audit:
    def __init__(self):
        self.artifacts = {}

    def file(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.artifacts:
            self.artifacts[key] = dict(path=key, sha256=sha(path), size_bytes=path.stat().st_size)
        return self.artifacts[key]

    def binding(self, item):
        actual = self.file(item['path'])
        require(actual['sha256'] == item['sha256'], f"SHA mismatch: {item['path']}")
        if 'size_bytes' in item:
            require(actual['size_bytes'] == item['size_bytes'], f"size mismatch: {item['path']}")
        return actual


def identities(directory, audit):
    summaries, requests = {}, {}
    for mode in MODES:
        summary = read_json(directory / mode / 'summary.json')
        request = read_json(directory / mode / 'request.json')
        require(summary['complete'] is True and summary['mode'] == mode, f'incomplete {mode}')
        require(summary['protocol'] == request['protocol'] == PROTOCOL, 'protocol mismatch')
        require(request['mode'] == mode, 'request mode mismatch')
        require(Path(summary['request']['path']).resolve() == (directory / mode / 'request.json').resolve(),
                'summary request path mismatch')
        audit.binding(summary['request'])
        audit.file(directory / mode / 'summary.json')
        require(set(summary['calls']) == set(EXPECTED_CALLS[mode]), 'unexpected hook names')
        for name, (calls, samples) in EXPECTED_CALLS[mode].items():
            require(summary['calls'][name] == dict(calls=calls, samples=samples),
                    f'hook count mismatch: {mode}/{name}')
        for key in ('bank', 'config', 'baseline_checkpoint'):
            audit.binding(request[key])
        for name, item in request['sources'].items():
            audit.binding(item)
            require(audit.file(ROOT / name)['sha256'] == item['sha256'], f'current source changed: {name}')
        require(request['baseline_checkpoint_step'] == 100080, 'wrong baseline step')
        require(request['architecture']['parameters_each'] == 3652224, 'wrong field size')
        require(request['images_decoded'] == 0 and request['fid'] is False, 'unexpected quality stage')
        summaries[mode], requests[mode] = summary, request
    shared = ('protocol', 'bank', 'config', 'baseline_checkpoint', 'architecture', 'seeds',
              'training', 'finite_solver', 'baseline_batch_policy', 'interpolation_arithmetic', 'precision')
    for mode in MODES[1:]:
        for key in shared:
            require(requests[mode][key] == requests['pilot'][key], f'stage identity differs: {mode}/{key}')
        require(set(requests[mode]['sources']) == set(requests['pilot']['sources']), 'source names differ')
        for key, item in requests[mode]['sources'].items():
            require(item['sha256'] == requests['pilot']['sources'][key]['sha256'], 'stage sources differ')
        prior_mode = 'pilot' if mode == 'train' else 'train'
        prior = requests[mode]['prior_summary']
        require(Path(prior['path']).resolve() == (directory / prior_mode / 'summary.json').resolve(),
                'wrong previous stage')
        audit.binding(prior)
    require(summaries['pilot']['native_head_parity'] and summaries['pilot']['zero_midpoint_parity']
            and summaries['pilot']['pilot_passed'], 'pilot failed')
    require(summaries['pilot']['learned_weights_saved'] is False, 'pilot weights were retained')
    for mode, updates in (('pilot', 8), ('train', 2048)):
        require(summaries[mode]['parameter_backwards'] == dict(candidate=updates, control=updates),
                'parameter backward count mismatch')
    train = summaries['train']
    require(train['updates'] == 2048 and train['extra_control_training_charged'], 'training budget mismatch')
    audit.binding(train['checkpoint'])
    checkpoint = torch.load(train['checkpoint']['path'], map_location='cpu', weights_only=False)
    require(checkpoint['protocol'] == PROTOCOL and checkpoint['updates'] == 2048, 'checkpoint identity mismatch')
    require(Path(checkpoint['request']).resolve() == (directory / 'train/request.json').resolve(),
            'checkpoint training request path mismatch')
    require(checkpoint['request_sha256'] == audit.file(directory / 'train/request.json')['sha256'],
            'checkpoint request hash mismatch')
    checkpoint_report = {}
    for arm in ('candidate', 'control'):
        state = checkpoint[arm]
        require(all(torch.isfinite(v).all().item() for v in state.values()), 'nonfinite checkpoint')
        require(all(v.device.type == 'cpu' and v.dtype == torch.float32 for v in state.values()),
                'wrong checkpoint dtype/device')
        require(state['time_frequencies'].shape == (16,), 'wrong fixed frequency buffer')
        count = sum(v.numel() for k, v in state.items() if k != 'time_frequencies')
        require(count == 3652224, 'checkpoint state size mismatch')
        checkpoint_report[arm] = dict(parameters=count, fixed_frequency_buffer_elements=16,
                                     all_finite=True, dtype='float32')
    require(set(checkpoint['candidate']) == set(checkpoint['control']), 'different auxiliary state keys')
    checkpoint_report['candidate_control_squared_parameter_distance'] = float(sum(
        (checkpoint['candidate'][k].double() - checkpoint['control'][k].double()).square().sum()
        for k in checkpoint['candidate']))
    del checkpoint
    supplement = read_json(directory / 'supplemental_source_environment.json')
    audit.file(directory / 'supplemental_source_environment.json')
    for item in supplement['sources']:
        require(audit.file(ROOT / item['path'])['sha256'] == item['sha256'], 'supplemental source changed')
    bank = read_json(requests['train']['bank']['path'])
    require(bank['complete'], 'bank incomplete')
    metadata = {}
    for name, count, per_class in (('train', 5000, 5), ('validation', 1000, 1)):
        record = bank['banks'][name]
        for kind in ('metadata', 'latents'):
            audit.binding(record[kind])
        with np.load(record['metadata']['path'], allow_pickle=False) as data:
            metadata[name] = {k: data[k].copy() for k in data.files}
        require(np.array_equal(metadata[name]['ids'], np.arange(count)), 'bank IDs invalid')
        require(np.array_equal(np.unique(metadata[name]['labels']), np.arange(1000)), 'bank labels invalid')
        require(np.all(np.bincount(metadata[name]['labels']) == per_class), 'bank class balance invalid')
        require(len(np.unique(metadata[name]['rows'])) == count, 'bank rows not unique')
        latents = np.load(record['latents']['path'], mmap_mode='r', allow_pickle=False)
        require(latents.shape == (count, 1024, 16, 16) and latents.dtype == np.float16, 'bank shape/dtype invalid')
    require(np.intersect1d(metadata['train']['rows'], metadata['validation']['rows']).size == 0,
            'source rows overlap')
    return summaries, requests, metadata, checkpoint_report


def training(directory, summary, audit):
    rows = read_json(directory / 'train/training_rows.json')
    csv_rows = read_csv(directory / 'train/training.csv')
    require(len(rows) == len(csv_rows) == 2048, 'missing training rows')
    for k, (row, csv_row) in enumerate(zip(rows, csv_rows), 1):
        require(row['update'] == csv_row['update'] == k, 'training update order invalid')
        require(all(np.isfinite(v) for v in row.values()), 'nonfinite training row')
        for key, value in csv_row.items():
            require(value == row[key], 'training CSV differs from complete row log')
        for arm in ('candidate', 'control'):
            require(row[arm + '_gradient_norm'] > 0, 'zero/negative training gradient')
    require(rows[-1] == summary['final_row'], 'training final summary differs')
    with np.load(directory / 'train/training_counts.npz', allow_pickle=False) as data:
        steps, images = data['step_counts'], data['image_counts']
        require(steps.shape == (100,) and images.shape == (5000,), 'training histogram shape')
        require(steps.sum() == images.sum() == 65536, 'training histogram total')
        require(np.array_equal(steps, summary['actual_time_histogram']), 'time histogram summary differs')
        require(np.count_nonzero(images) == summary['unique_training_images'] == 5000, 'image coverage differs')
    for name in ('training_rows.json', 'training.csv', 'training_counts.npz'):
        audit.file(directory / 'train' / name)
    return dict(updates=2048, fresh_teacher_pairs=65536, unique_source_images=5000,
                step_counts=describe(steps), image_counts=describe(images),
                final_row=rows[-1], interpretation='training log only; not held-out or quality evidence')


def validation(directory, summary, labels, out, audit):
    rows = read_csv(directory / 'validate/teacher_regression.csv')
    require(len(rows) == summary['records'] == 10000, 'validation count')
    columns = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    require(all(np.isfinite(value).all() for value in columns.values()), 'nonfinite validation rows')
    ids, steps, classes = (columns[key].astype(np.int64) for key in ('sample_id', 'step_index', 'label'))
    for name, integers in (('sample_id', ids), ('step_index', steps), ('label', classes)):
        require(np.array_equal(columns[name], integers), 'noninteger validation identity')
    require(np.all((ids >= 0) & (ids < 1000)) and np.all((steps >= 0) & (steps < 100)), 'validation ID bounds')
    require(np.array_equal(classes, labels[ids]), 'validation bank labels differ')
    require(np.all(np.bincount(ids, minlength=1000) == 10), 'source cluster sizes differ')
    require(np.all(np.bincount(steps, minlength=100) == 100), 'time counts differ')
    require(len(np.unique(ids * 100 + steps)) == 10000, 'repeated validation source/time')
    assignment = np.random.default_rng(202609145).permutation(1000)
    require(np.array_equal(assignment, summary['assignment']), 'validation assignment seed mismatch')
    expected_ids = np.concatenate([assignment[(k % 10)*100:(k % 10+1)*100] for k in range(100)])
    require(np.array_equal(ids, expected_ids) and np.array_equal(steps, np.repeat(np.arange(100), 100)),
            'fixed group assignment/order changed')
    base = torch.linspace(1., 0., 101, dtype=torch.float32)
    grid = (8 * base / (1 + 7 * base)).numpy()
    require(np.array_equal(columns['t'], grid[steps].astype(np.float64)), 'query grid differs')
    require(np.array_equal(columns['s'], grid[steps + 1].astype(np.float64)), 'successor grid differs')
    beta = (grid[steps] - grid[steps + 1]) / np.maximum(grid[steps], np.float32(.05))
    require(np.array_equal(columns['beta'], beta.astype(np.float64)), 'native beta differs')
    target = columns[TEACHERS[0] + '_target_energy']
    aggregates, class_rows, time_rows = {}, [], []
    for name in TEACHERS:
        require(np.array_equal(target, columns[name + '_target_energy']), 'teacher target energy differs between queries')
        p, r, c = (columns[name + '_' + key] for key in ('prediction_energy', 'target_energy', 'cross'))
        require(np.all(p >= 0) and np.all(r >= 0), 'negative energy')
        require(np.all(c*c <= p*r + 1e-6 * np.maximum(p*r, 1e-20)), 'Cauchy energy inconsistency')
        columns[name + '_risk_gain'] = 2*c-p
        mean_terms = {term: float(columns[name + '_' + term].mean()) for term in TERMS}
        class_means = {term: np.bincount(classes, weights=columns[name + '_' + term], minlength=1000) / 10
                       for term in TERMS}
        mean_terms['risk_gain_over_target_energy'] = mean_terms['risk_gain'] / mean_terms['target_energy']
        aggregates[name] = dict(mean=mean_terms,
            class_cluster_descriptions={term: describe(values) for term, values in class_means.items()},
            positive_risk_gain_classes=int(np.count_nonzero(class_means['risk_gain'] > 0)))
        for label in range(1000):
            class_rows.append(dict(teacher=name, label=label, records=10,
                                   **{term: float(values[label]) for term, values in class_means.items()}))
        for k in range(100):
            mask = steps == k
            time_rows.append(dict(teacher=name, step_index=k, t=float(grid[k]), s=float(grid[k+1]), records=100,
                **{term: float(columns[name + '_' + term][mask].mean()) for term in TERMS}))
    require(np.all(columns['official_shift_energy'] == 0), 'official finite map shifted itself')
    write_csv(out / 'teacher_per_class.csv', class_rows)
    write_csv(out / 'teacher_per_time.csv', time_rows)
    for name in ('teacher_regression.csv', 'finite_map_moments.csv', 'moments.npz'):
        audit.file(directory / 'validate' / name)
    return dict(records=10000, classes=1000, records_per_class=10, records_per_time=100,
        assignment_verified=True, native_grid_verified=True, teacher_queries=aggregates,
        finite_map_shift_energy={arm: describe(columns[arm + '_shift_energy']) for arm in ARMS},
        uncertainty_scope='Descriptive per-class clusters only. Each image and noise is reused at 10 assigned times; no independent-row CI.',
        guarantee_scope='Teacher interpolation regression only; no actual-midpoint velocity target, population KL, or FID claim.')


def finite_moments(directory, out):
    original = read_csv(directory / 'validate/finite_map_moments.csv')
    require(len(original) == 100, 'finite-map moment CSV length')
    with np.load(directory / 'validate/moments.npz', allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    counts = arrays['counts']
    require(counts.shape == (100,) and np.all(counts == 100), 'moment counts')
    for arm in ('target',) + ARMS:
        require(arrays[arm + '_sum'].shape == (100, 2048), 'moment sum shape')
    require(all(np.isfinite(value).all() for value in arrays.values()), 'nonfinite moment arrays')
    rows, aggregate = [], {}
    base = torch.linspace(1., 0., 101, dtype=torch.float32)
    grid = (8 * base / (1 + 7 * base)).numpy()
    for k in range(100):
        require(original[k]['step_index'] == k and original[k]['samples'] == counts[k], 'moment CSV identity')
        require(original[k]['t'] == float(grid[k]) and original[k]['s'] == float(grid[k+1]),
                'moment CSV time differs')
        row = dict(step_index=k, t=original[k]['t'], s=original[k]['s'], samples=int(counts[k]))
        for arm in ARMS:
            delta = arrays[arm + '_sum'][k] - arrays['target_sum'][k]
            square = arrays[arm + '_squared_delta_sum']
            require(square.shape == (100,) and square[k] >= 0, 'squared delta shape/value')
            gap = float(np.square(delta/counts[k]).mean())
            cross = float((np.square(delta).sum() - square[k]) / (counts[k]*(counts[k]-1)*2048))
            for suffix, value in (('empirical_squared_mean_gap', gap), ('offdiagonal_cross_statistic', cross)):
                require(np.isclose(value, original[k][arm + '_' + suffix], rtol=2e-13, atol=2e-16),
                        'recomputed moment statistic differs from CSV')
                row[arm + '_' + suffix] = value
            row[arm + '_first1024_empirical_squared_mean_gap'] = float(np.square(delta[:1024]/counts[k]).mean())
            row[arm + '_second1024_empirical_squared_mean_gap'] = float(np.square(delta[1024:]/counts[k]).mean())
        rows.append(row)
    for arm in ARMS:
        aggregate[arm] = {suffix: describe([r[arm + '_' + suffix] for r in rows]) for suffix in (
            'empirical_squared_mean_gap', 'first1024_empirical_squared_mean_gap',
            'second1024_empirical_squared_mean_gap', 'offdiagonal_cross_statistic')}
    comparisons = {}
    for arm in ('candidate', 'control'):
        comparisons[arm + '_minus_official'] = {}
        for suffix in ('empirical_squared_mean_gap', 'first1024_empirical_squared_mean_gap',
                       'second1024_empirical_squared_mean_gap', 'offdiagonal_cross_statistic'):
            delta = np.array([r[arm+'_'+suffix] - r['official_'+suffix] for r in rows])
            comparisons[arm + '_minus_official'][suffix] = dict(
                difference=describe(delta), times_with_smaller_value=int(np.count_nonzero(delta < 0)),
                negative_difference_sum=float(delta[delta < 0].sum()),
                positive_difference_sum=float(delta[delta > 0].sum()),
                largest_absolute_time_contributions=[dict(step_index=int(k), t=float(grid[k]),
                    difference=float(delta[k]), fraction_of_total_absolute_difference=(
                        float(abs(delta[k])/np.abs(delta).sum()) if np.abs(delta).sum() else 0.))
                    for k in np.argsort(-np.abs(delta), kind='stable')[:10]],
                contribution_scope='All 100 fixed times retained; ranked contributions are descriptions, not a selected time window.')
    write_csv(out / 'finite_map_moments_recomputed.csv', rows)
    return dict(per_arm=aggregate, comparisons=comparisons, original_csv_verified=True,
        scope='Fixed heterogeneous class/image cohort; off-diagonal cross statistic is not claimed unbiased for a population squared mean gap and can be negative.',
        split_limitation='The archive stores only all-2048-dimensional squared_delta_sum. First/second 1024 empirical mean gaps can be split, but their separate off-diagonal cross statistics cannot be reconstructed.',
        not_distribution_distance_or_quality_score=True)


def rollout(directory, summary, out, audit):
    require(summary['actual_rollouts'] == 96 and summary['images_decoded'] == 0 and not summary['fid'],
            'rollout scope differs')
    require(summary['labels'] == list(range(32)), 'rollout labels differ')
    with np.load(directory / 'rollout/rollout_moments.npz', allow_pickle=False) as data:
        features = {arm: data[arm].copy() for arm in ARMS}
        require(set(data.files) == set(ARMS), 'rollout feature arm names')
    reports, rows = {}, []
    for arm in ARMS:
        value = features[arm]
        require(value.shape == (101, 32, 2048) and value.dtype == np.float64 and np.isfinite(value).all(),
                'invalid rollout features')
        require(np.array_equal(value[0], features['official'][0]), 'rollout initial states differ in saved moments')
        path = directory / 'rollout' / (arm + '_endpoints.npy')
        endpoint = np.load(path, allow_pickle=False)
        require(endpoint.shape == (32, 1024, 16, 16) and endpoint.dtype == np.float32,
                'endpoint shape/dtype')
        require(np.isfinite(endpoint).all(), 'nonfinite endpoint')
        require(array_sha(endpoint) == summary['endpoint_sha256'][arm], 'endpoint raw tensor SHA differs')
        double = endpoint.astype(np.float64)
        calculated = np.concatenate((double.mean((2,3)), np.square(double).mean((2,3))), axis=1)
        difference = float(np.abs(calculated - value[-1]).max())
        require(np.allclose(calculated, value[-1], rtol=1e-12, atol=1e-12), 'endpoint moments differ from trajectory archive')
        reports[arm] = dict(file=audit.file(path), tensor_sha256=array_sha(endpoint),
            shape=list(endpoint.shape), dtype='float32', finite=True,
            endpoint_moments_max_absolute_recompute_difference=difference,
            endpoint_per_sample_mean=describe(double.mean((1,2,3))),
            endpoint_per_sample_rms=describe(np.sqrt(np.square(double).mean((1,2,3)))),
            endpoint_absolute_max=float(np.abs(double).max()))
        for k in range(101):
            means, seconds = value[k,:,:1024], value[k,:,1024:]
            delta = value[k] - features['official'][k]
            rows.append(dict(arm=arm, state_index=k,
                mean_spatial_channel_mean=float(means.mean()),
                rms_spatial_channel_mean=float(np.sqrt(np.square(means).mean())),
                mean_spatial_channel_second_moment=float(seconds.mean()),
                first1024_squared_feature_difference_from_official=float(np.square(delta[:,:1024]).mean()),
                second1024_squared_feature_difference_from_official=float(np.square(delta[:,1024:]).mean())))
    with (directory / 'rollout/diagnostic_costs.csv').open(newline='') as source:
        costs = list(csv.DictReader(source))
    require(len(costs) == 12, 'rollout cost rows')
    timings = {}
    for arm in ARMS:
        subset = [row for row in costs if row['mode'] == arm]
        require([int(row['first_id']) for row in subset] == [0,8,16,24], 'rollout batch identities')
        require(all(int(row['count']) == 8 for row in subset), 'rollout batch sizes')
        values = [float(row['seconds_including_per_step_diagnostics']) for row in subset]
        require(np.isfinite(values).all() and min(values) > 0, 'invalid diagnostic times')
        timings[arm] = dict(total_seconds=sum(values), batch_seconds=values)
    for name in ('rollout_moments.npz', 'diagnostic_costs.csv'):
        audit.file(directory / 'rollout' / name)
    write_csv(out / 'rollout_moment_descriptions.csv', rows)
    return dict(endpoints=reports, per_arm_diagnostic_timing=timings,
        scope='96 actual rollouts, 32 fixed classes. All moments describe visited states; no true X, denoising risk, decoded image, or FID. Timing includes FP64 moments and copies.',
        initial_moment_pairing_verified=True, gaussian_noise_sha256=summary['noise_sha256'])


def costs(directory, summaries, requests, audit):
    original = read_json(directory / 'pipeline_status.json')
    resumed = read_json(directory / 'pipeline_resume_status.json')
    require(resumed['complete'] and [s['mode'] for s in resumed['stages']] == ['validate', 'rollout'],
            'resumed pipeline incomplete')
    require(resumed['recovery']['train_restarted'] is False, 'training was restarted')
    require(resumed['recovery']['training_summary_sha256'] == audit.file(directory/'train/summary.json')['sha256'],
            'recovery training summary differs')
    require(resumed['recovery']['training_final_checkpoint_sha256'] == summaries['train']['checkpoint']['sha256'],
            'recovery checkpoint differs')
    audit.file(directory / 'pipeline_status.json')
    audit.file(directory / 'pipeline_resume_status.json')
    require(audit.file(directory / 'run_fixed_stages.py')['sha256'] == original['driver_sha256'], 'old driver changed')
    require(audit.file(directory / 'resume_remaining_stages.py')['sha256'] == resumed['driver_sha256'], 'resume driver changed')
    stage_costs = {}
    for mode in MODES:
        value = summaries[mode]
        stage_costs[mode] = {key: value[key] for key in ('stage_seconds', 'wall_seconds_from_first_time_import',
            'cpu_seconds_from_first_time_import', 'peak_gpu_memory_allocated_bytes', 'calls')}
        if mode in ('validate', 'rollout'):
            entry = next(s for s in resumed['stages'] if s['mode'] == mode)
            require(entry['complete'] and entry['returncode'] == 0, 'resumed stage failed')
            require(entry['summary_sha256'] == audit.file(directory/mode/'summary.json')['sha256'], 'resume summary differs')
            stage_costs[mode]['outer_child_wall_seconds'] = entry['outer_child_wall_seconds']
            stage_costs[mode]['outer_returncode'] = 0
        else:
            stage_costs[mode]['outer_child_wall_seconds'] = None
            stage_costs[mode]['outer_returncode'] = None
    train_entry = next(s for s in original['stages'] if s['mode'] == 'train')
    start = datetime.fromisoformat(train_entry['started_utc'])
    finish = datetime.fromisoformat(summaries['train']['finished_utc'])
    observed_absent = datetime.fromisoformat(resumed['recovery']['missing_processes_confirmed_utc'])
    stage_costs['train']['outer_elapsed_wall_clock_bounds_seconds'] = [
        (finish-start).total_seconds(), (observed_absent-start).total_seconds()]
    stage_costs['train']['outer_boundary_note'] = ('Original driver/session disappeared after the final training artifacts were written. '
        'Return code and exact perf_counter outer wall were not captured. Bounds use UTC timestamps and the later absence observation, '
        'not a recovered monotonic elapsed measurement.')
    totals = {name: {unit: sum(summaries[mode]['calls'][name][unit] for mode in MODES)
                     for unit in ('calls', 'samples')} for name in ('baseline','candidate','control')}
    return dict(stages=stage_costs, total_forward_hooks=totals,
        total_parameter_backwards=dict(candidate=2056,control=2056),
        sum_runner_wall_seconds=sum(summaries[m]['wall_seconds_from_first_time_import'] for m in MODES),
        joint_fit_seconds=summaries['train']['joint_training_seconds'],
        resumed_outer_child_wall_seconds=resumed['resumed_outer_children_wall_seconds'],
        inherited_bank_cost=requests['train']['inherited_data_cost'],
        scope='Joint research ledger includes both fitted fields and controls. Shared teacher/data costs are not halved. Prior source-selection/history costs and original training outer completion are not fully measured; no complete amortized-cost or deployment-speed claim.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    started, cpu_started = time.perf_counter(), time.process_time()
    torch.set_num_threads(4)
    directory, out = args.input_dir.resolve(), args.output_dir.resolve()
    require(not out.exists(), 'refuse to overwrite analysis directory')
    require(all((directory/m/'summary.json').is_file() for m in MODES), 'all four completed stage summaries required')
    require(read_json(directory/'pipeline_resume_status.json').get('complete') is True, 'resume driver must be complete')
    out.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    summaries, requests, metadata, checkpoint = identities(directory, audit)
    train = training(directory, summaries['train'], audit)
    teacher = validation(directory, summaries['validate'], metadata['validation']['labels'], out, audit)
    moment_result = finite_moments(directory, out)
    actual = rollout(directory, summaries['rollout'], out, audit)
    cost = costs(directory, summaries, requests, audit)
    result = dict(protocol='raev2_paired_bridge_cpu_summary_v1', complete=True,
        input_dir=str(directory), created_utc=datetime.now(timezone.utc).isoformat(),
        source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        verified_artifact_count=len(audit.artifacts), verified_artifacts=list(audit.artifacts.values()),
        checkpoint=checkpoint, training=train, teacher_validation=teacher,
        finite_teacher_map_moments=moment_result, actual_rollout=actual, costs=cost,
        analysis_wall_seconds=time.perf_counter()-started, analysis_cpu_seconds=time.process_time()-cpu_started,
        new_model_calls=0, images_decoded=0, fid_evaluated=False,
        scope='Descriptive mechanism pilot. No independent-row confidence intervals, unbiased-population cross-statistic claim, KL guarantee for finite learned solver, or generation-quality conclusion.')
    write_json(out/'summary.json', result)
    lines = ['# RAEv2 固定配对桥机制实验：CPU 汇总草案', '',
        '已核对四阶段 request/source/checkpoint 链、固定预算和调用计数；完成 10000 条 teacher 记录的赋值与时刻检查，以及 96 条实际轨迹末端的 FP32、有限性、原始张量 SHA 和 moment 复算。', '',
        '| Teacher 查询 | target energy | prediction energy | cross | risk gain = 2cross − pred |',
        '|---|---:|---:|---:|---:|']
    for name in TEACHERS:
        means = teacher['teacher_queries'][name]['mean']
        lines.append('| '+name+' | '+' | '.join(f'{means[k]:.10g}' for k in TERMS)+' |')
    lines += ['', '上述风险只针对真正 teacher 插值，按原生 β 归一化的向量目标。每图及噪声复用于 10 个时刻；per-class 文件是描述性聚类汇总，不给独立行 CI。部署 midpoint 不使用配对 target 冒充监督。', '',
        '| 有限 teacher map | 全 2048 维经验均值差平方（100 时刻平均） | 全 2048 维非对角交叉量（100 时刻平均） |',
        '|---|---:|---:|']
    for arm in ARMS:
        stat = moment_result['per_arm'][arm]
        lines.append(f"| {arm} | {stat['empirical_squared_mean_gap']['mean']:.10g} | {stat['offdiagonal_cross_statistic']['mean']:.10g} |")
    lines += ['', '固定 cohort 包含异质类别/图像；交叉量不是总体 squared mean gap 的无偏估计，可以为负。归档可以拆分前 1024 维空间均值和后 1024 维空间二阶矩的经验均值差，但没有分块 squared_delta_sum，不能拆出各自交叉量。', '',
        'actual rollout 的 moment 仅描述实际访问状态；32 个固定类不代表全类别质量。没有真实 X、denoising risk、解码或 FID。详见 rollout_moment_descriptions.csv。', '',
        f"四阶段 runner 内墙钟相加为 {cost['sum_runner_wall_seconds']:.6f} s；正式两网 joint fit 为 {cost['joint_fit_seconds']:.6f} s。调用总账见 summary.json，数值 pilot、mean-only 对照、验证及轨迹诊断均保留。", '',
        '原训练父进程消失，最终权重和训练 summary 完整，之后只续跑验证/rollout。训练外层返回码及精确墙钟缺测，未伪造；UTC 边界另记。既有 bank 6000 次编码/750 次前向、113.860739 s 为历史局部成本，更早来源选择与研究成本未闭合。不能据此宣称完整成本匹配。', '',
        '**本实验不是 FID 评估；5% 目标仍未达到。**']
    with (out/'README_ZH.md').open('x') as dest:
        dest.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True, output_dir=str(out), verified_artifacts=len(audit.artifacts),
        analysis_wall_seconds=result['analysis_wall_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
