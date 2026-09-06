#!/usr/bin/env python3
"""Bounded independent CPU audit; no project imports, model execution, or fitting."""
import time
START = time.perf_counter()
CPU_START = time.process_time()
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import numpy as np

R = Path(__file__).resolve().parent
REPO = Path('/home/zhoushunyu/eqvae')
OUTPUT = R / 'potential_omitted_witness_independent_review_v1.json'
FIELDS = ('residual_dot_g1', 'residual_dot_g2', 'g1_energy', 'g1_dot_g2', 'g2_energy', 'residual_energy_fp64')
HASHED = {}
CHECKS = 0


def require(ok, message):
    global CHECKS
    CHECKS += 1
    if not ok:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in HASHED:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        require(path.stat().st_mtime_ns == stat.st_mtime_ns, f'changed during hash: {path}')
        HASHED[key] = {'path': str(path), 'sha256': digest.hexdigest(), 'size_bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns}
    return HASHED[key]


def verify(rec):
    actual = record(rec['path'])
    for key in ('sha256', 'size_bytes', 'mtime_ns'):
        if key in rec:
            require(actual[key] == rec[key], f'{key} mismatch: {rec["path"]}')
    return actual


def read(path):
    return json.loads(Path(path).read_text())


def same_content(a, b, name):
    require(a['sha256'] == b['sha256'] and a['size_bytes'] == b['size_bytes'], f'{name} identity differs')


def main():
    require(not OUTPUT.exists(), 'refusing to overwrite an earlier review')
    prep_summary_path = R / 'potential_omitted_mean_v1/summary.json'
    prep = read(prep_summary_path)
    verify(prep['request'])
    prep_request = read(prep['request']['path'])
    verify(prep_request['training_request'])
    train = read(prep_request['training_request']['path'])
    require(prep['complete'] and prep['rank'] == 128, 'invalid preparation')
    for name in ('potential', 'config'):
        verify(prep_request[name])
    source_paths = {
        'train_raev2_observable_potential.py': REPO / 'experiments/train_raev2_observable_potential.py',
        'raev2_observable_potential.py': REPO / 'experiments/raev2_observable_potential.py',
        'guidance_utils.py': REPO / 'external/RAEv2/src/utils/guidance_utils.py',
    }
    source_evidence = []
    for name, path in source_paths.items():
        frozen = verify(train['sources'][name])
        current = record(path)
        same_content(current, frozen, name)
        source_evidence.append({'module': name, 'current_repository_source': current,
                               'frozen_training_source': frozen, 'sha256_equal': True})

    summary_root = R / 'potential_omitted_witness_summary_v1'
    reported = read(summary_root / 'summary.json')
    verify(reported['request'])
    verify(reported['paired_by_image'])
    aggregation_request = read(reported['request']['path'])
    require(reported['complete'] and reported['times'] == 100 and reported['samples'] == 1000,
            'incomplete aggregation')
    require(len(aggregation_request['inputs']) == 4, 'expected four aggregation sources')
    aggregated_steps = {int(Path(a['path']).stem[4:]): a for a in aggregation_request['step_artifacts']}
    aggregated_old = {int(Path(a['path']).stem[4:]): a for a in aggregation_request['old_step_artifacts']}
    require(len(aggregation_request['step_artifacts']) == len(aggregation_request['old_step_artifacts']) == 100,
            'aggregation needs exactly 100 new and old artifacts')
    requests, workers, worker_records = [], [], []
    step_records, old_records = {}, {}
    common = ('protocol', 'mode', 'prepare_summary', 'prepare_request', 'runner', 'time_grid',
              'time_probability', 'time_weight_normalizer', 'validation_metadata', 'baseline',
              'potential', 'batch_size', 'noise_seed', 'precision', 'old_validation_root')
    old_requests = []
    for shard in range(3):
        old_path = R / f'observable_potential_validation_v1/shard{shard}'
        old_summary = read(old_path / 'summary.json')
        verify(old_summary['request'])
        old_requests.append(read(old_path / 'request.json'))
    for shard in range(4):
        path = R / f'potential_omitted_witness_v1/shard{shard}'
        request, worker = read(path / 'request.json'), read(path / 'summary.json')
        request_record = verify(worker['request'])
        worker_record = record(path / 'summary.json')
        same_content(request_record, verify(aggregation_request['inputs'][shard]['request']), 'aggregate request')
        same_content(worker_record, verify(aggregation_request['inputs'][shard]['summary']), 'aggregate worker')
        require(worker['complete'] and worker['old_residual_mse_bitwise_parity'], 'worker failed')
        require(request['shard_index'] == shard and request['num_shards'] == 4, 'shard identity')
        expected = list(range(shard, 100, 4))
        require(request['time_indices'] == worker['time_indices'] == expected, 'time partition')
        require(len(worker['step_artifacts']) == len(request['old_step_artifacts']) == 25, 'worker artifact count')
        require(sorted(p.name for p in path.glob('step*.npz')) == [f'step{k:03d}.npz' for k in expected],
                'unexpected or missing step files')
        if requests:
            for key in common:
                require(request[key] == requests[0][key], f'four request disagreement: {key}')
        same_content(verify(request['prepare_summary']), record(prep_summary_path), 'prepare summary')
        same_content(verify(request['prepare_request']), prep['request'], 'prepare request')
        same_content(verify(request['runner']), record(path / 'runner_source.py'), 'frozen worker runner')
        for old in old_requests:
            for key in ('time_grid', 'time_probability', 'time_weight_normalizer', 'batch_size'):
                require(request[key] == old[key], f'old validation {key} mismatch')
            require(request['noise_seed'] == old['seeds']['validation_noise'] == 202609094, 'noise identity')
            same_content(request['baseline'], old['baseline_checkpoint'], 'old baseline')
            same_content(request['potential'], old['potential_checkpoint'], 'old potential')
            same_content(prep_request['config'], old['config'], 'old config')
        for k, new_record, old_record in zip(expected, worker['step_artifacts'], request['old_step_artifacts'], strict=True):
            require(k not in step_records, 'duplicate time')
            require(Path(new_record['path']).resolve() == (path / f'step{k:03d}.npz').resolve(), 'new step path')
            require(Path(old_record['path']).resolve() == (R / f'observable_potential_validation_v1/shard{k%3}/step{k:03d}.npz').resolve(), 'old step path')
            step_records[k], old_records[k] = verify(new_record), verify(old_record)
            same_content(step_records[k], verify(aggregated_steps[k]), 'aggregation new step')
            same_content(old_records[k], verify(aggregated_old[k]), 'aggregation old step')
        requests.append(request)
        workers.append(worker)
        worker_records.append({'request': request_record, 'summary': worker_record})
    require(sorted(step_records) == list(range(100)), 'incomplete 100 step coverage')
    reference = requests[0]
    verify(reference['validation_metadata'])
    with np.load(reference['validation_metadata']['path'], allow_pickle=False) as data:
        ids, labels = data['ids'].copy(), data['labels'].copy()
    require(np.array_equal(ids, np.arange(1000)), 'bank ID order')
    require(np.array_equal(np.sort(labels), np.arange(1000)), 'one heldout image per class')
    grid = np.asarray(reference['time_grid'], dtype=np.float64)
    probability = np.asarray(reference['time_probability'], dtype=np.float64)
    require(grid.shape == (101,) and grid[-1] == 0 and np.all(np.diff(grid) < 0), 'time grid')
    require(probability.shape == (100,) and abs(probability.sum() - 1) < 1e-12, 'probability')
    weights = probability * reference['time_weight_normalizer']
    formula_weights = -np.diff(grid) / np.square(grid[:-1])
    weight_error = float(np.max(np.abs(weights - formula_weights)))
    require(np.allclose(weights, formula_weights, rtol=1e-12, atol=1e-12), 'weight units h/t^2')

    # Deliberately use a sequential weighted sum, independent of the production einsum.
    integrated = np.zeros((1000, 6), dtype=np.float64)
    old_integrated = np.zeros((1000, 3), dtype=np.float64)
    parity_count, max_parity_error, max_precision_error = 0, 0.0, 0.0
    manifest = []
    for k in range(100):
        with np.load(step_records[k]['path'], allow_pickle=False) as new, np.load(old_records[k]['path'], allow_pickle=False) as old:
            require(int(new['step']) == k and float(new['time']) == grid[k], 'per-step time identity')
            for value in (new, old):
                require(np.array_equal(value['ids'], ids) and np.array_equal(value['labels'], labels), 'per-step IDs/labels')
            residual = new['residual_mse_fp32_parity']
            require(residual.dtype == old['residual_mse'].dtype == np.float64, 'parity array dtype')
            require(np.array_equal(residual, old['residual_mse']), f'R2 exact parity failed at {k}')
            require(residual.tobytes() == old['residual_mse'].tobytes(), f'R2 bitwise parity failed at {k}')
            max_parity_error = max(max_parity_error, float(np.max(np.abs(residual - old['residual_mse']))))
            parity_count += residual.size
            for j, key in enumerate(FIELDS):
                a = new[key]
                require(a.shape == (1000,) and a.dtype == np.float64 and np.isfinite(a).all(), f'invalid {key}')
                integrated[:, j] += weights[k] * a
            max_precision_error = max(max_precision_error, float(np.max(np.abs(new['residual_energy_fp64'] - residual))))
            for j, key in enumerate(('residual_mse', 'correction_energy', 'residual_dot_correction')):
                require(old[key].shape == (1000,) and np.isfinite(old[key]).all(), f'invalid old {key}')
                old_integrated[:, j] += weights[k] * old[key]
        manifest.append({'step': k, 'new': step_records[k], 'old': old_records[k], '1000_R2_values_bitwise_equal': True})

    b_by_image = integrated[:, :2]
    b = np.mean(b_by_image, axis=0)
    centered = b_by_image - b
    covariance = centered.T @ centered / 999
    sem = np.sqrt(np.diag(covariance) / 1000)
    h11, h12, h22 = np.mean(integrated[:, 2:5], axis=0)
    H = np.array([[h11, h12], [h12, h22]])
    determinant = h11 * h22 - h12 * h12
    require(h11 > 0 and determinant > 0, 'Gram must be positive definite')
    # Analytic 2x2 solve independently checks the primary np.linalg.solve computation.
    alpha = np.array([(h22 * b[0] - h12 * b[1]) / determinant,
                      (h11 * b[1] - h12 * b[0]) / determinant])
    plugin = float((h22 * b[0]**2 - 2*h12*b[0]*b[1] + h11*b[1]**2) / determinant)
    old_gain_by_image = 2 * old_integrated[:, 2] - old_integrated[:, 1]
    old_own_by_image = old_integrated[:, 2] - old_integrated[:, 1]
    old_gain = float(old_gain_by_image.mean())
    baseline = float(old_integrated[:, 0].mean())
    relative_old_gain = plugin / old_gain
    additional_old_scale = float(old_own_by_image.mean()**2 / old_integrated[:, 1].mean())
    metrics = {
        'b': b.tolist(), 'H': H.tolist(), 'descriptive_class_sem': sem.tolist(),
        'covariance_of_image_integrals': covariance.tolist(),
        'gram_eigenvalues': np.linalg.eigvalsh(H).tolist(),
        'same_bank_plugin_energy': plugin, 'analytic_diagnostic_coefficients': alpha.tolist(),
        'old_integrated_proxy_gain': old_gain,
        'old_gain_descriptive_class_sem': float(old_gain_by_image.std(ddof=1) / np.sqrt(1000)),
        'plugin_fraction_of_old_gain': relative_old_gain,
        'plugin_percent_of_old_gain': relative_old_gain * 100,
        'old_baseline_velocity_energy': baseline,
        'plugin_fraction_of_old_baseline_energy': plugin / baseline,
        'old_potential_own_witness': float(old_own_by_image.mean()),
        'old_potential_own_witness_sem': float(old_own_by_image.std(ddof=1) / np.sqrt(1000)),
        'old_scale_additional_energy_plugin': additional_old_scale,
    }
    comparisons = []
    def compare(name, actual, expected):
        a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        error = float(np.max(np.abs(a - e)))
        require(a.shape == e.shape and np.allclose(a, e, rtol=1e-12, atol=1e-14), f'aggregation numerical mismatch: {name}')
        comparisons.append({'quantity': name, 'max_absolute_difference': error, 'within_rtol_1e-12_atol_1e-14': True})
    compare('b', b, [reported['witness_g1']['mean'], reported['witness_g2']['mean']])
    compare('SEM', sem, [reported['witness_g1']['descriptive_class_sem'], reported['witness_g2']['descriptive_class_sem']])
    compare('H', H, reported['gram'])
    compare('covariance', covariance, reported['covariance_of_image_integrals'])
    compare('Gram eigenvalues', np.linalg.eigvalsh(H), reported['gram_eigenvalues'])
    compare('plugin energy', plugin, reported['same_bank_critic_plugin']['energy'])
    compare('analytic diagnostic coefficients', alpha, reported['same_bank_critic_plugin']['coefficients_for_diagnostic_only'])
    compare('old achieved proxy gain', old_gain, reported['existing_potential_competitor']['achieved_gain']['mean'])
    compare('old proxy gain SEM', metrics['old_gain_descriptive_class_sem'], reported['existing_potential_competitor']['achieved_gain']['descriptive_class_sem'])
    compare('fraction of old achieved proxy gain', relative_old_gain, reported['same_bank_critic_plugin']['fraction_of_old_achieved_gain'])
    compare('baseline energy', baseline, reported['old_baseline_velocity_coupling_energy_per_dimension'])
    compare('fraction of baseline energy', plugin / baseline, reported['same_bank_critic_plugin']['fraction_of_old_residual_energy'])
    compare('old potential own witness', metrics['old_potential_own_witness'], reported['existing_potential_competitor']['own_witness']['mean'])
    compare('old potential own witness SEM', metrics['old_potential_own_witness_sem'], reported['existing_potential_competitor']['own_witness']['descriptive_class_sem'])
    compare('old potential extra scale plugin', additional_old_scale, reported['existing_potential_competitor']['same_bank_additional_energy_plugin'])
    with np.load(reported['paired_by_image']['path'], allow_pickle=False) as paired:
        require(np.array_equal(paired['ids'], ids) and np.array_equal(paired['labels'], labels), 'aggregate paired IDs/labels')
        compare('paired_by_image integrated_b', b_by_image, paired['integrated_b'])
        expected_gram = np.zeros((1000, 2, 2))
        expected_gram[:, 0, 0] = integrated[:, 2]
        expected_gram[:, 0, 1] = expected_gram[:, 1, 0] = integrated[:, 3]
        expected_gram[:, 1, 1] = integrated[:, 4]
        compare('paired_by_image integrated_gram', expected_gram, paired['integrated_gram'])
        for j, key in enumerate(('old_residual_energy', 'old_c2', 'old_rc')):
            compare('paired_by_image ' + key, old_integrated[:, j], paired[key])

    cost_comparisons = {}
    for key, expected in (('stage2_forward_calls', 3200), ('stage2_sample_forwards', 100000),
                          ('potential_input_gradient_calls', 4), ('potential_input_gradient_sample_forwards', 128)):
        value = sum(worker[key] for worker in workers)
        require(value == reported['cost'][key] == expected, f'model call ledger mismatch: {key}')
        cost_comparisons[key] = value
    compare('sum worker phase wall', sum(w['phase_wall_seconds'] for w in workers), reported['cost']['sum_worker_phase_wall_seconds'])
    compare('sum worker script wall', sum(w['cost']['wall_seconds_from_script_start'] for w in workers), reported['cost']['sum_worker_script_wall_seconds'])
    compare('max worker allocated memory', max(w['peak_gpu_allocated_bytes'] for w in workers), reported['cost']['max_worker_gpu_allocated_bytes'])
    result = {
        'protocol': 'raev2_potential_omitted_witness_independent_cpu_review_v1',
        'complete': True, 'passed': True, 'utc_finished': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'script': record(__file__), 'reviewed_summary': record(summary_root / 'summary.json'),
        'reviewed_aggregation_request': record(summary_root / 'request.json'),
        'worker_inputs': worker_records, 'checks_passed': CHECKS,
        'counts': {'new_steps': 100, 'old_steps': 100, 'new_requests': 4, 'old_requests': 3,
                   'paired_R2_entries': parity_count, 'witness_scalar_entries': 600000,
                   'actual_repository_sources_matched': 3, 'unique_files_hashed': len(HASHED)},
        'all_old_R2_entries_bitwise_equal': True, 'max_old_R2_absolute_error': max_parity_error,
        'max_FP64_vs_FP32_R2_difference_expected': max_precision_error,
        'max_h_over_t_squared_weight_absolute_difference': weight_error,
        'recomputed_metrics': metrics, 'summary_comparisons': comparisons,
        'max_aggregate_statistic_absolute_difference': max(c['max_absolute_difference'] for c in comparisons),
        'source_identity_gap_remediation': {
            'gap': 'The prepare runner verifies frozen training source snapshots but lacks an explicit equality assertion against actual repository modules imported at execution.',
            'remediation': 'This independent audit explicitly compares SHA256 and sizes of all three current repository source paths against verified frozen training snapshots; all pass. No main source was changed and no GPU rerun was required.',
            'source_evidence': source_evidence,
        },
        'model_call_ledger_checked': cost_comparisons,
        'step_manifest': manifest,
        'limitations': [
            'CPU audit of stored artifacts; no GPU/model/noise regeneration. All 100000 stored FP32 residual MSE values were independently compared byte-for-byte with the old validation.',
            'Current source file identity is verified, not historical loaded Python bytecode; no source mutation was observed during hashing.',
            'Baseline checkpoint digest identity is checked against the original requests; this bounded audit does not rehash the 10.5GB checkpoint or full latent banks.',
            'No independent recomputation of the per-step Q projections or model outputs; prior code review verified their formulas and frozen rowspace provenance.',
            'SEM describes variation over 1000 fixed class/image units after all 100 times are integrated, and is not a repeated-noise population confidence interval.',
            'bTHinvb and coefficients are same-bank descriptive plug-in diagnostics with estimation optimism; not a certified population bound, fitted guidance, or FID prediction.',
            'Worker wall times are summed overlapping process windows, not elapsed critical-path duration or pure GPU time; GPU allocated memory peaks cover the validation phase after model loading.',
            'Audit timing includes Python imports after the time import, hash reads, NPZ reads, and numerical checks; excludes interpreter startup, prior interactive inspection/source preparation, final JSON serialization and exit.',
        ],
        'cost': {'wall_seconds_from_script_start': time.perf_counter() - START,
                 'cpu_seconds_from_script_start': time.process_time() - CPU_START,
                 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 'gpu_calls': 0, 'model_calls': 0, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'python': sys.version, 'numpy': np.__version__},
    }
    temporary = OUTPUT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUTPUT)
    print(json.dumps({'passed': True, 'output': str(OUTPUT), 'checks_passed': CHECKS,
                      'counts': result['counts'], 'max_difference': result['max_aggregate_statistic_absolute_difference'],
                      'metrics': metrics, 'cost': result['cost']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
