#!/usr/bin/env python3
"""Analyze complete, explicit three-arm 5K caches on CPU; never extract features."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_name] = '4'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from fid_influence import features_influence, paired_contrast, stratified_covariance

HERE = Path(__file__).resolve().parent
ARMS = ('official100', 'potential100', 'official105')
SEED, N, DIM = 202609101, 5000, 2048
COMMIT = '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
REFERENCE_SHA = '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
LIMITS = [
    'Local first-order influence/delta-method approximation, not calibrated finite-N confidence coverage.',
    'Reference moments, trained candidate, class allocation and evaluation protocol are held fixed.',
    'SE describes within-class noise variability; it excludes reference estimation, training, and method-selection uncertainty.',
    'Does not remove model-dependent finite-sample FID bias or extrapolate 5K FID to 50K/population FID.',
    'High feature dimension (2048) relative to N=5000 can make first-order normal intervals inaccurate.',
    'Cannot replace independent-noise confirmation, appropriately sized evaluation or full preparation-cost matching.',
    'Feature-to-ID order is bound by evaluator source order, cache key and sample archive identity; caches contain no independent IDs.',
]


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'size_bytes': path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text())


def assert_record(actual, expected, context):
    if (Path(expected['path']).resolve() != Path(actual['path']).resolve()
            or actual['sha256'] != expected['sha256']
            or actual['size_bytes'] != expected['size_bytes']):
        raise ValueError(f'{context}: artifact identity mismatch')


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='append', nargs=4, required=True,
                        metavar=('NAME', 'FEATURE_PT', 'SAMPLES_NPZ', 'MERGE_SUMMARY'))
    parser.add_argument('--evaluation-request', type=Path, required=True)
    parser.add_argument('--metrics-json', type=Path, required=True)
    parser.add_argument('--execution-summary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=HERE/'results_v1')
    args = parser.parse_args(argv)
    names = [v[0] for v in args.arm]
    if len(names) != 3 or set(names) != set(ARMS):
        parser.error('--arm must name each of official100, potential100, official105 exactly once')
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(HERE) or args.output_dir == HERE:
        parser.error('output must be a new subdirectory of this analysis directory')
    return args


def main(args):
    started, cpu_started = time.perf_counter(), time.process_time()
    if args.output_dir.exists():
        raise FileExistsError('refusing to overwrite or implicitly resume an analysis')
    # Establish completion and published metric identities before opening features.
    execution = read_json(args.execution_summary)
    if execution.get('complete') is not True or execution.get('evaluation_process', {}).get('returncode') != 0:
        raise ValueError('all sampling and common evaluation must be complete first')
    metrics_record = record(args.metrics_json)
    assert_record(metrics_record, execution['metrics'], 'completed metrics')
    evaluation = read_json(args.evaluation_request)
    if (evaluation.get('all_625_batch_noises_match_across_three_arms') is not True
            or evaluation.get('full_common_protocol_matches_except_mode_and_steps') is not True
            or evaluation.get('evaluator_commit') != COMMIT):
        raise ValueError('paired evaluation identity/protocol is incomplete')
    rows = read_json(args.metrics_json)
    by_name = {row['branch']: row for row in rows}
    expected_names = {f'{name}_seed{SEED}' for name in ARMS}
    if len(rows) != 3 or set(by_name) != expected_names:
        raise ValueError('metrics must contain exactly the frozen three arms and seed')
    reference_record = record(args.reference)
    assert_record(reference_record, evaluation['evaluator_assets']['reference'], 'reference')
    if reference_record['sha256'] != REFERENCE_SHA:
        raise ValueError('frozen reference changed')
    with np.load(args.reference, allow_pickle=False) as values:
        if {'mu', 'sigma'}.issubset(values.files):
            reference_mean, reference_covariance = values['mu'].copy(), values['sigma'].copy()
        elif {'ref_mu', 'ref_sigma'}.issubset(values.files):
            reference_mean, reference_covariance = values['ref_mu'].copy(), values['ref_sigma'].copy()
        else:
            raise ValueError('unrecognized reference schema')
    if reference_mean.shape != (DIM,) or reference_covariance.shape != (DIM, DIM):
        raise ValueError('reference dimension differs')
    toy = read_json(HERE/'toy_validation.json')
    if toy.get('passed') is not True or toy.get('checks') != 96:
        raise ValueError('successful derivative/variance validation is missing')
    for source in toy['source_records']:
        if record(source['path'])['sha256'] != source['sha256']:
            raise ValueError('validated mathematical source changed')
    inputs, feature_paths = {}, {}
    expected_ids = np.arange(N, dtype=np.int64)
    labels = expected_ids % 1000
    for name, raw_feature, raw_archive, raw_summary in args.arm:
        feature, archive, summary_path = map(lambda x: Path(x).resolve(), (raw_feature, raw_archive, raw_summary))
        row = by_name[f'{name}_seed{SEED}']
        if row['evaluator_commit'] != COMMIT or row['fid_reference'] != 'imagenet_256_fid_stats':
            raise ValueError(f'{name}: evaluation protocol differs')
        archive_record = record(archive)
        assert_record(archive_record, evaluation['input_archives'][name], f'{name} evaluator input')
        if archive_record['sha256'] != row['sample_sha256'] or archive != Path(row['sample_path']).resolve():
            raise ValueError(f'{name}: metrics refer to different samples')
        summary = read_json(summary_path)
        expected_mode, expected_steps = ('potential', 100) if name == 'potential100' else ('official', int(name[-3:]))
        if (summary.get('complete') is not True or summary.get('samples') != N
                or summary.get('classes') != 1000 or summary.get('samples_per_class') != 5
                or summary.get('seed') != SEED or summary.get('mode') != expected_mode
                or summary.get('num_steps') != expected_steps):
            raise ValueError(f'{name}: merge metadata differs from frozen arm')
        assert_record(archive_record, summary['sample_archive'], f'{name} merged archive')
        # np.load is lazy here: only ids and labels are decompressed, never arr_0.
        with np.load(archive, allow_pickle=False) as values:
            if not np.array_equal(values['ids'], expected_ids) or not np.array_equal(values['labels'], labels):
                raise ValueError(f'{name}: paired feature order is invalid')
        expected_cache_name = f"{row['branch']}-{row['sample_sha256'][:16]}-inception.features.pt"
        if feature.name != expected_cache_name:
            raise ValueError(f'{name}: feature cache key is not bound to evaluator input')
        inputs[name] = {'features': record(feature), 'sample_archive': archive_record,
                        'merge_summary': record(summary_path), 'official_metrics': row}
        feature_paths[name] = feature
    source_records = [record(HERE/name) for name in ('run_scale_analysis.py', 'fid_influence.py', 'validate_toy.py', 'toy_validation.json')]
    args.output_dir.mkdir()
    write_json(args.output_dir/'request.json', {
        'protocol': 'raev2_fixed_three_arm_fid_influence_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'inputs': inputs,
        'reference': reference_record, 'metrics': metrics_record,
        'evaluation_request': record(args.evaluation_request), 'execution_summary': record(args.execution_summary),
        'source_records': source_records, 'feature_dimension': DIM, 'samples': N,
        'classes': 1000, 'noise_replicates_per_class': 5, 'seed': SEED,
        'limits': LIMITS, 'cpu_threads': 4, 'gpu_calls': 0, 'model_calls': 0,
        'source_toy_checks_repeated': False,
    })
    import torch
    torch.set_num_threads(4)
    arms, summaries, arrays = {}, {}, {'ids': expected_ids, 'labels': labels}
    for name in ARMS:
        tensor = torch.load(feature_paths[name], map_location='cpu', weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tuple(tensor.shape) != (N, DIM):
            raise ValueError(f'{name}: expected raw FP32 feature tensor [5000,2048]')
        result = features_influence(tensor.numpy(), reference_mean, reference_covariance)
        official_fid = float(inputs[name]['official_metrics']['fid'])
        discrepancy = result['fid'] - official_fid
        if abs(discrepancy) > 1e-5:
            raise ValueError(f'{name}: local FID does not reproduce official metric: {discrepancy}')
        variance, _, _, _ = stratified_covariance(result['influence'], labels)
        summaries[name] = {'official_fid': official_fid, 'cpu_spd_fid': result['fid'],
            'cpu_minus_official_fid': discrepancy, 'stratified_first_order_se': float(np.sqrt(variance[0,0])),
            'mean_influence': float(result['influence'].mean()), 'alpha_N_over_N_minus_1': result['alpha'],
            'audit': result['audit']}
        arrays[f'{name}_influence'] = result['influence']
        # Keep only the paired scalar influence/FID; release high-dimensional matrices.
        arms[name] = {'fid': result['fid'], 'influence': result['influence']}
        del result, tensor
        print(json.dumps({'arm_complete': name, **summaries[name]}), flush=True)
    contrasts = {}
    for baseline in ('official100', 'official105'):
        name = f'potential100_vs_{baseline}'
        contrast, diff_if, relative_if, within = paired_contrast(arms['potential100'], arms[baseline], labels)
        contrasts[name] = contrast
        arrays[f'{name}_difference_influence'] = diff_if
        arrays[f'{name}_relative_influence'] = relative_if
        arrays[f'{name}_within_class_variances'] = within
    joint, classes, counts, _ = stratified_covariance(np.column_stack([arms[name]['influence'] for name in ARMS]), labels)
    arrays['class_ids'], arrays['class_counts'] = classes, counts
    arrays['arm_estimator_stratified_covariance'] = joint
    array_path = args.output_dir/'influences.npz'
    with array_path.open('xb') as stream:
        np.savez(stream, **arrays)
    summary = {
        'complete': True, 'protocol': 'raev2_fixed_three_arm_fid_influence_v1',
        'arms': summaries, 'contrasts': contrasts, 'paired_arm_order': list(ARMS),
        'arm_estimator_stratified_covariance': joint.tolist(),
        'influences': record(array_path), 'request': record(args.output_dir/'request.json'),
        'limits': LIMITS, 'goal_achieved': False, 'full_cost_comparison_complete': False,
        'gpu_calls': 0, 'model_calls': 0, 'feature_extraction_calls': 0,
        'wall_seconds_excluding_imports_and_final_summary_write': time.perf_counter()-started,
        'cpu_seconds_excluding_imports_and_final_summary_write': time.process_time()-cpu_started,
        'maximum_resident_set_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
    }
    write_json(args.output_dir/'summary.json', summary)
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main(parse_args())
