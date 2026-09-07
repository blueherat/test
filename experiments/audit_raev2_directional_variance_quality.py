"""Independently audit the one-scalar directional covariance's completed1K and5K, without fitting.

Reuses previously audited controls, verifies new merged pixels and every input
identity, and reconstructs candidate FID independently of scipy.sqrtm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.audit_raev2_screen_fid_20260907 import load_moments, resolve
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def independent_fid(features, reference_mean, reference_covariance, reference_root=None):
    x = np.asarray(features, dtype=np.float64)
    mean = x.mean(0)
    centered = x-mean
    mean_term = float(np.sum((mean-reference_mean)**2))
    trace = float(np.sum(centered**2)/(len(x)-1))
    if len(x) < x.shape[1]:
        product = centered@reference_covariance@centered.T/(len(x)-1)
        method = 'rank-N Gram eigenvalues'
    else:
        assert reference_root is not None
        covariance = centered.T@centered/(len(x)-1)
        product = reference_root@covariance@reference_root
        method = 'symmetric reference-root covariance product'
    spectrum = np.linalg.eigvalsh((product+product.T)/2)
    assert spectrum.min() > -1e-8
    covariance_term = float(trace+np.trace(reference_covariance)-2*np.sqrt(np.maximum(spectrum, 0)).sum())
    return {'fid': mean_term+covariance_term, 'mean_term': mean_term,
            'covariance_term': covariance_term, 'sample_covariance_trace': trace,
            'method': method, 'minimum_product_eigenvalue': float(spectrum.min())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, choices=[1000, 5000], required=True)
    args = parser.parse_args()
    n = args.samples
    suffix = 'screen1k' if n == 1000 else 'confirm5k'
    folder = DATA/f'directional_variance_{suffix}'
    execution = json.loads((folder/'execution.json').read_text())
    assert execution['complete']
    fit_path = DATA/'conditional_variance_fit/execution.json'
    fit = json.loads(fit_path.read_text())
    assert fit['complete'] and fit['optimizer_converged'] and fit['validation']['entry_condition_passed']
    assert sha(DATA/'conditional_variance_fit/head.pt') == fit['checkpoint_sha256']
    calibration_path = DATA/'directional_variance_moments/calibration.json'
    calibration = json.loads(calibration_path.read_text())
    assert calibration['complete'] and calibration['validation']['entry_condition_passed']
    assert calibration['spherical_head_sha256'] == fit['checkpoint_sha256']
    assert calibration['plan']['global_fitted_parameters'] == 1
    mode = 'directional_variance'
    metric, = json.loads((folder/'metrics.json').read_text())
    summary = json.loads((folder/mode/'summary.json').read_text())
    expected_seed = 202609071 if n == 1000 else 202609072
    assert metric['branch'] == mode and summary['complete']
    assert summary['count'] == n and summary['seed'] == expected_seed and summary['steps'] == 100
    assert summary['sample_model_calls'] == summary['sample_variance_feature_queries'] == summary['sample_directional_covariance_queries'] == n*100
    assert summary['sample_prefix_forward_calls'] == summary['sample_prefix_backward_calls'] == summary['sample_ratio_backward_calls'] == summary['sample_critic_backward_calls'] == 0
    assert metric['sample_sha256'] == summary['sample_sha256'] == sha(Path(metric['sample_path']))
    assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
    control_folder = DATA/('ancestral_screen1k' if n == 1000 else 'weak_confirm5k')
    control = json.loads((control_folder/'official/summary.json').read_text())
    control_metrics = json.loads((control_folder/'metrics.json').read_text())
    control_metric = next(row for row in control_metrics if row['branch'] == 'official')
    interval_metric = next(row for row in control_metrics if row['branch'] == 'piecewise')
    previous_path = ROOT/'experiments/results/raev2_guidance_20260907'/('fid_audit.json' if n == 1000 else 'semantic_confirm5k.json')
    previous = json.loads(previous_path.read_text())
    assert previous['complete']
    old = next(row for row in previous['rows'] if row['mode'] == 'official')
    previous_fid = old['fid'] if n == 1000 else old['independent_fid']
    assert abs(previous_fid-control_metric['fid']) < 1e-3
    assert sha(Path(control_metric['sample_path'])) == control_metric['sample_sha256'] == control['sample_sha256']
    control_features = control_folder/'official_feature_cache'/f"official-{control_metric['sample_sha256'][:16]}-inception.features.pt"
    assert sha(control_features) == old['feature_sha256']
    interval = json.loads((control_folder/'piecewise/summary.json').read_text())
    interval_old = next(row for row in previous['rows'] if row['mode'] == 'piecewise')
    interval_previous_fid = interval_old['fid'] if n == 1000 else interval_old['independent_fid']
    assert abs(interval_previous_fid-interval_metric['fid']) < 1e-3
    assert interval['paired_noise_labels_sha256'] == control['paired_noise_labels_sha256']
    assert sha(Path(interval_metric['sample_path'])) == interval_metric['sample_sha256'] == interval['sample_sha256']
    interval_features = control_folder/'official_feature_cache'/f"piecewise-{interval_metric['sample_sha256'][:16]}-inception.features.pt"
    assert sha(interval_features) == interval_old['feature_sha256']
    reference_path = Path(resolve('imagenet_256_fid_stats'))
    assert sha(reference_path) == previous['reference_sha256'] == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    reference_mean, reference_covariance = load_moments(str(reference_path))
    began = time.perf_counter()
    reference_root = None
    if n >= 2048:
        values, vectors = np.linalg.eigh(reference_covariance)
        assert values.min() > -1e-9
        reference_root = (vectors*np.sqrt(np.maximum(values, 0)))@vectors.T
    for original, digest in execution['sources'].items():
        assert sha(folder/'frozen_source'/Path(original).name) == digest
    with np.load(metric['sample_path']) as data:
        merged = data['arr_0']
    assert merged.shape == (n, 256, 256, 3) and merged.dtype == np.uint8
    records, ids, identities = [], [], []
    keys = ['checkpoint_sha256', 'config_sha256', 'decoder_sha256', 'stats_sha256',
            'state_key', 'torch_version', 'batch', 'time_grid']
    for rank in range(4):
        shard = folder/f'shard{rank}'
        request = json.loads((shard/'request.json').read_text())
        baseline_request = json.loads((control_folder/f'shard{rank}/request.json').read_text())
        identity = {key: request[key] for key in keys}
        assert identity == {key: baseline_request[key] for key in keys}
        identities.append(identity)
        guidance = request['conditional_variance']
        assert guidance['checkpoint_sha256'] == fit['checkpoint_sha256']
        assert guidance['plan'] == fit['plan'] and guidance['all_100_times']
        assert guidance['calibration_sha256'] == fit['calibration_sha256']
        assert guidance['no_additional_model_calls_or_input_backward']
        directional = request['directional_variance']
        assert directional['calibration_sha256'] == sha(calibration_path)
        assert directional['kappa'] == calibration['kappa'] and directional['plan'] == calibration['plan']
        assert directional['spherical_head_sha256'] == fit['checkpoint_sha256']
        assert directional['no_extra_model_calls_or_input_backward'] and directional['all_100_times']
        part = json.loads((shard/mode/'summary.json').read_text())
        assert part['complete'] and sha(shard/mode/'samples.npz') == part['sample_sha256']
        records.extend(part['initial_noise'])
        with np.load(shard/mode/'samples.npz') as data:
            index = data['ids']
            assert np.array_equal(merged[index], data['arr_0'])
            ids.extend(index.tolist())
    assert all(item == identities[0] for item in identities)
    assert sorted(ids) == list(range(n))
    records.sort(key=lambda row: row['batch'])
    assert [row['batch'] for row in records] == list(range(n//8))
    paired_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    assert paired_hash == summary['paired_noise_labels_sha256'] == control['paired_noise_labels_sha256']
    feature_path = folder/'official_feature_cache'/f"{mode}-{metric['sample_sha256'][:16]}-inception.features.pt"
    features = torch.load(feature_path, map_location='cpu', weights_only=True).double().numpy()
    assert features.shape == (n, 2048) and np.isfinite(features).all()
    independent = independent_fid(features, reference_mean, reference_covariance, reference_root)
    assert abs(independent['fid']-metric['fid']) < 1e-3
    feature_execution_path = DATA/'conditional_variance_features/execution.json'
    feature_execution = json.loads(feature_execution_path.read_text())
    assert feature_execution['complete'] and feature_execution['all_used_input_hashes_verified']
    assert sha(feature_execution_path) == fit['feature_execution_sha256']
    for path, identity in feature_execution['sources'].items():
        if '/external/RAEv2/src/stage2/models/' in path:
            assert sha(Path(path)) == identity, 'native model source changed since feature extraction'
    global_folder = DATA/('calibrated_screen1k' if n == 1000 else 'final8_legacy_confirm5k')
    global_metric = next(row for row in json.loads((global_folder/'metrics.json').read_text()) if row['branch'] == 'calibrated')
    global_summary = json.loads((global_folder/'calibrated/summary.json').read_text())
    assert global_summary['complete'] and global_summary['count'] == n and global_summary['seed'] == expected_seed
    assert global_summary['steps'] == 100 and global_summary['paired_noise_labels_sha256'] == paired_hash
    assert sha(Path(global_metric['sample_path'])) == global_metric['sample_sha256'] == global_summary['sample_sha256']
    for rank in range(4):
        request = json.loads((global_folder/f'shard{rank}/request.json').read_text())
        assert {key: request[key] for key in keys} == identities[rank]
        assert request['variance_calibration_sha256'] == fit['calibration_sha256']
    global_feature_path = global_folder/'official_feature_cache'/f"calibrated-{global_metric['sample_sha256'][:16]}-inception.features.pt"
    global_features = torch.load(global_feature_path, map_location='cpu', weights_only=True).double().numpy()
    assert global_features.shape == (n, 2048) and np.isfinite(global_features).all()
    global_reconstruction = independent_fid(global_features, reference_mean, reference_covariance, reference_root)
    assert abs(global_reconstruction['fid']-global_metric['fid']) < 1e-3
    costs = {'incremental_teacher_sample_main_calls': sum(s['teacher_sample_main_calls'] for s in feature_execution['splits'].values()),
             'extra_teacher_parity_sample_main_calls': sum(s['extra_parity_sample_main_calls'] for s in feature_execution['splits'].values()),
             'feature_worker_seconds_including_data_access': sum(s['worker_seconds_including_data_access'] for s in feature_execution['splits'].values()),
             'reuse_note': 'Real latent encoding and initial epsilon banks reused; stored native rollout states not used; full historical bank costs in linked native/real execution manifests',
             'native_history_execution_sha256': feature_execution['native_execution_sha256'],
             'real_history_execution_sha256': feature_execution['real_execution_sha256'],
             'convex_fit_cpu_elapsed_seconds': fit['cpu_elapsed_seconds'],
             'candidate_inference_gpu_seconds': summary['trajectory_seconds_sum']+summary['decode_seconds_sum'],
             'official_inference_gpu_seconds': control['trajectory_seconds_sum']+control['decode_seconds_sum'],
             'global_variance_inference_gpu_seconds': global_summary['trajectory_seconds_sum']+global_summary['decode_seconds_sum']}
    moment_path = DATA/'directional_variance_moments/execution.json'
    moments = json.loads(moment_path.read_text())
    assert moments['complete'] and sha(moment_path) == calibration['feature_execution_sha256']
    costs['directional_teacher_sample_main_calls'] = sum(v['teacher_sample_main_calls'] for v in moments['splits'].values())
    costs['directional_teacher_worker_seconds_including_data_access'] = sum(v['worker_seconds_including_data_access'] for v in moments['splits'].values())
    spherical_path = ROOT/f'experiments/results/raev2_guidance_20260907/conditional_variance_{suffix}_audit.json'
    spherical = json.loads(spherical_path.read_text())
    assert spherical['complete'] and spherical['samples'] == n and spherical['seed'] == expected_seed
    assert spherical['checkpoint_sha256'] == fit['checkpoint_sha256']
    assert spherical['paired_noise_labels_sha256'] == paired_hash
    spherical_metric = spherical['candidate']
    assert sha(Path(spherical_metric['sample_path'])) == spherical_metric['sample_sha256']
    spherical_feature_path = DATA/f'conditional_variance_{suffix}'/'official_feature_cache'/f"conditional_variance-{spherical_metric['sample_sha256'][:16]}-inception.features.pt"
    assert sha(spherical_feature_path) == spherical['candidate_feature_sha256']
    record = {'complete': True, 'goal_achieved': False, 'samples': n, 'seed': expected_seed,
              'candidate': metric, 'independent_reconstruction': independent, 'official': control_metric,
              'kappa': calibration['kappa'], 'directional_calibration_sha256': sha(calibration_path),
              'spherical_control': spherical_metric, 'spherical_control_audit_sha256': sha(spherical_path),
              'improvement_vs_spherical_percent': 100*(1-metric['fid']/spherical_metric['fid']),
              'global_variance': global_metric, 'global_variance_reconstruction': global_reconstruction,
              'global_variance_feature_sha256': sha(global_feature_path),
              'improvement_vs_global_variance_percent': 100*(1-metric['fid']/global_metric['fid']),
              'historical_interval': interval_metric, 'interval_prior_independent_fid': interval_previous_fid,
              'official_prior_independent_fid': previous_fid, 'prior_control_audit_sha256': sha(previous_path),
              'candidate_feature_sha256': sha(feature_path), 'checkpoint_sha256': fit['checkpoint_sha256'],
              'reference_sha256': sha(reference_path), 'execution_sha256': sha(folder/'execution.json'),
              'all_merged_pixels_verified_against_shards': True, 'source_snapshots_verified': True,
              'paired_noise_labels_sha256': paired_hash, 'identity': identities[0],
              'relative_fid_improvement_percent': 100*(1-metric['fid']/control_metric['fid']),
              'improvement_vs_best_existing_control_percent': 100*(1-metric['fid']/min(control_metric['fid'], interval_metric['fid'], global_metric['fid'], spherical_metric['fid'])),
              'inference_cost_ratio': costs['candidate_inference_gpu_seconds']/costs['official_inference_gpu_seconds'],
              'costs': costs, 'quality_and_measured_cost_require_review_if_three_percent_passes': True,
              'audit_cpu_elapsed_seconds': time.perf_counter()-began}
    output = ROOT/f'experiments/results/raev2_guidance_20260907/directional_variance_{suffix}_audit.json'
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
