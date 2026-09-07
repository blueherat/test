"""Independent FID, all pixels and frozen-1K formula audit for two legacy5K arms."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.audit_raev2_prefix_ratio64k_quality import independent_fid
from experiments.audit_raev2_screen_fid_20260907 import load_moments, resolve
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    folder = DATA/'final8_legacy_confirm5k'
    execution = json.loads((folder/'execution.json').read_text())
    assert execution['complete']
    for original, digest in execution['sources'].items():
        assert sha(folder/'frozen_source'/Path(original).name) == digest
    previous = json.loads((ROOT/'experiments/results/raev2_guidance_20260907/semantic_confirm5k.json').read_text())
    assert previous['complete']
    controls = [row for row in previous['rows'] if row['mode'] in ['official', 'piecewise']]
    assert len(controls) == 2
    for row in controls:
        assert sha(Path(row['sample_path'])) == row['sample_sha256']
        feature_path = DATA/'weak_confirm5k/official_feature_cache'/f"{row['mode']}-{row['sample_sha256'][:16]}-inception.features.pt"
        assert sha(feature_path) == row['feature_sha256']
    baseline = next(row for row in controls if row['mode'] == 'official')
    reference_path = Path(resolve('imagenet_256_fid_stats'))
    assert sha(reference_path) == previous['reference_sha256']
    mean_ref, covariance_ref = load_moments(str(reference_path))
    values, vectors = np.linalg.eigh(covariance_ref)
    assert values.min() > -1e-9
    root_ref = (vectors*np.sqrt(np.maximum(values, 0)))@vectors.T
    metrics = json.loads((folder/'metrics.json').read_text())
    assert {row['branch'] for row in metrics} == {'paired_ratio_calibrated', 'calibrated'}
    rows = []
    for metric in metrics:
        mode = metric['branch']
        summary = json.loads((folder/mode/'summary.json').read_text())
        assert summary['complete'] and summary['count'] == 5000 and summary['seed'] == 202609072 and summary['steps'] == 100
        assert summary['sample_model_calls'] == 500000
        assert summary['sample_ratio_backward_calls'] == (500000 if mode == 'paired_ratio_calibrated' else 0)
        assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert sha(Path(metric['sample_path'])) == metric['sample_sha256'] == summary['sample_sha256']
        with np.load(metric['sample_path']) as data:
            merged = data['arr_0']
        assert merged.shape == (5000, 256, 256, 3) and merged.dtype == np.uint8
        records, ids = [], []
        for rank in range(4):
            shard = folder/f'shard{rank}'
            request = json.loads((shard/'request.json').read_text())
            control_request = json.loads((DATA/f'weak_confirm5k/shard{rank}/request.json').read_text())
            keys = ['checkpoint_sha256', 'config_sha256', 'decoder_sha256', 'stats_sha256',
                    'state_key', 'torch_version', 'batch', 'time_grid']
            assert {k: request[k] for k in keys} == {k: control_request[k] for k in keys}
            old_folder = 'paired_ratio_calibrated_screen1k' if mode == 'paired_ratio_calibrated' else 'calibrated_screen1k'
            old_request = json.loads((DATA/old_folder/f'shard{rank}/request.json').read_text())
            assert {k: request[k] for k in keys} == {k: old_request[k] for k in keys}
            parameter_keys = ['paired_ratio'] if mode == 'paired_ratio_calibrated' else ['variance_calibration_sha256', 'variance_formula']
            current_parameters = {k: request[k] for k in parameter_keys}
            previous_parameters = {k: old_request[k] for k in parameter_keys}
            if mode == 'paired_ratio_calibrated':
                # A later actual-native experiment added a descriptive source
                # label to both critic routes. It is not a sampling parameter.
                current_parameters['paired_ratio'] = dict(current_parameters['paired_ratio'])
                previous_parameters['paired_ratio'] = dict(previous_parameters['paired_ratio'])
                assert current_parameters['paired_ratio'].pop('source_law') == 'renoised_endpoints'
                assert previous_parameters['paired_ratio'].pop('source_law', 'renoised_endpoints') == 'renoised_endpoints'
            assert current_parameters == previous_parameters
            part = json.loads((shard/mode/'summary.json').read_text())
            assert part['complete'] and sha(shard/mode/'samples.npz') == part['sample_sha256']
            records.extend(part['initial_noise'])
            with np.load(shard/mode/'samples.npz') as data:
                ids.extend(data['ids'].tolist())
                assert np.array_equal(merged[data['ids']], data['arr_0'])
        assert sorted(ids) == list(range(5000))
        records.sort(key=lambda row: row['batch'])
        assert [row['batch'] for row in records] == list(range(625))
        paired = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        assert paired == summary['paired_noise_labels_sha256'] == baseline['paired_noise_labels_sha256']
        feature_path = folder/'official_feature_cache'/f"{mode}-{metric['sample_sha256'][:16]}-inception.features.pt"
        features = torch.load(feature_path, map_location='cpu', weights_only=True).double().numpy()
        assert features.shape == (5000, 2048) and np.isfinite(features).all()
        reconstruction = independent_fid(features, mean_ref, covariance_ref, root_ref)
        assert abs(reconstruction['fid']-metric['fid']) < 1e-3
        cost = summary['trajectory_seconds_sum']+summary['decode_seconds_sum']
        rows.append({**metric, **summary, 'independent_fid': reconstruction,
                     'inference_gpu_seconds': cost, 'inference_cost_ratio': cost/baseline['inference_gpu_seconds'],
                     'improvement_vs_official_percent': 100*(1-metric['fid']/baseline['fid']),
                     'improvement_vs_best_existing_control_percent': 100*(1-metric['fid']/min(row['fid'] for row in controls)),
                     'feature_sha256': sha(feature_path), 'unchanged_1k_formula_and_parameters_verified': True})
        print(mode, metric['fid'], rows[-1]['improvement_vs_official_percent'], flush=True)
    result = {'complete': True, 'goal_achieved': False, 'rows': rows, 'controls': controls,
              'paired_inputs_and_all_merged_pixels_verified': True,
              'source_snapshots_verified': True, 'execution_sha256': sha(folder/'execution.json'),
              'legacy_request_compatibility': 'Normalize only optional source_law=renoised_endpoints annotation; all checkpoint, alpha, plan and other parameter fields remain exact',
              'audit_source_sha256': sha(Path(__file__).resolve()),
              'reference_sha256': sha(reference_path),
              'preparation_cost_note': 'Reuse fixed prior critic and two probability/variance calibrations; see original fit and calibration records, no retraining here',
              'quality_success_requires_review_and_appropriate_cost_comparison': True}
    (ROOT/'experiments/results/raev2_guidance_20260907/final8_legacy5k_audit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
