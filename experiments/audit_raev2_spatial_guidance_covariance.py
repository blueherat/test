"""Independently audit completed spatial-covariance 5K samples against both controls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_raev2_guidance_quadrature import audit_pixels, read
from experiments.audit_raev2_prefix_ratio64k_quality import independent_fid
from experiments.audit_raev2_screen_fid_20260907 import load_moments, resolve
from experiments.run_raev2_guidance_quadrature import sha

DATA = Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--official-folder', type=Path, default=DATA/'weak_confirm5k')
    p.add_argument('--global-folder', type=Path, default=DATA/'directional_variance_confirm5k')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    began = time.perf_counter()
    run = a.folder.resolve()
    execution = read(run/'execution.json')
    assert execution['complete'] and execution['sampling_complete']
    assert execution['args']['samples'] == 5000
    for source, expected in execution['sources'].items():
        assert sha(run/'frozen_source'/Path(source).relative_to(ROOT)) == expected
    for index, (source, expected) in enumerate(execution['calibration_sources'].items()):
        assert sha(run/'frozen_calibration'/f'{index}_{Path(source).name}') == expected
    ref = Path(resolve('imagenet_256_fid_stats'))
    # Fixed reference identity used throughout the archived ImageNet256 study.
    assert sha(ref) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    mean_ref, cov_ref = load_moments(str(ref))
    eig, vec = np.linalg.eigh(cov_ref)
    assert eig.min() > -1e-9
    root_ref = (vec*np.sqrt(np.maximum(eig, 0)))@vec.T
    def verify_metric(folder, metric):
        assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metric['fid_reference'] == 'imagenet_256_fid_stats'
        path = folder/'official_feature_cache'/f"{metric['branch']}-{metric['sample_sha256'][:16]}-inception.features.pt"
        features = torch.load(path, map_location='cpu', weights_only=True).double().numpy()
        assert features.shape == (5000, 2048) and np.isfinite(features).all()
        result = independent_fid(features, mean_ref, cov_ref, root_ref)
        assert abs(result['fid']-metric['fid']) < 1e-5
        return {'independent_fid': result, 'feature_sha256': sha(path)}
    controls, common = {}, None
    for name, folder, branch in [('official', a.official_folder.resolve(), 'official'),
                                 ('global', a.global_folder.resolve(), 'directional_variance')]:
        metric = next(r for r in read(folder/'metrics.json') if r['branch'] == branch)
        summary = read(folder/branch/'summary.json')
        assert summary['sample_sha256'] == metric['sample_sha256']
        records = audit_pixels(folder, branch, 5000, summary, 4)
        assert common is None or common == records
        common = records
        request = read(folder/'shard0/request.json')
        assert request['seed'] == execution['args']['seed']
        controls[name] = {**metric, **verify_metric(folder, metric),
            'inference_seconds': summary['trajectory_seconds_sum']+summary['decode_seconds_sum'],
            'request': request}
    rows = []
    for metric in read(run/'metrics.json'):
        mode = metric['branch']
        summary = read(run/mode/'summary.json')
        assert summary['sample_model_calls'] == 500000 and summary['samples'] == 5000
        assert metric['sample_sha256'] == summary['sample_sha256']
        assert audit_pixels(run, mode, 5000, summary, len(execution['args']['gpus'])) == common
        for rank in range(len(execution['args']['gpus'])):
            request = read(run/f'shard{rank}/request.json')
            for name in controls:
                for key in ['checkpoint_sha256','config_sha256','decoder_sha256','stats_sha256',
                            'state_key','time_grid','batch','torch_version','seed']:
                    assert request[key] == controls[name]['request'][key], (name, key)
            assert request['steps'] == 100
            assert request['precision'] == 'native BF16 heads/mix, FP32 Euler and colored noise, FP64 covariance algebra, TF32 on'
            assert request['covariance_calibration_sha256'] == sha(run/'frozen_calibration/0_calibration.json')
            assert request['covariance_sha256'] == sha(run/'frozen_calibration/1_covariance.npz')
            assert request['spherical_head_sha256'] == sha(run/'frozen_calibration/2_head.pt')
            assert request['global_calibration_sha256'] == sha(run/'frozen_calibration/3_calibration.json')
            assert request['scalar_calibration_sha256'] == sha(run/'frozen_calibration/4_calibration.json')
        cost = summary['trajectory_seconds_sum']+summary['decode_seconds_sum']
        row = {**metric, **verify_metric(run, metric), 'mode':mode, 'samples':5000,
            'relative_improvement_percent':100*(1-metric['fid']/controls['official']['fid']),
            'relative_improvement_over_global_percent':100*(1-metric['fid']/controls['global']['fid']),
            'discovery_point_meets_3_percent':metric['fid'] <= .97*controls['official']['fid'],
            'inference_seconds':cost, 'cost_ratio_to_official':cost/controls['official']['inference_seconds'],
            'cost_ratio_to_global':cost/controls['global']['inference_seconds'],
            'sample_model_calls':summary['sample_model_calls'],
            'inactive_image_queries':summary['inactive_image_queries']}
        rows.append(row)
        print(mode, row['fid'], row['relative_improvement_percent'], flush=True)
    for control in controls.values():
        del control['request']
    result = {'complete':True, 'goal_achieved':False,
        'note':'Discovery bank; assess the full/diagonal mechanism and independently confirm a promising candidate.',
        'controls':controls, 'rows':rows, 'paired_all_batch_noise_labels_verified':True,
        'all_merged_pixels_and_shard_hashes_verified':True, 'frozen_sources_and_calibrations_verified':True,
        'reference_sha256':sha(ref), 'execution_sha256':sha(run/'execution.json'),
        'audit_source_sha256':sha(Path(__file__).resolve()), 'elapsed_seconds':time.perf_counter()-began}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
