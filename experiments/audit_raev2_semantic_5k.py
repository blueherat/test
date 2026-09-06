"""Archive the fixed semantic confirmation, independently reconstructing FID.

Uses the symmetric reference-whitened covariance product, unlike the official
non-symmetric scipy sqrtm implementation. It never changes samples or guidance.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.audit_raev2_screen_fid_20260907 import load_moments, resolve
from experiments.summarize_raev2_guidance_20260907 import DATA, ROOT, sha


def main():
    reference = Path(resolve('imagenet_256_fid_stats'))
    assert sha(reference) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    mr, cr = load_moments(str(reference))
    values, vectors = np.linalg.eigh(cr)
    assert values.min() > -1e-9
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    rows, identities = [], []
    for study, modes in [('weak_confirm5k', ['official', 'piecewise']),
                         ('semantic_confirm5k', ['semantic_orthogonal'])]:
        folder = DATA / study
        execution = json.loads((folder / 'execution.json').read_text())
        assert execution['complete']
        for original, digest in execution['sources'].items():
            assert sha(folder / 'frozen_source' / Path(original).name) == digest
        metrics = {r['branch']: r for r in json.loads((folder / 'metrics.json').read_text())}
        for mode in modes:
            metric = metrics[mode]
            summary = json.loads((folder / mode / 'summary.json').read_text())
            assert summary['complete'] and summary['count'] == 5000 and summary['seed'] == 202609072
            assert summary['steps'] == 100
            assert metric['sample_sha256'] == summary['sample_sha256'] == sha(Path(metric['sample_path']))
            assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
            with np.load(metric['sample_path']) as data:
                merged = data['arr_0']
            assert merged.shape == (5000, 256, 256, 3) and merged.dtype == np.uint8
            records, ids = [], []
            for rank in range(4):
                shard = folder / f'shard{rank}'
                request = json.loads((shard / 'request.json').read_text())
                identity = {k: request[k] for k in ('checkpoint_sha256', 'config_sha256',
                           'decoder_sha256', 'stats_sha256', 'state_key', 'torch_version', 'batch', 'time_grid')}
                identities.append(identity)
                part = json.loads((shard / mode / 'summary.json').read_text())
                assert part['complete'] and sha(shard / mode / 'samples.npz') == part['sample_sha256']
                records.extend(part['initial_noise'])
                with np.load(shard / mode / 'samples.npz') as data:
                    index = data['ids']
                    assert np.array_equal(merged[index], data['arr_0'])
                    ids.extend(index.tolist())
            assert sorted(ids) == list(range(5000))
            records.sort(key=lambda r: r['batch'])
            assert [r['batch'] for r in records] == list(range(625))
            paired = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            assert paired == summary['paired_noise_labels_sha256'] == 'b59864ce96fcfb63735061f00ecb903b07ad894ffb504a73918d7b704db86cc8'
            features = folder / 'official_feature_cache' / f"{mode}-{metric['sample_sha256'][:16]}-inception.features.pt"
            x = torch.load(features, map_location='cpu', weights_only=True).double().numpy()
            assert x.shape == (5000, 2048) and np.isfinite(x).all()
            mean = x.mean(0)
            centered = x - mean
            covariance = centered.T @ centered / 4999
            cross = root @ covariance @ root
            spectrum = np.linalg.eigvalsh((cross + cross.T) / 2)
            assert spectrum.min() > -1e-8
            mean_term = float(np.sum((mean - mr)**2))
            covariance_term = float(np.trace(covariance) + np.trace(cr) - 2 * np.sqrt(np.maximum(spectrum, 0)).sum())
            fid = mean_term + covariance_term
            assert abs(fid - metric['fid']) < 1e-3
            row = {**metric, **summary, 'study': study, 'independent_fid': fid,
                   'mean_term': mean_term, 'covariance_term': covariance_term,
                   'sample_covariance_trace': float(np.trace(covariance)),
                   'feature_sha256': sha(features), 'execution_sha256': sha(folder / 'execution.json'),
                   'inference_gpu_seconds': summary['trajectory_seconds_sum'] + summary['decode_seconds_sum']}
            rows.append(row)
            print(mode, fid, flush=True)
    assert all(r == identities[0] for r in identities)
    baseline, candidate = rows[0], rows[-1]
    for row in rows:
        row['relative_fid_improvement_percent'] = 100 * (1 - row['fid'] / baseline['fid'])
        row['inference_cost_ratio'] = row['inference_gpu_seconds'] / baseline['inference_gpu_seconds']
    queue = DATA / 'semantic_cost_continuation/state.json'
    state = json.loads(queue.read_text())
    assert state['complete']
    quality_pass = candidate['fid'] <= .97 * min(r['fid'] for r in rows[:-1])
    record = {'complete': True, 'goal_achieved': False, 'three_percent_quality_passed': quality_pass,
              'rows': rows, 'identity': identities[0], 'reference_sha256': sha(reference),
              'source_snapshots_verified': True, 'all_merged_pixels_verified_against_shards': True,
              'formula': 'FP64 eigvalsh(sqrt(C_ref) C_sample sqrt(C_ref)); Bessel covariance',
              'queue_state': state, 'queue_state_sha256': sha(queue),
              'max_fid_error': max(abs(r['fid'] - r['independent_fid']) for r in rows)}
    out = ROOT / 'experiments/results/raev2_guidance_20260907/semantic_confirm5k.json'
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(candidate['relative_fid_improvement_percent'], candidate['inference_cost_ratio'], out, flush=True)


if __name__ == '__main__':
    main()
