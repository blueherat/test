"""Independent artifact, pairing, cost and feature-Gram FID verification."""
import json
from pathlib import Path
import numpy as np
import torch

from experiments.raev2_training_core import file_sha256

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_condition_carrier_20260908')
ARMS = ['native_ig', 'output_control', 'condition_carrier']


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(4)
    completion = read(ROOT/'complete.json')
    assert completion['complete']
    request = read(ROOT/'quality/request.json')
    smoke_request = read(ROOT/'smoke/request.json')
    assert request['sources'] == smoke_request['sources']
    assert request['samples'] == 1000 and request['batch'] == 4 and request['steps'] == 100
    assert request['seed'] == 202609413 and request['gamma'] == .78 and request['continued_guidance']
    for path,sha in request['sources'].items():
        assert file_sha256(Path(path)) == sha, path
    parallel = completion.get('execution') == 'parallel_candidate'
    if parallel:
        parallel_request = read(ROOT/'parallel/request.json')
        for key in ['samples','seed','batch','steps','gamma','adapter_sha256','checkpoint_sha256','precision','continued_guidance']:
            assert parallel_request[key] == request[key], key
        for path,sha in parallel_request['sources'].items():
            assert file_sha256(Path(path)) == sha, path
        original_source = Path('experiments/sample_raev2_condition_carrier.py').read_text()
        parallel_source = Path('experiments/sample_raev2_condition_carrier_parallel.py').read_text()
        assert parallel_source == original_source.replace("arms=['native_ig','output_control','condition_carrier']", "arms=['condition_carrier']")
    fit = read(ROOT/'fit_result.json')
    assert fit['complete'] and not fit['quality_evaluated']
    assert file_sha256(ROOT/'adapter.pt') == request['adapter_sha256'] == fit['adapter_sha256']
    assert file_sha256(ROOT/'request.json') == fit['request_sha256']
    assert read(ROOT/'production_readout_parity.json')['complete']
    old = ROOT.parent/'raev2_fsg_clock_transfer_20260908/quality/ordinary100'
    old_summary = read(old/'summary.json')
    np.testing.assert_array_equal(np.load(ROOT/'quality/native_ig/samples.npz')['arr_0'],np.load(old/'samples.npz')['arr_0'])
    reference = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
    assert file_sha256(reference) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    real = np.load(reference)
    mu,cov = real['mu'].astype(float),real['sigma'].astype(float)
    metrics = {x['branch']:x for x in read(ROOT/'quality/fid.json')}
    assert set(metrics) == set(ARMS)
    rows = []
    for arm in ARMS:
        dest = ROOT/'quality'/arm
        summary = read(dest/'summary.json')
        assert summary['complete'] and summary['samples'] == 1000
        request_path = ROOT/('parallel' if parallel and arm=='condition_carrier' else 'quality')/'request.json'
        assert summary['request_sha256'] == file_sha256(request_path)
        for key in ['noise_sha256','label_sha256']:
            assert summary[key] == old_summary[key]
        expected = dict(encoder_samples=100000,ddt_readout_samples=199000 if arm=='output_control' else 100000,
                        base_readout_samples=100000 if arm=='native_ig' else 0,
                        adapter_samples=0 if arm=='native_ig' else 99000)
        assert summary['calls'] == expected, (arm, summary['calls'])
        pixels = np.load(dest/'samples.npz')['arr_0']
        assert pixels.shape == (1000,256,256,3) and pixels.dtype == np.uint8
        np.testing.assert_array_equal(pixels[:8],np.load(ROOT/'smoke'/arm/'samples.npz')['arr_0'])
        metric = metrics[arm]
        sha = file_sha256(dest/'samples.npz')
        assert sha == metric['sample_sha256']
        assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metric['fid_reference'] == 'imagenet_256_fid_stats'
        feature_path, = (ROOT/'quality/features').glob(f'{arm}-{sha[:16]}*.features.pt')
        features = torch.load(feature_path,map_location='cpu',weights_only=True).numpy().astype(float)
        assert features.shape == (1000,2048) and np.isfinite(features).all()
        mean = features.mean(0)
        centered = features-mean
        gram = centered@cov@centered.T/999
        ev = np.linalg.eigvalsh((gram+gram.T)/2)
        assert ev.min() > -1e-7
        recomputed = float(np.square(mean-mu).sum()+np.square(centered).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
        assert abs(recomputed-metric['fid']) < 2e-4
        rows.append(dict(arm=arm,fid=metric['fid'],independent_fid=recomputed,inception_score=metric['inception_score'],
                         seconds=summary['seconds'],calls=summary['calls'],pixel_sha256=sha,feature_sha256=file_sha256(feature_path)))
    result = dict(complete=True,rows=rows,fit_and_validation_seconds=fit['seconds'],
                  adapter_parameters=fit['adapter_parameters'],scope='Exploratory paired 1K; not independent quality confirmation or 5K evidence.')
    (ROOT/'independent_quality_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
