"""Evaluate every fixed prototype arm and independently check feature FID."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from experiments.raev2_training_core import file_sha256

ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/ig_future_prototype_20260908')
REFERENCE = ROOT.parent/'ig_condition_carrier_20260908/quality/native_ig'
ARMS = ['full_prototype', 'ig_prototype', 'balanced_ig_prototype', 'ordinary160']
STATS = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')


def read(path):
    return json.loads(path.read_text())


def main():
    torch.set_num_threads(4)
    baseline = read(REFERENCE/'summary.json')
    archives = {'ordinary100':REFERENCE/'samples.npz'}
    summaries = {}
    for arm in ARMS:
        dest = ROOT/arm
        summary,request,checks = [read(dest/name) for name in ['summary.json','request.json','checks.json']]
        assert summary['complete'] and summary['samples'] == 1000
        assert summary['request_sha256'] == file_sha256(dest/'request.json')
        assert summary['samples_sha256'] == file_sha256(dest/'samples.npz')
        assert request['arm'] == arm and request['seed'] == 202609413
        assert request['events'] == [20,40,60,80] and request['horizon_grid_steps'] == 4
        assert request['steps'] == (160 if arm == 'ordinary160' else 100)
        assert request['gamma'] == .78 and request['continued_native_ig'] is True
        assert request['checkpoint_sha256'] == '723c56d7fa77ace9613909f7e38cb2386b898608218dc9b52649bb373d513c9a'
        for path,value in request['sources'].items():
            assert file_sha256(Path(path)) == value
        assert file_sha256(Path('docs/IG_FUTURE_PROTOTYPE_METHOD_20260908_ZH.md')) == request['protocol_sha256']
        assert checks['prefix_bitwise'] and checks['native_eight_pixel_parity']
        for key in ['noise_sha256','label_sha256']:
            assert summary[key] == baseline[key]
        full = 160 if arm == 'ordinary160' else 132
        prefix = 96 if arm == 'balanced_ig_prototype' else (0 if arm == 'ordinary160' else 32)
        assert summary['calls'] == dict(full_batch_calls=250*full,prefix_batch_calls=250*prefix)
        pixels = np.load(dest/'samples.npz')['arr_0']
        assert pixels.shape == (1000,256,256,3) and pixels.dtype == np.uint8
        np.testing.assert_array_equal(pixels[:8],np.load(dest/'smoke.npz')['arr_0'])
        summaries[arm] = summary
        archives[arm] = dest/'samples.npz'
    command = [sys.executable,'experiments/evaluate_raev2_official_samples.py',
               '--output',str(ROOT/'fid.csv'),'--batch-size','32','--device','cuda',
               '--feature-cache-dir',str(ROOT/'features')]
    for arm,path in archives.items():
        command += ['--branch',arm+'='+str(path)]
    with (ROOT/'evaluation.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert file_sha256(STATS) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    real = np.load(STATS)
    mu,cov = real['mu'].astype(float),real['sigma'].astype(float)
    metrics = {row['branch']:row for row in read(ROOT/'fid.json')}
    assert set(metrics) == set(archives)
    rows = []
    for arm,path in archives.items():
        metric = metrics[arm]
        digest = file_sha256(path)
        assert digest == metric['sample_sha256']
        assert metric['evaluator_commit'] == '19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metric['fid_reference'] == 'imagenet_256_fid_stats'
        feature_path, = (ROOT/'features').glob(f'{arm}-{digest[:16]}*.features.pt')
        features = torch.load(feature_path,map_location='cpu',weights_only=True).numpy().astype(float)
        assert features.shape == (1000,2048) and np.isfinite(features).all()
        mean = features.mean(0)
        centered = features-mean
        gram = centered@cov@centered.T/999
        eig = np.linalg.eigvalsh((gram+gram.T)/2)
        assert eig.min() > -1e-7
        fid = float(np.square(mean-mu).sum()+np.square(centered).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(eig,0)).sum())
        assert abs(fid-metric['fid']) < 2e-4
        rows.append(dict(arm=arm,fid=metric['fid'],independent_fid=fid,inception_score=metric['inception_score'],
                         seconds=(baseline if arm == 'ordinary100' else summaries[arm])['seconds'],
                         calls=(baseline if arm == 'ordinary100' else summaries[arm])['calls'],
                         samples_sha256=digest,features_sha256=file_sha256(feature_path)))
    result = dict(complete=True,rows=rows,scope='Fixed exploratory paired 1K; not independent method confirmation.',
                  source_sha256=file_sha256(Path(__file__)),
                  paired_inputs_verified=True,all_smoke_parity_verified=True,all_cost_counts_verified=True)
    (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    portable = Path('experiments/results/terminal_defect_20260908/ig_future_prototype_quality.json')
    with portable.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
