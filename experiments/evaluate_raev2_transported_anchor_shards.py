"""Merge only the new method, evaluate once, reuse the existing baseline."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from experiments.raev2_training_core import file_sha256

ROOT=Path('/home/zhoushunyu/data/eqvae/experiments/ig_transported_anchor_fourcard_20260908')
BASELINE=ROOT.parent/'ig_condition_carrier_20260908/quality'


def read(path):
    return json.loads(path.read_text())


def main():
    images=np.empty((1000,256,256,3),dtype=np.uint8)
    seen=set()
    seconds=[]
    shard_seconds=[]
    baseline=read(BASELINE/'native_ig/summary.json')
    for rank in range(4):
        dest=ROOT/f'rank{rank}'
        s=read(dest/'summary.json'); req=read(dest/'request.json')
        assert s['complete'] and s['rank']==rank and req['rank']==rank
        assert s['request_sha256']==file_sha256(dest/'request.json')
        for filename,value in req['sources'].items():
            assert file_sha256(Path(filename))==value
        for key in ['noise_sha256','label_sha256']:
            assert s[key]==baseline[key]
        shard_seconds.append(s['seconds'])
        for record in s['files']:
            start=record['start']
            assert start not in seen and (start//4)%4==rank
            seen.add(start)
            path=dest/record['file']
            assert file_sha256(path)==record['sha256']
            with np.load(path) as batch:
                assert str(batch['request_sha256'])==s['request_sha256']
                np.testing.assert_array_equal(batch['labels'],np.arange(start,start+4))
                assert batch['arr_0'].shape==(4,256,256,3) and batch['arr_0'].dtype==np.uint8
                assert int(batch['full_batch_calls'])==100 and int(batch['prefix_batch_calls'])==98
                images[start:start+4]=batch['arr_0']
                seconds.append(float(batch['seconds']))
    assert seen==set(range(0,1000,4))
    np.savez(ROOT/'samples.npz',arr_0=images)
    command=[sys.executable,'experiments/evaluate_raev2_official_samples.py',
             '--branch','transported_anchor='+str(ROOT/'samples.npz'),'--output',str(ROOT/'fid.csv'),
             '--batch-size','32','--device','cuda','--feature-cache-dir',str(ROOT/'features')]
    with (ROOT/'evaluation.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    metric,=read(ROOT/'fid.json')
    assert metric['sample_sha256']==file_sha256(ROOT/'samples.npz')
    existing=next(row for row in read(BASELINE/'fid.json') if row['branch']=='native_ig')
    assert existing['sample_sha256']==file_sha256(BASELINE/'native_ig/samples.npz')
    assert existing['fid_reference']==metric['fid_reference']
    assert existing['evaluator_commit']==metric['evaluator_commit']
    result=dict(complete=True,method=metric,reused_baseline=existing,
                relative_fid_improvement=1-metric['fid']/existing['fid'],
                samples=1000,coverage_verified=True,smoke_pixels_verified=False,
                full_sample_calls=100000,prefix_sample_calls=98000,
                sum_batch_gpu_seconds=sum(seconds),shard_seconds=shard_seconds,
                note='Exploratory paired 1K. Baseline was not regenerated or reevaluated.')
    (ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    portable=Path('experiments/results/terminal_defect_20260908/ig_transported_anchor_quality.json')
    with portable.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
