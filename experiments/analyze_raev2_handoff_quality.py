"""Verify the frozen1K comparison and recompute FID from feature banks."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from experiments.raev2_training_core import file_sha256

def main():
    root=Path(__file__).resolve().parents[1];data=Path('/home/zhoushunyu/data/eqvae/experiments')
    parent=data/'raev2_handoff_quality_20260908';done=json.loads((parent/'complete.json').read_text())
    assert all(done.values())
    reference=Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
    assert file_sha256(reference)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    r=np.load(reference);mu=r['mu'].astype(float);cov=r['sigma'].astype(float)
    paths={'write_drop':parent/'quality','full':data/'raev2_pfr_working_point_20260908/full/quality',
           'native_ig':data/'raev2_fsg_clock_transfer_20260908/quality/ordinary100'}
    candidate=json.loads((paths['write_drop']/'summary.json').read_text())
    assert candidate['H']==.125 and candidate['K']==2 and candidate['gamma']==2.
    assert candidate['full_calls']==26500 and candidate['post_write_null_sample_calls']==100000
    assert candidate['precision']=='bf16 TF32' and not candidate['pilot']
    smoke=json.loads((parent/'smoke/summary.json').read_text())
    assert candidate['sources']==smoke['sources']
    for path,h in candidate['sources'].items():assert file_sha256(Path(path))==h
    np.testing.assert_array_equal(np.load(paths['write_drop']/'samples.npz')['arr_0'][:8],np.load(parent/'smoke/samples.npz')['arr_0'])
    for name,ref in [('native',data/'raev2_fsg_clock_transfer_20260908/smoke/ordinary100'),
                     ('full',data/'raev2_pfr_working_point_20260908/full/smoke')]:
        np.testing.assert_array_equal(np.load(parent/name/'samples.npz')['arr_0'],np.load(ref/'samples.npz')['arr_0'])
    pilot=np.load(data/'fsg_condition_handoff_20260908/calibrate_drop.npz')
    np.testing.assert_array_equal(np.load(parent/'pilot/samples.npz')['arr_0'],pilot['pixels'][:8])
    np.testing.assert_array_equal(np.load(parent/'pilot/written.npy'),pilot['written'][:8])
    rows=[]
    for name,q in paths.items():
        m=json.loads((q/'summary.json').read_text());metric=json.loads((q/'fid.json').read_text())[0]
        assert m['complete'] and m['samples']==1000 and m['batch_size']==4 and m['seed']==202609413
        for key in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[key]==candidate[key]
        assert file_sha256(q/'samples.npz')==metric['sample_sha256']
        assert metric['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metric['fid_reference']=='imagenet_256_fid_stats'
        f,=(q/'features').glob('*.features.pt')
        x=torch.load(f,map_location='cpu',weights_only=True).numpy().astype(float)
        assert x.shape==(1000,2048) and np.isfinite(x).all()
        mean=x.mean(0);x-=mean
        gram=x@cov@x.T/999;ev=np.linalg.eigvalsh((gram+gram.T)/2);assert ev.min()>-1e-7
        mean_term=float(np.square(mean-mu).sum())
        cov_term=float(np.square(x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
        assert abs(mean_term+cov_term-metric['fid'])<2e-4
        rows.append(dict(arm=name,fid=metric['fid'],independent_fid=mean_term+cov_term,mean_term=mean_term,
                         covariance_term=cov_term,inception_score=metric['inception_score'],seconds=m['seconds'],
                         full_calls_per_image=m['full_calls']/250,pixel_sha256=file_sha256(q/'samples.npz'),feature_sha256=file_sha256(f)))
    out=root/'experiments/results/terminal_defect_20260908/raev2_handoff_quality.csv'
    with out.open('x') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
