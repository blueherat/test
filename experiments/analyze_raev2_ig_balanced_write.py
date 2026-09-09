"""Independent fixed1K FID and identity audit for IG initial writes."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from experiments.raev2_training_core import file_sha256

def main():
    root=Path(__file__).resolve().parents[1];data=Path('/home/zhoushunyu/data/eqvae/experiments')
    reference=Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')
    assert file_sha256(reference)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    ref=np.load(reference);mu=ref['mu'].astype(float);cov=ref['sigma'].astype(float)
    paths={name:data/'raev2_ig_balanced_write_20260908'/name/'quality' for name in ['raw','balanced']}
    paths.update(full=data/'raev2_pfr_working_point_20260908/full/quality',native_ig=data/'raev2_fsg_clock_transfer_20260908/quality/ordinary100')
    baseline=json.loads((paths['full']/'summary.json').read_text());rows=[]
    for name,q in paths.items():
        m=json.loads((q/'summary.json').read_text());metric=json.loads((q/'fid.json').read_text())[0]
        assert m['complete'] and m['samples']==1000 and m['batch_size']==4 and m['seed']==202609413
        for key in ['noise_sha256','label_sha256','checkpoint_sha256']:assert m[key]==baseline[key]
        if name in ['raw','balanced']:
            parent=q.parent;done=json.loads((parent/'complete.json').read_text());assert all(done.values())
            assert m['H']==.125 and m['K']==2 and m['ig_scale']==1.78 and not m['zero_write']
            assert m['post_write_null_sample_calls']==0 and m['full_calls']==(104 if name=='raw' else 106)*250
            assert m['sources']==json.loads((parent/'smoke/summary.json').read_text())['sources']
            for path,digest in m['sources'].items():assert file_sha256(Path(path))==digest
            np.testing.assert_array_equal(np.load(parent/'zero/samples.npz')['arr_0'],np.load(paths['full'].parent/'smoke/samples.npz')['arr_0'])
            np.testing.assert_array_equal(np.load(q/'samples.npz')['arr_0'][:8],np.load(parent/'smoke/samples.npz')['arr_0'])
        assert metric['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metric['fid_reference']=='imagenet_256_fid_stats'
        assert file_sha256(q/'samples.npz')==metric['sample_sha256']
        path,=(q/'features').glob('*.features.pt');x=torch.load(path,map_location='cpu',weights_only=True).numpy().astype(float)
        assert x.shape==(1000,2048) and np.isfinite(x).all()
        mean=x.mean(0);x-=mean;gram=x@cov@x.T/999;ev=np.linalg.eigvalsh((gram+gram.T)/2);assert ev.min()>-1e-7
        mt=float(np.square(mean-mu).sum());ct=float(np.square(x).sum()/999+np.trace(cov)-2*np.sqrt(np.maximum(ev,0)).sum())
        assert abs(mt+ct-metric['fid'])<2e-4
        rows.append(dict(arm=name,fid=metric['fid'],independent_fid=mt+ct,mean_term=mt,covariance_term=ct,
                         inception_score=metric['inception_score'],seconds=m['seconds'],full_calls_per_image=m['full_calls']/250,
                         pixel_sha256=metric['sample_sha256'],feature_sha256=file_sha256(path)))
    out=root/'experiments/results/terminal_defect_20260908/raev2_ig_balanced_write.csv'
    with out.open('x') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
