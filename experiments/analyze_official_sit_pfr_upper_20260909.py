"""Verify six new scale points and combine the immutable eighteen-point evidence."""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.analyze_official_sit_pfr_symmetric_20260909 import sha,array_sha,fid_components,REF
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_upper_20260909')
PREVIOUS=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_symmetric_20260909')
OUT=ROOT/'docs/data/guidance_goal_20260909'


def main():
    req=json.loads((DATA/'request.json').read_text())
    assert json.loads((DATA/'status.json').read_text())['phase']=='complete'
    for path,digest in req['sources'].items(): assert sha(Path(path))==digest,path
    assert sha(PREVIOUS/'request.json')==req['previous_request_sha256']
    assert sha(OUT/'official_sit_pfr_symmetric.csv')==req['previous_csv_sha256']
    assert sha(OUT/'official_sit_pfr_symmetric_audit.json')==req['previous_audit_sha256']
    old_audit=json.loads((OUT/'official_sit_pfr_symmetric_audit.json').read_text());assert old_audit['passed']
    assert sha(DATA/'inputs.npz')==req['input_file_sha256']
    with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
    assert array_sha(noise)==req['noise_sha256'] and array_sha(labels)==req['label_sha256']
    parity=json.loads((DATA/'parity_passed.json').read_text())
    assert parity['passed'] and parity['unique_inputs']==16 and parity['image_outputs']==48
    assert sorted(i for p in parity['records'] for i in p['indices'])==list(range(16))
    assert all(all(r['passed'] for r in p['records'].values()) for p in parity['records'])
    assert sha(REF)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    with np.load(REF) as ref: mu,cov=ref['mu'].astype(float),ref['sigma'].astype(float)
    with (OUT/'official_sit_pfr_symmetric.csv').open() as f: rows=list(csv.DictReader(f))
    ints=('samples','full_per_image','prefix8_per_image','block_evaluations_per_image')
    floats=('scale','fid','independent_fid','mean_term','covariance_term','inception_score','inference_gpu_seconds')
    for row in rows:
        for key in ints: row[key]=int(row[key])
        for key in floats: row[key]=float(row[key])
        row['reused']=True
        assert row['noise_sha256']==req['noise_sha256'] and row['label_sha256']==req['label_sha256']
    assert len(rows)==12
    new_hashes={};diagnostics={}
    for arm,(method,scale) in req['arms'].items():
        directory=DATA/arm;result=json.loads((directory/'result.json').read_text())
        assert result['method']==method and result['scale']==scale
        with np.load(directory/'samples.npz') as f: pixels=f['arr_0']
        assert pixels.shape==(1000,256,256,3) and pixels.dtype==np.uint8
        covered=np.zeros(1000,dtype=int);summaries=[]
        for rank in range(4):
            s=json.loads((directory/f'rank{rank}.json').read_text());summaries.append(s)
            assert s['complete'] and s['method']==method and s['scale']==scale and s['batch_size']==4
            assert s['request_sha256']==sha(DATA/'request.json')
            assert s['pixel_file_sha256']==sha(directory/f'rank{rank}.npz')
            with np.load(directory/f'rank{rank}.npz') as shard:
                idx=shard['indices']
                assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                assert np.array_equal(pixels[idx],shard['arr_0'])
                assert array_sha(noise[idx])==s['noise_sha256'] and array_sha(labels[idx])==s['label_sha256']
                covered[idx]+=1
        assert np.all(covered==1) and sum(s['samples'] for s in summaries)==1000
        full,prefix=(115,0) if method=='ordinary' else (100,50)
        assert sum(s['full_sample_calls'] for s in summaries)==full*1000
        assert sum(s['prefix_sample_calls'] for s in summaries)==prefix*1000
        if prefix:
            assert sha(directory/'diagnostics.npz')==result['diagnostic_sha256']
            with np.load(directory/'diagnostics.npz') as d:
                stats=d['diagnostics'];assert stats.shape==(1000,5,4) and np.isfinite(stats).all()
                assert np.array_equal(d['labels'],labels) and np.isfinite(d['latents']).all()
            diagnostics[arm]=dict(steps=[0,10,25,40,49],fields=['alpha','query_shift_rms','correction_rms','ordinary_drift_rms'],
                mean=stats.mean(0,dtype=np.float64).tolist(),median=np.median(stats,axis=0).tolist(),
                alpha_zero_fraction=(stats[:,:,0]==0).mean(0).tolist())
        metrics=json.loads((directory/'fid.json').read_text())[0]
        assert metrics['sample_sha256']==sha(directory/'samples.npz')
        assert metrics['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metrics['fid_reference']=='imagenet_256_fid_stats'
        feature,=(directory/'features').glob('*.features.pt')
        x=torch.load(feature,map_location='cpu',weights_only=True).numpy().astype(float)
        assert x.shape==(1000,2048) and np.isfinite(x).all()
        mean_term,cov_term=fid_components(x,mu,cov)
        assert abs(mean_term+cov_term-metrics['fid'])<2e-4
        new_hashes[arm]=sha(feature)
        rows.append(dict(arm=arm,method=method,scale=scale,samples=1000,fid=metrics['fid'],independent_fid=mean_term+cov_term,
            mean_term=mean_term,covariance_term=cov_term,inception_score=metrics['inception_score'],full_per_image=full,
            prefix8_per_image=prefix,block_evaluations_per_image=full*28+prefix*8,
            inference_gpu_seconds=result['gpu_inference_seconds_sum'],reused=False,
            noise_sha256=req['noise_sha256'],label_sha256=req['label_sha256'],pixel_sha256=metrics['sample_sha256']))
        print(json.dumps({k:rows[-1][k] for k in ('arm','fid','mean_term','covariance_term')}),flush=True)
    methods=('ordinary','time_only','projected')
    for method in methods: assert sorted(r['scale'] for r in rows if r['method']==method)==[1.35,1.4,1.5,1.75,2.,2.5]
    rows.sort(key=lambda r:(methods.index(r['method']),r['scale']))
    best={m:min((r for r in rows if r['method']==m),key=lambda r:r['fid']) for m in methods}
    audit=dict(passed=True,research_goal_achieved=False,sampling_bank_role='reused_1k_discovery',
        request_sha256=sha(DATA/'request.json'),source_sha256=sha(Path(__file__)),fid_helper_sha256=sha(ROOT/'experiments/analyze_official_sit_pfr_symmetric_20260909.py'),
        previous_csv_sha256=req['previous_csv_sha256'],previous_audit_sha256=req['previous_audit_sha256'],
        new_feature_sha256=new_hashes,new_point_count=6,previous_verified_point_count=12,
        maximum_fid_reconstruction_error=max(abs(r['fid']-r['independent_fid']) for r in rows),
        selected_by_method={m:{k:r[k] for k in ('arm','scale','fid','inception_score')} for m,r in best.items()},
        grid_minimum_is_interior={m:r['scale'] not in (1.35,2.5) for m,r in best.items()},
        best_pfr_minus_best_ordinary={m:best[m]['fid']-best['ordinary']['fid'] for m in ('time_only','projected')},
        limitation='One reused discovery bank; selection is not independent confirmation or a new mechanism. Endpoint minima leave tuning incomplete.')
    outputs=['official_sit_pfr_extended.csv','official_sit_pfr_extended_audit.json','official_sit_pfr_upper_diagnostics.json']
    assert not any((OUT/name).exists() for name in outputs)
    with (OUT/outputs[0]).open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/outputs[1]).write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/outputs[2]).write_text(json.dumps(diagnostics,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    names={'ordinary':'IG (115 full)','time_only':'PFR time only (100 full + 50 prefix)','projected':'PFR projected (100 full + 50 prefix)'}
    for method in methods:
        rr=[r for r in rows if r['method']==method]
        for ax,key in zip(axes,('fid','inception_score')):
            ax.plot([r['scale'] for r in rr],[r[key] for r in rr],marker='o',label=names[method])
    for ax,label in zip(axes,('FID-1K (lower is better)','Inception Score (higher is better)')):
        ax.set_xlabel('IG scale');ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Official SiT-XL/2 + joint IG: eighteen paired discovery points',fontsize=11)
    for ext in ('png','pdf'): fig.savefig(OUT/f'official_sit_pfr_extended.{ext}',dpi=200)
    plt.close(fig)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__': main()
