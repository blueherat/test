"""Native SDE sample/cost audit and independent FID arithmetic."""
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
from experiments.analyze_official_sit_pfr_symmetric_20260909 import fid_components,sha,REF

DATA=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_native_sde_20260909')
OUT=ROOT/'docs/data/guidance_goal_20260909'


def main():
    req=json.loads((DATA/'request.json').read_text())
    assert json.loads((DATA/'status.json').read_text())['phase']=='complete'
    for path,digest in req['sources'].items(): assert sha(Path(path))==digest,path
    assert sha(DATA/'inputs.npz')==req['input_file_sha256']
    for files in req['vae_files'].values():
        for path,digest in files.items(): assert sha(Path(path))==digest,path
    assert len(set(req['brownian_seeds']))==250
    parity=json.loads((DATA/'parity_passed.json').read_text())
    assert parity['passed'] and parity['images']==16 and all(p['passed'] for p in parity['records'])
    assert sorted(i for p in parity['records'] for i in p['indices'])==list(range(16))
    summaries=[json.loads((DATA/f'rank{r}.json').read_text()) for r in range(4)]
    assert sum(s['samples'] for s in summaries)==1000
    assert sum(s['full_batch_calls'] for s in summaries)==62500
    assert sum(s['full_sample_calls'] for s in summaries)==req['expected_calls']['full_per_image']*1000
    assert req['expected_calls']['full_per_image']==422
    with np.load(DATA/'inputs.npz') as bank: labels=bank['labels']
    with np.load(DATA/'latents.npz') as d: latents=d['latents'];assert np.array_equal(d['labels'],labels)
    assert latents.shape==(1000,4,32,32) and np.isfinite(latents).all()
    merged={}
    for name in ('ema','mse'):
        with np.load(DATA/name/'samples.npz') as d: merged[name]=d['arr_0']
        assert merged[name].shape==(1000,256,256,3) and merged[name].dtype==np.uint8
    coverage=np.zeros(1000,dtype=int)
    for rank,s in enumerate(summaries):
        assert s['request_sha256']==sha(DATA/'request.json')
        assert s['pixel_file_sha256']==sha(DATA/f'rank{rank}.npz')
        with np.load(DATA/f'rank{rank}.npz') as d:
            idx=d['indices'];assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
            assert np.array_equal(latents[idx],d['latents'])
            for name in merged: assert np.array_equal(merged[name][idx],d[name])
            coverage[idx]+=1
    assert np.all(coverage==1)
    assert sha(REF)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    with np.load(REF) as f: mu,cov=f['mu'].astype(float),f['sigma'].astype(float)
    rows=[];features={}
    for name in ('ema','mse'):
        directory=DATA/name
        metrics=json.loads((directory/'fid.json').read_text())[0]
        result=json.loads((directory/'result.json').read_text())
        assert sha(directory/'samples.npz')==metrics['sample_sha256']
        assert metrics['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b'
        assert metrics['fid_reference']=='imagenet_256_fid_stats'
        feature,=(directory/'features').glob('*.features.pt')
        x=torch.load(feature,map_location='cpu',weights_only=True).numpy().astype(float)
        assert x.shape==(1000,2048) and np.isfinite(x).all()
        mean_term,cov_term=fid_components(x,mu,cov)
        assert abs(mean_term+cov_term-metrics['fid'])<2e-4
        features[name]=sha(feature)
        rows.append(dict(arm='native_sde250_'+name,vae=name,samples=1000,fid=metrics['fid'],
            independent_fid=mean_term+cov_term,mean_term=mean_term,covariance_term=cov_term,
            inception_score=metrics['inception_score'],cfg_scale=1.35,ig_scale=1.4,cfg_high=.7,
            full_per_image=422,block_evaluations_per_image=422*28,
            trajectory_gpu_seconds=result['trajectory_gpu_seconds_sum'],decode_gpu_seconds=result['decode_gpu_seconds_sum'],
            inference_gpu_seconds=result['trajectory_gpu_seconds_sum']+result['decode_gpu_seconds_sum'],
            pixel_sha256=metrics['sample_sha256'],vae_state_sha256=req['vae_state_sha256'][name]))
    audit=dict(passed=True,research_goal_achieved=False,request_sha256=sha(DATA/'request.json'),
        analysis_source_sha256=sha(Path(__file__)),fid_helper_source_sha256=sha(ROOT/'experiments/analyze_official_sit_pfr_symmetric_20260909.py'),
        feature_sha256=features,maximum_fid_reconstruction_error=max(abs(r['fid']-r['independent_fid']) for r in rows),
        sampling_bank_role='reused_1k_discovery',shared_latent_sha256=sha(DATA/'latents.npz'),
        limitation='A controlled author-parameter baseline, not the paper FID-50K reproduction. No candidate comparison at equal SDE cost yet.')
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'official_sit_native_sde.csv').exists() and not (OUT/'official_sit_native_sde_audit.json').exists()
    with (OUT/'official_sit_native_sde.csv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'official_sit_native_sde_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(rows=rows,audit=audit),indent=2),flush=True)


if __name__=='__main__': main()
