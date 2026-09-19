"""Verify SDE PFR/control images, coupling provenance, compute, and FID."""
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
DATA=Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_sde_pfr_control_20260909')
NATIVE=DATA.parent/'official_sit_native_sde_20260909'
OUT=ROOT/'docs/data/guidance_goal_20260909'


def main():
    req=json.loads((DATA/'request.json').read_text())
    assert json.loads((DATA/'status.json').read_text())['phase']=='complete'
    for p,d in req['sources'].items(): assert sha(Path(p))==d,p
    assert sha(NATIVE/'request.json')==req['native_request_sha256']
    assert sha(OUT/'official_sit_native_sde_audit.json')==req['native_audit_sha256']
    assert sha(OUT/'official_sit_native_sde.csv')==req['native_csv_sha256']
    assert sha(DATA/'inputs.npz')==req['input_file_sha256']
    for files in req['vae_files'].values():
        for p,d in files.items(): assert sha(Path(p))==d,p
    old=json.loads((NATIVE/'request.json').read_text())
    assert req['brownian_seeds']==old['brownian_seeds'] and len(set(req['bridge_seeds']))==250
    for rank in range(4):
        with np.load(NATIVE/f'rank{rank}.npz') as shard:
            assert [req['source_rng_end_hashes'][str(b)] for b in range(rank,250,4)]==list(shard['brownian_rng_end_sha256'])
    with np.load(DATA/'inputs.npz') as bank: noise,labels=bank['noise'],bank['labels']
    assert array_sha(noise)==req['noise_sha256'] and array_sha(labels)==req['label_sha256']
    parity=json.loads((DATA/'parity_passed.json').read_text())
    assert parity['passed'] and parity['unique_inputs']==16
    assert sorted(i for p in parity['records'] for i in p['indices'])==list(range(16))
    for p in parity['records']:
        assert p['passed'] and all(p[k] for k in ('native250_exact','copied250_exact','zero_response_exact',
                                                'copied281_exact','vae_pixels_exact','source_rng_exact'))
        assert len(p['prefix_checks'])==2
        assert sha(DATA/f"preflight_rank{p['rank']}.npz")==p['preview_sha256']
    assert sha(REF)=='925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    with np.load(REF) as f: mu,cov=f['mu'].astype(float),f['sigma'].astype(float)
    rows=[];feature_hashes={}
    with (OUT/'official_sit_native_sde.csv').open() as f:
        for old_row in csv.DictReader(f):
            row=dict(arm='ordinary250',vae=old_row['vae'],samples=1000,fid=float(old_row['fid']),
                independent_fid=float(old_row['independent_fid']),mean_term=float(old_row['mean_term']),
                covariance_term=float(old_row['covariance_term']),inception_score=float(old_row['inception_score']),
                full_per_image=422,prefix_per_image=0,block_evaluations_per_image=11816,
                trajectory_gpu_seconds=float(old_row['trajectory_gpu_seconds']),noise_coupling_gpu_seconds=0.,
                decode_gpu_seconds=float(old_row['decode_gpu_seconds']),inference_gpu_seconds=float(old_row['inference_gpu_seconds']),
                pixel_sha256=old_row['pixel_sha256'],reused=True)
            rows.append(row)
    assert len(rows)==2
    for arm,cfg in req['arms'].items():
        directory=DATA/arm;cost=req['counts'][arm]
        with np.load(directory/'latents.npz') as f:
            latents=f['latents'];assert np.array_equal(f['labels'],labels)
        assert latents.shape==(1000,4,32,32) and np.isfinite(latents).all()
        merged={}
        for name in ('ema','mse'):
            with np.load(directory/name/'samples.npz') as f: merged[name]=f['arr_0']
            assert merged[name].shape==(1000,256,256,3) and merged[name].dtype==np.uint8
        covered=np.zeros(1000,dtype=int);summaries=[]
        for rank in range(4):
            s=json.loads((directory/f'rank{rank}.json').read_text());summaries.append(s)
            assert s['complete'] and s['request_sha256']==sha(DATA/'request.json')
            assert s['pixel_file_sha256']==sha(directory/f'rank{rank}.npz')
            with np.load(directory/f'rank{rank}.npz') as f:
                idx=f['indices'];assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                assert array_sha(noise[idx])==s['noise_sha256'] and array_sha(labels[idx])==s['label_sha256']
                assert np.array_equal(latents[idx],f['latents'])
                for name in merged: assert np.array_equal(merged[name][idx],f[name])
                covered[idx]+=1
            with np.load(DATA/f'preflight_rank{rank}.npz') as f:
                assert np.array_equal(f[arm],latents[rank*4:rank*4+4])
        assert np.all(covered==1) and sum(s['samples'] for s in summaries)==1000
        assert sum(s['full_batch_calls'] for s in summaries)==cfg['steps']*250
        assert sum(s['full_sample_calls'] for s in summaries)==1000*cost['full_per_image']
        assert sum(s['prefix_sample_calls'] for s in summaries)==1000*cost['prefix_per_image']
        assert cost['block_evaluations_per_image']==13272
        for name in ('ema','mse'):
            sub=directory/name;metrics=json.loads((sub/'fid.json').read_text())[0]
            result=json.loads((sub/'result.json').read_text())
            assert metrics['sample_sha256']==sha(sub/'samples.npz')
            assert metrics['evaluator_commit']=='19dfb4c2705333eb8b97e454fb354d47d1fe135b'
            assert metrics['fid_reference']=='imagenet_256_fid_stats'
            feature,=(sub/'features').glob('*.features.pt')
            x=torch.load(feature,map_location='cpu',weights_only=True).numpy().astype(float)
            assert x.shape==(1000,2048) and np.isfinite(x).all()
            mean_term,cov_term=fid_components(x,mu,cov)
            assert abs(mean_term+cov_term-metrics['fid'])<2e-4
            feature_hashes[arm+'_'+name]=sha(feature)
            trajectory=sum(s['timings']['trajectory_seconds'] for s in summaries)
            coupling=sum(s['timings']['noise_coupling_seconds'] for s in summaries)
            decoding=sum(s['timings']['decode_seconds'][name] for s in summaries)
            assert trajectory==result['trajectory_gpu_seconds'] and coupling==result['noise_coupling_gpu_seconds']
            assert decoding==result['decode_gpu_seconds']
            rows.append(dict(arm=arm,vae=name,samples=1000,fid=metrics['fid'],independent_fid=mean_term+cov_term,
                mean_term=mean_term,covariance_term=cov_term,inception_score=metrics['inception_score'],
                full_per_image=cost['full_per_image'],prefix_per_image=cost['prefix_per_image'],
                block_evaluations_per_image=cost['block_evaluations_per_image'],trajectory_gpu_seconds=trajectory,
                noise_coupling_gpu_seconds=coupling,decode_gpu_seconds=decoding,inference_gpu_seconds=trajectory+coupling+decoding,
                pixel_sha256=metrics['sample_sha256'],reused=False))
            print(json.dumps(rows[-1]),flush=True)
    indexed={(r['arm'],r['vae']):r for r in rows}
    contrasts={name:{'pfr_minus_native_fid':indexed['time250',name]['fid']-indexed['ordinary250',name]['fid'],
                     'pfr_minus_budget_control_fid':indexed['time250',name]['fid']-indexed['ordinary281',name]['fid'],
                     'pfr_to_budget_control_time':indexed['time250',name]['inference_gpu_seconds']/indexed['ordinary281',name]['inference_gpu_seconds']}
               for name in ('ema','mse')}
    audit=dict(passed=True,research_goal_achieved=False,sampling_bank_role='reused_1k_discovery',
        request_sha256=sha(DATA/'request.json'),analysis_source_sha256=sha(Path(__file__)),
        fid_helper_sha256=sha(ROOT/'experiments/analyze_official_sit_pfr_symmetric_20260909.py'),
        native_audit_sha256=req['native_audit_sha256'],feature_sha256=feature_hashes,contrasts=contrasts,
        maximum_fid_reconstruction_error=max(abs(r['fid']-r['independent_fid']) for r in rows),
        limitation='Fixed author-parameter SDE transfer; no retuned CFG/IG or independent sampling confirmation. Time includes Brownian coupling overhead.')
    paths=[OUT/'official_sit_sde_pfr_control.csv',OUT/'official_sit_sde_pfr_control_audit.json']
    assert not any(p.exists() for p in paths)
    with paths[0].open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paths[1].write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__': main()
