"""Verify all symmetric-grid evidence and export discovery curves, not a success claim."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_pfr_symmetric_20260909')
BASE = Path('/home/zhoushunyu/data/eqvae/experiments/official_sit_baseline_control_20260909')
OUT = ROOT/'docs/data/guidance_goal_20260909'
REF = Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def array_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def fid_components(x,mu,cov):
    mean = x.mean(0)
    centered = x-mean
    gram = centered @ cov @ centered.T / (len(x)-1)
    eig = np.linalg.eigvalsh((gram+gram.T)*.5)
    assert eig.min() > -1e-7
    mean_term = float(np.square(mean-mu).sum())
    cov_term = float(np.square(centered).sum()/(len(x)-1)+np.trace(cov)-2*np.sqrt(np.maximum(eig,0)).sum())
    return mean_term,cov_term


def main():
    req = json.loads((DATA/'request.json').read_text())
    assert json.loads((DATA/'status.json').read_text())['phase'] == 'complete'
    assert sha(BASE/'request.json') == req['baseline_request_sha256']
    previous = json.loads((BASE/'request.json').read_text())
    for path,digest in {**previous['sources'],**req['sources']}.items(): assert sha(Path(path)) == digest,path
    assert req['time_reversal']['passed']
    parity = json.loads((DATA/'parity_passed.json').read_text())
    assert parity['passed'] and parity['images'] == 16
    assert all(r['prefix_parity'] and r['passed'] for r in parity['records'])
    assert sha(DATA/'inputs.npz') == req['input_file_sha256']
    with np.load(DATA/'inputs.npz') as bank: noise,labels = bank['noise'],bank['labels']
    assert array_sha(noise) == req['noise_sha256'] and array_sha(labels) == req['label_sha256']
    assert sha(REF) == '925e8b5b4ced42137f9847f97a63250a2bd59b70f33f3f356e03453d0775f1ac'
    with np.load(REF) as ref: mu,cov = ref['mu'].astype(float),ref['sigma'].astype(float)
    entries = []
    diagnostics_summary = {}
    for arm,ref in req['references'].items():
        directory = Path(ref['directory'])
        assert sha(directory/'samples.npz') == ref['pixel_sha256']
        assert sha(directory/'fid.json') == ref['metrics_sha256']
        if (directory/'summary.json').exists():
            s = json.loads((directory/'summary.json').read_text())
            assert s['noise_sha256'] == req['noise_sha256'] and s['label_sha256'] == req['label_sha256']
            assert s['checkpoint_sha256'] == req['checkpoint_sha256'] and s['vae_state_sha256'] == req['vae_state_sha256']
            seconds = s['seconds']
        else:
            s = json.loads((directory/'result.json').read_text())
            seconds = s['gpu_inference_seconds_sum']
            assert s['scale'] == ref['scale'] and s['high'] == 1.
        entries.append((arm,ref['method'],ref['scale'],directory,seconds,True))
    for arm,(method,scale) in req['arms'].items():
        directory = DATA/arm
        result = json.loads((directory/'result.json').read_text())
        assert result['method'] == method and result['scale'] == scale
        with np.load(directory/'samples.npz') as pixels: merged = pixels['arr_0']
        covered = np.zeros(1000,dtype=int)
        summaries = []
        for rank in range(4):
            s = json.loads((directory/f'rank{rank}.json').read_text())
            assert s['complete'] and s['scale'] == scale and s['mode'] == method
            assert s['steps'] == 100 and s['batch_size'] == 4 and s['guidance_high'] == 1.
            assert s['request_sha256'] == sha(DATA/'request.json')
            assert sha(directory/f'rank{rank}.npz') == s['pixel_file_sha256']
            with np.load(directory/f'rank{rank}.npz') as shard:
                idx = shard['indices']
                assert np.array_equal(idx,np.concatenate([np.arange(b*4,b*4+4) for b in range(rank,250,4)]))
                assert np.array_equal(merged[idx],shard['arr_0'])
                assert array_sha(noise[idx]) == s['noise_sha256'] and array_sha(labels[idx]) == s['label_sha256']
                covered[idx] += 1
            summaries.append(s)
        assert np.all(covered == 1)
        assert sum(s['samples'] for s in summaries) == 1000
        assert sum(s['full_sample_calls'] for s in summaries) == 100000
        assert sum(s['prefix_sample_calls'] for s in summaries) == 50000
        assert sum(s['probe_batch_calls'] for s in summaries) == 0
        assert sha(directory/'diagnostics.npz') == result['diagnostic_sha256']
        with np.load(directory/'diagnostics.npz') as d:
            stats = d['diagnostics']; assert stats.shape == (1000,5,4) and np.isfinite(stats).all()
            assert np.array_equal(d['labels'],labels)
            assert d['latents'].shape == (1000,4,32,32) and np.isfinite(d['latents']).all()
        diagnostics_summary[arm] = dict(steps=req['diagnostic_steps'],fields=req['diagnostic_fields'],
            mean=stats.mean(0,dtype=np.float64).tolist(),median=np.median(stats,axis=0).tolist(),
            q95=np.quantile(stats,.95,axis=0).tolist(),alpha_zero_fraction=(stats[:,:,0]==0).mean(0).tolist(),
            median_correction_drift_ratio=np.median(stats[:,:,2]/np.maximum(stats[:,:,3],1e-20),axis=0).tolist())
        entries.append((arm,method,scale,directory,result['gpu_inference_seconds_sum'],False))
    rows,feature_hashes,evaluator_ids = [],{},set()
    for arm,method,scale,directory,seconds,reused in entries:
        metrics = json.loads((directory/'fid.json').read_text())[0]
        assert metrics['sample_sha256'] == sha(directory/'samples.npz')
        with np.load(directory/'samples.npz') as pixels:
            assert pixels['arr_0'].shape == (1000,256,256,3) and pixels['arr_0'].dtype == np.uint8
        feature, = (directory/'features').glob('*.features.pt')
        x = torch.load(feature,map_location='cpu',weights_only=True).numpy().astype(float)
        assert x.shape == (1000,2048) and np.isfinite(x).all()
        mean_term,cov_term = fid_components(x,mu,cov)
        assert abs(mean_term+cov_term-metrics['fid']) < 2e-4
        full,prefix = (115,0) if method == 'ordinary' else (100,50)
        rows.append(dict(arm=arm,method=method,scale=scale,samples=1000,fid=metrics['fid'],
            independent_fid=mean_term+cov_term,mean_term=mean_term,covariance_term=cov_term,
            inception_score=metrics['inception_score'],full_per_image=full,prefix8_per_image=prefix,
            block_evaluations_per_image=full*28+prefix*8,inference_gpu_seconds=seconds,reused=reused,
            noise_sha256=req['noise_sha256'],label_sha256=req['label_sha256'],pixel_sha256=metrics['sample_sha256']))
        feature_hashes[arm] = sha(feature)
        evaluator_ids.add((metrics['evaluator_commit'],metrics['fid_reference']))
        print(json.dumps({k:rows[-1][k] for k in ['arm','fid','mean_term','covariance_term']}),flush=True)
    assert evaluator_ids == {('19dfb4c2705333eb8b97e454fb354d47d1fe135b','imagenet_256_fid_stats')}
    methods = ('ordinary','time_only','projected')
    assert all(sorted(r['scale'] for r in rows if r['method']==m)==[1.35,1.4,1.5,1.75] for m in methods)
    rows.sort(key=lambda r:(methods.index(r['method']),r['scale']))
    best = {m:min((r for r in rows if r['method']==m),key=lambda r:r['fid']) for m in methods}
    audit = dict(passed=True,research_goal_achieved=False,sampling_bank_role='reused_1k_discovery',
        source_sha256=sha(Path(__file__)),request_sha256=sha(DATA/'request.json'),reference_sha256=sha(REF),
        feature_sha256=feature_hashes,maximum_fid_reconstruction_error=max(abs(r['fid']-r['independent_fid']) for r in rows),
        selected_by_method={m:{k:r[k] for k in ('arm','scale','fid','inception_score')} for m,r in best.items()},
        best_pfr_minus_best_ordinary={m:best[m]['fid']-best['ordinary']['fid'] for m in ('time_only','projected')},
        matched_scale_pfr_minus_ordinary={m:{str(r['scale']):r['fid']-next(o['fid'] for o in rows if o['method']=='ordinary' and o['scale']==r['scale']) for r in rows if r['method']==m} for m in ('time_only','projected')},
        limitation='One reused discovery bank; grid selection is not independent confirmation or global tuning. No new-method or causal-mechanism claim.')
    OUT.mkdir(parents=True,exist_ok=True)
    outputs = ['official_sit_pfr_symmetric.csv','official_sit_pfr_symmetric_audit.json','official_sit_pfr_symmetric_diagnostics.json']
    assert not any((OUT/name).exists() for name in outputs)
    with (OUT/outputs[0]).open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/outputs[1]).write_text(json.dumps(audit,indent=2)+'\n')
    (OUT/outputs[2]).write_text(json.dumps(diagnostics_summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    names={'ordinary':'IG (115 full)','time_only':'PFR time only (100 full + 50 prefix)','projected':'PFR projected (100 full + 50 prefix)'}
    for m in methods:
        rr=[r for r in rows if r['method']==m]
        for ax,key in zip(axes,('fid','inception_score')):
            ax.plot([r['scale'] for r in rr],[r[key] for r in rr],marker='o',label=names[m])
    for ax,label in zip(axes,('FID-1K (lower is better)','Inception Score (higher is better)')):
        ax.set_xlabel('IG scale');ax.set_ylabel(label);ax.grid(alpha=.2)
        ax.set_xticks([1.35,1.4,1.5,1.75]);ax.tick_params(axis='x',labelrotation=35)
    axes[0].legend(fontsize=7)
    fig.suptitle('Official SiT-XL/2 + joint IG: paired 1K discovery bank',fontsize=11)
    for ext in ('png','pdf'): fig.savefig(OUT/f'official_sit_pfr_symmetric.{ext}',dpi=200)
    plt.close(fig)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__': main()
