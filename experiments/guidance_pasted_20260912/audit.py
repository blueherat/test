"""Audit completed artifacts and reconstruct FID from their cached features."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from . import common as c

REFS={'sit_small':c.EXPS.parent/'imagenet_sit_flow/adm_reference_stats/imagenet100_validation_n5000_adm_stats.npz',
    'raev2':Path('/home/zhoushunyu/.cache/nanogen-evals/stats/datasets--nanovisionx--nanogen-evals-stats/snapshots/0227134b29f25704c3856ec002ce4a2183cc7419/imagenet_256_fid_stats.npz')}


def fid_from_features(features,mu,cov):
    x=np.asarray(features,dtype=np.float64);mu=np.asarray(mu,dtype=np.float64);cov=np.asarray(cov,dtype=np.float64)
    center=x.mean(0);x=x-center
    gram=x@cov@x.T/(len(x)-1)
    eig=np.linalg.eigvalsh((gram+gram.T)*.5)
    assert eig.min()>-1e-6
    return float(np.square(center-mu).sum()+np.square(x).sum()/(len(x)-1)+np.trace(cov)-2*np.sqrt(np.maximum(eig,0)).sum())


def audit(model,stage):
    root=c.ROOT/model/stage
    if not (root/'results.json').exists():return
    results=c.read(root/'results.json');all_rows=[]
    with np.load(REFS[model]) as d:mu=d['mu'];cov=d['sigma']
    inputs=c.bank(model,stage)
    for result in results:
        out=root/result['arm'];path=out/'audit.json'
        if path.exists():
            row=c.read(path);assert row['samples_sha256']==result['samples_sha256'];all_rows.append(row);continue
        summary=c.read(out/'summary.json');assert summary['complete'] and c.sha(out/'samples.npz')==summary['samples_sha256']
        with np.load(out/'samples.npz') as d:pixels=d['arr_0']
        starts=[];seconds=0.;replay=[]
        for record in summary['records']:
            p=Path(record['file']);assert c.sha(p)==record['sha256']
            with np.load(p) as d:
                start=int(d['start']);n=len(d['arr_0']);starts.extend(range(start,start+n))
                np.testing.assert_array_equal(d['arr_0'],pixels[start:start+n])
                np.testing.assert_array_equal(d['labels'],inputs[2][start:start+n])
                assert str(d['noise_sha256'])==c.array_sha(inputs[0][start:start+n])
                assert np.isfinite(d['latents']).all() and np.isfinite(d['second_latents']).all()
                assert d['pair_pixels'].shape==d['arr_0'].shape
                seconds+=float(d['seconds'])
        assert starts==list(range(len(pixels))) and abs(seconds-summary['seconds'])<1e-8
        if model=='sit_small':
            fp=out/'activations.npz'
            with np.load(fp) as d:features=d['pool_3']
        else:
            paths=list((out/'features').glob('*.features.pt'));assert len(paths)==1;fp=paths[0]
            features=torch.load(fp,map_location='cpu',weights_only=True).numpy()
        fid=fid_from_features(features,mu,cov);error=abs(fid-result['fid']);assert error<2e-3,(model,stage,result['arm'],error)
        if stage=='ig_calibrated_screen_400' and result['arm']=='native_base':
            previous=c.ROOT/model/'ig_screen_400/native_base/samples.npz'
            with np.load(previous) as d:np.testing.assert_array_equal(pixels,d['arr_0'])
            replay.append(dict(raw_native_samples_sha256=c.sha(previous),primary_pixels_exact=True))
        row=dict(passed=True,model=model,stage=stage,arm=result['arm'],samples=len(pixels),
            samples_sha256=summary['samples_sha256'],coverage_and_input_hashes=True,pair_secondary_preserved=True,
            costs_reconciled=True,fid_reported=result['fid'],fid_fp64_same_features=fid,absolute_error=error,
            reference=str(REFS[model]),reference_sha256=c.sha(REFS[model]),features_sha256=c.sha(fp),
            independent_feature_extractor=False,baseline_replay=replay)
        c.atomic(path,row);all_rows.append(row)
    immutable={}
    request_path=root/'request.json'
    if stage=='ig_screen_400':request_path=c.ROOT/model/'ig_source_data/request.json'
    elif stage=='ig_calibrated_screen_400':request_path=c.ROOT/model/'ig_calibrated_source/request.json'
    if request_path.exists():
        request=c.read(request_path)
        for group in ('sources','assets','inputs'):
            for path,h in request.get(group,{}).items():assert c.sha(path)==h,path;immutable[path]=h
    c.atomic(root/'audit.json',dict(passed=True,arms=len(all_rows),quality_results_sha256=c.sha(root/'results.json'),
        immutable_files_verified=len(immutable),records=all_rows))
    print(model,stage,'audited',len(all_rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--stage',required=True);a=p.parse_args()
    torch.set_num_threads(4);audit(a.model,a.stage)
