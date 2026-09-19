"""Reconcile actual images, source hashes, moments, and finite-sample FID."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_pasted_20260912.audit import REFS,fid_from_features
from . import mixture as m


def cached_features(root,model):
    if model=='sit_small':
        p=root/'activations.npz'
        with np.load(p) as d:x=d['pool_3']
    else:
        files=list((root/'features').glob('*.features.pt'));assert len(files)==1
        p=files[0];x=torch.load(p,map_location='cpu',weights_only=True).numpy()
    return x,p


def arm(model,stage,kind):
    root=m.ROOT/model/stage/kind
    summary=c.read(root/'summary.json');metric=c.read(root/'metrics.json')
    assert summary['complete']
    out=root/'audit.json'
    if out.exists():
        old=c.read(out);assert old['samples_sha256']==summary['samples_sha256'];return old
    with np.load(root/'samples.npz') as d:pixels=d['arr_0']
    assert c.sha(root/'samples.npz')==summary['samples_sha256']==metric['samples_sha256']
    noise=np.load(m.ROOT/model/stage/'inputs/first.npy',mmap_mode='r')
    labels=np.load(m.ROOT/model/stage/'inputs/labels.npy')
    request=m.ROOT/model/stage/'request.json'
    coverage=[];seconds=0.;full_sum=prefix_sum=0;traces=[]
    for record in summary['records']:
        p=Path(record['file']);assert c.sha(p)==record['sha256']
        with np.load(p) as d:
            start=int(d['start']);n=len(d['arr_0']);coverage.extend(range(start,start+n))
            np.testing.assert_array_equal(d['arr_0'],pixels[start:start+n])
            np.testing.assert_array_equal(d['labels'],labels[start:start+n])
            assert str(d['noise_sha256'])==c.array_sha(noise[start:start+n])
            assert str(d['request_sha256'])==c.sha(request)
            assert np.isfinite(d['latents']).all()
            seconds+=float(d['seconds']);full_sum+=int(d['full_calls'])*n;prefix_sum+=int(d['prefix_calls'])*n
            if 'trace' in d:traces.append(d['trace'])
    assert coverage==list(range(len(pixels)))
    assert abs(seconds-summary['seconds'])<1e-6
    assert abs(full_sum/len(pixels)-summary['full_calls_per_output'])<1e-8
    if stage!='endpoint_sources':assert prefix_sum==0
    if traces:
        with np.load(root/'traces.npz') as d:np.testing.assert_array_equal(d['values'],np.concatenate(traces))
    with np.load(REFS[model]) as d:mu=d['mu'];cov=d['sigma']
    x,fp=cached_features(root,model);assert len(x)==len(pixels)
    recalculated=fid_from_features(x,mu,cov);error=abs(recalculated-metric['fid'])
    assert error<2e-3,(model,stage,kind,error)
    result=dict(passed=True,model=model,stage=stage,arm=kind,samples=len(pixels),images_and_labels_exact=True,
        source_noise_and_request_hashes_verified=True,costs_reconciled=True,
        fid_reported=metric['fid'],fid_fp64_from_same_features=recalculated,absolute_error=error,
        independent_feature_extraction=False,samples_sha256=summary['samples_sha256'],
        features_sha256=c.sha(fp),reference_sha256=c.sha(REFS[model]),records=len(summary['records']))
    c.atomic(out,result);print(model,stage,kind,'audit',error,flush=True)
    return result


def immutable(model,stage):
    rp=m.ROOT/model/stage/'request.json';request=c.read(rp);count=0
    for group in ('sources','assets','inputs','fitted_artifacts'):
        for p,h in request.get(group,{}).items():assert c.sha(p)==h,p;count+=1
    return dict(request_sha256=c.sha(rp),immutable_files=count)


def run(model,stage):
    m.configure();root=m.ROOT/model/stage
    results=c.read(root/'results.json')
    expected=3 if stage=='endpoint_sources' else 6
    assert len(results)==expected
    records=[arm(model,stage,r['arm']) for r in results]
    c.atomic(root/'audit.json',dict(passed=True,records=records,**immutable(model,stage)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True)
    p.add_argument('--stage',required=True);a=p.parse_args()
    torch.set_num_threads(4);run(a.model,a.stage)
