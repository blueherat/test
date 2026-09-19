"""A single global contamination proportion fitted without generated FID."""
import argparse
from pathlib import Path
import numpy as np
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_pasted_20260912.audit import REFS
from .endpoints import ROOT,STAGE


def load(model,kind):
    root=ROOT/model/STAGE/kind
    summary=c.read(root/'metrics.json')
    assert summary['complete'] and c.sha(root/'samples.npz')==summary['samples_sha256']
    if model=='sit_small':
        p=root/'activations.npz'
        with np.load(p) as d:x=d['pool_3'].astype(np.float64)
    else:
        paths=list((root/'features').glob('*.features.pt'));assert len(paths)==1
        p=paths[0];x=torch.load(p,map_location='cpu',weights_only=True).numpy().astype(np.float64)
    assert x.shape==(1000,2048) and np.isfinite(x).all()
    return x,dict(path=str(p),sha256=c.sha(p),metric_sha256=c.sha(root/'metrics.json'),
                  samples_sha256=summary['samples_sha256'])


def kernels(s,w,mu,cov):
    # Reference covariance is treated as a fixed target population moment.
    moment=cov+np.outer(mu,mu)
    scale2=np.trace(moment)
    assert scale2>0
    k=lambda x,y:(1+x@y.T/scale2)**2
    kt=lambda x:1+2*(x@mu)/scale2+np.einsum('nd,nd->n',x@moment,x)/scale2**2
    ktt=1+2*(mu@mu)/scale2+np.square(moment).sum()/scale2**2
    return dict(ss=k(s,s),sw=k(s,w),ww=k(w,w),st=kt(s),wt=kt(w),tt=ktt,scale2=scale2)


def terms(k,counts=None):
    n=len(k['st'])
    if counts is None:counts=np.ones(n)
    counts=np.asarray(counts,dtype=np.float64)
    # Disallow identical ORIGINAL seeds even when an index is resampled.
    denom=counts.sum()**2-(counts**2).sum()
    assert denom>0
    def u(key):
        m=k[key]
        return (counts@m@counts-np.dot(counts**2,np.diag(m)))/denom
    st=np.dot(counts,k['st'])/counts.sum();wt=np.dot(counts,k['wt'])/counts.sum()
    return np.array([u('ss')-2*st+k['tt'],
                     u('sw')-st-wt+k['tt'],
                     u('ww')-2*wt+k['tt']])


def subset(k,ids):
    ids=np.asarray(ids)
    return {key:(x[np.ix_(ids,ids)] if key in ('ss','sw','ww') else
                 x[ids] if key in ('st','wt') else x) for key,x in k.items()}


def corrected(a,kappa):
    d0,ab,bb=a
    return (d0-2*kappa*ab+kappa*kappa*bb)/(1-kappa)**2


def analyze(model):
    out=ROOT/model/'moment_contamination'
    s,srec=load(model,'strong')
    with np.load(REFS[model]) as d:mu=d['mu'].astype(np.float64);cov=d['sigma'].astype(np.float64)
    rows=[];arrays={}
    for reference in ('weak','null'):
        w,wrec=load(model,reference);allk=kernels(s,w,mu,cov)
        fitk=subset(allk,np.arange(500));testk=subset(allk,np.arange(500,1000))
        train=terms(fitk);test=terms(testk)
        kappa=float(train[1]/train[2]) if train[2]>0 else None
        admissible=kappa is not None and 0<kappa<1
        rng=np.random.default_rng(2026121319)
        boot=[]
        # Approximate paired-seed resampling variability, not a significance test
        # for stratified class sampling or an interval for the true density.
        for _ in range(500):
            weight=np.bincount(rng.integers(0,500,500),minlength=500)
            train_b=terms(fitk,weight)
            test_b=terms(testk,np.bincount(rng.integers(0,500,500),minlength=500))
            kb=train_b[1]/train_b[2] if train_b[2]>0 else np.nan
            boot.append([kb,*test_b,corrected(test_b,kappa) if admissible else np.nan])
        boot=np.asarray(boot)
        arrays[reference+'_bootstrap']=boot
        arrays[reference+'_train_test_terms']=np.stack([train,test])
        before=float(test[0]);after=float(corrected(test,kappa)) if admissible else None
        remaining=float(test[0]-2*kappa*test[1]+kappa**2*test[2]) if kappa is not None else None
        row=dict(model=model,track='IG' if reference=='weak' else 'CFG',reference=reference,
            kappa_unconstrained=kappa,admissible_global_range=admissible,
            kappa_resampling_2p5_97p5=np.nanquantile(boot[:,0],[.025,.975]).tolist(),
            fit_terms=train.tolist(),heldout_terms=test.tolist(),
            heldout_mmd2_strong=before,heldout_mmd2_subtracted=after,
            heldout_change=(after-before) if admissible else None,
            heldout_mixture_equation_residual=remaining,
            heldout_fraction_equation_residual=remaining/before if before>0 and remaining is not None else None,
            corrected_change_resampling_2p5_97p5=(
                np.quantile(boot[:,4]-boot[:,1],[.025,.975]).tolist() if admissible else None),
            reference_scale2=float(allk['scale2']),features={'strong':srec,'reference':wrec},
            positivity_of_actual_densities_established=False,
            coefficient_identified_from_densities=False,
            reference_target='fixed first/second moments; covariance treated as population',
            sampling_caveat='paired seeds, fixed class balance; bootstrap descriptive, not calibrated CI')
        rows.append(row)
        print(model,row['track'],'kappa',kappa,'heldout',before,'->',after,flush=True)
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'resampling.npz',**arrays)
    c.atomic(out/'results.json',dict(complete=True,model=model,rows=rows,
        source_sha256=c.sha(Path(__file__)),protocol_sha256=c.sha(c.WORK/'docs/GUIDANCE_DISTRIBUTION_DATA_PROTOCOL_20260912_ZH.md'),
        endpoint_request_sha256=c.sha(ROOT/model/STAGE/'request.json'),
        reference_path=str(REFS[model]),reference_sha256=c.sha(REFS[model]),
        resampling_sha256=c.sha(out/'resampling.npz')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args()
    torch.set_num_threads(4);analyze(a.model)
