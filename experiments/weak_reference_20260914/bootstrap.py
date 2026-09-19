"""Paired, class-stratified bootstrap for the independent SiT confirmation.

Uncertainty is conditional on this 1K sample bank and fixed reference stats.
It does not remove FID small-sample bias or establish cross-model validity.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
from experiments.guidance_pasted_20260912 import common as c
from experiments.cfg_transport_search_20260913.runner import REFERENCE

ROOT=c.EXPS/'weak_reference_20260914/strong_confirm_1k'
OUT=c.WORK/'docs/data/weak_reference_20260914/bootstrap'
ARMS=['strong_e64','strong_log_k02_w1','strong_e192']


class FixedFeatures:
    def __init__(self,features,mu,cov):
        self.f=features.astype(np.float64);self.mu=mu.astype(np.float64)
        self.g=self.f@cov@self.f.T
        self.squared=(self.f*self.f).sum(1);self.trace=np.trace(cov)
    def fid(self,counts):
        ids=np.flatnonzero(counts);w=counts[ids].astype(np.float64);n=w.sum()
        f=self.f[ids];mean=w@f/n
        g=self.g[np.ix_(ids,ids)];gm=g@w/n;center=float(w@gm/n)
        g=g-gm[:,None]-gm[None,:]+center
        root=np.sqrt(w/(n-1));g*=root[:,None]*root[None,:]
        ev=eigvalsh(g,overwrite_a=True,check_finite=False)
        trace=(w@self.squared[ids]-n*(mean@mean))/(n-1)
        return float(np.sum((mean-self.mu)**2)+trace+self.trace-2*np.sqrt(ev.clip(0)).sum())


def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.perf_counter();seed=2026091419;replicates=200
    with np.load(ROOT/'inputs.npz') as d:labels=d['labels']
    with np.load(REFERENCE) as d:mu=d['mu'].astype(np.float64);cov=d['sigma'].astype(np.float64)
    models=[];hashes={str(REFERENCE):c.sha(REFERENCE),str(ROOT/'inputs.npz'):c.sha(ROOT/'inputs.npz')}
    for arm in ARMS:
        path=ROOT/arm/'inception_activations.npz';hashes[str(path)]=c.sha(path)
        with np.load(path) as d:models.append(FixedFeatures(d['pool_3'],mu,cov))
    observed=np.array([m.fid(np.ones(len(labels))) for m in models])
    for value,arm in zip(observed,ARMS):assert abs(value-c.read(ROOT/arm/'fid.json')['fid'])<.002
    rng=np.random.default_rng(seed);strata=[np.flatnonzero(labels==k) for k in range(100)]
    values=np.empty((replicates,len(ARMS)))
    for i in range(replicates):
        sampled=np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in strata])
        counts=np.bincount(sampled,minlength=len(labels))
        values[i]=[m.fid(counts) for m in models]
        if (i+1)%25==0:print('paired bootstrap',i+1,'/',replicates,flush=True)
    rows=[]
    for index in range(len(ARMS)):
        if index==1:continue
        diff=values[:,1]-values[:,index]
        rows.append(dict(contrast=ARMS[1]+' minus '+ARMS[index],
            observed_delta=float(observed[1]-observed[index]),bootstrap_mean=float(diff.mean()),
            percentile_95_interval=np.quantile(diff,[.025,.975]).tolist(),
            bootstrap_fraction_below_zero=float((diff<0).mean())))
    result=dict(seed=seed,replicates=replicates,class_stratified=True,paired=True,
        arms=ARMS,observed_fids=observed.tolist(),rows=rows,hashes=hashes,
        seconds=time.perf_counter()-start,source_sha256=c.sha(Path(__file__)),
        scope='Conditional bootstrap uncertainty for fixed 1K generated samples and reference. FID bias persists; does not establish generalization to other models or datasets.')
    c.atomic(OUT/'results.json',result)
    np.savez(OUT/'replicates.npz',fid=values,arms=np.array(ARMS))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--band',action='store_true');args=p.parse_args()
    if args.band:
        ROOT=c.EXPS/'weak_reference_20260914/strong_band_confirm_1k'
        OUT=c.WORK/'docs/data/weak_reference_20260914/bootstrap_band'
        ARMS=['strong_e320','strong_band_w1','strong_log_k04_e64_w1',
              'strong_log_k04_e106_w1','strong_log_k04_e106_w075']
    with threadpool_limits(limits=2):main()
