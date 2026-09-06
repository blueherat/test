#!/usr/bin/env python3
"""Estimate a channel covariance from the existing ImageNet TRAIN bank only.

This is a source-statistics diagnostic, not a FID-selected guidance schedule.
Pooling all spatial tokens imposes stationarity; no class or full-image
covariance claim is made. Held-out rows are used only to assess covariance fit.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

BANK=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/potential_clean_bank_fp32_v1')
OUT=Path('/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_20260907/channel_covariance')


def moments(path):
    a=np.load(path,mmap_mode='r')
    total=np.zeros(1024,dtype=np.float64)
    gram=np.zeros((1024,1024),dtype=np.float64)
    for i in range(0,len(a),20):
        x=np.array(a[i:i+20],dtype=np.float64).transpose(0,2,3,1).reshape(-1,1024)
        total+=x.sum(0)
        gram+=x.T@x
    n=len(a)*256
    mean=total/n
    return mean,(gram-n*np.outer(mean,mean))/(n-1),a.shape


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    mu,cov,shape=moments(BANK/'train/latents.npy')
    eig,vec=np.linalg.eigh(cov)
    mu_v,cov_v,shape_v=moments(BANK/'validation/latents.npy')
    np.savez(OUT/'covariance.npz',mean=mu,covariance=cov,eigenvalues=eig,eigenvectors=vec)
    d=eig[::-1]
    result={'complete':True,'train_bank':str(BANK/'train/latents.npy'),'train_shape':shape,
        'source_summary_sha256':hashlib.sha256((BANK/'summary.json').read_bytes()).hexdigest(),
        'train_only_estimator':True,'validation_used_for_estimation':False,'validation_shape':shape_v,
        'trace':float(eig.sum()),'min_eigenvalue':float(eig[0]),'max_eigenvalue':float(eig[-1]),
        'effective_rank_participation':float(eig.sum()**2/(eig@eig)),
        'top_fraction':{str(k):float(d[:k].sum()/d.sum()) for k in [1,8,32,128,512]},
        'validation_relative_covariance_error':float(np.linalg.norm(cov_v-cov)/np.linalg.norm(cov_v)),
        'validation_mean_rms_gap':float(np.mean((mu_v-mu)**2)**.5),
        'seconds':time.perf_counter()-start,'pid':os.getpid()}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
