"""Maximum-likelihood Gaussian covariance with exchangeable spatial tokens.

The constant spatial mode has its own full channel covariance; all orthogonal
modes share another. These are direct training-bank moments, not FID fits.
"""
import json
from pathlib import Path
import time
import numpy as np
from experiments.audit_raev2_channel_covariance_20260907 import BANK, OUT as TOKEN_OUT


def main():
    out=TOKEN_OUT.parent/'exchangeable_covariance'
    out.mkdir(exist_ok=False)
    started=time.perf_counter()
    z=np.load(BANK/'train/latents.npy',mmap_mode='r')
    means=np.concatenate([np.array(z[i:i+20],dtype=np.float64).mean((2,3)) for i in range(0,len(z),20)])
    mean_cov=np.cov(means,rowvar=False,ddof=0)
    token=np.load(TOKEN_OUT/'covariance.npz')['covariance']
    token=token*(len(z)*256-1)/(len(z)*256)
    constant=256*mean_cov
    residual=256/255*(token-mean_cov)
    eig0,vec0=np.linalg.eigh(constant)
    eigr,vecr=np.linalg.eigh(residual)
    if eig0.min()<-1e-9 or eigr.min()<-1e-9: raise ValueError('covariance not PSD')
    np.savez(out/'covariance.npz',constant_values=np.maximum(eig0,0),constant_vectors=vec0,
             residual_values=np.maximum(eigr,0),residual_vectors=vecr)
    report={'complete':True,'source':str(BANK/'train/latents.npy'),'train_samples':len(z),'fid_used':False,
        'spatial_modes':256,'constant_trace':float(eig0.sum()),'residual_trace':float(eigr.sum()),
        'mean_coordinate_variance':float((eig0.sum()+255*eigr.sum())/(256*1024)),
        'constant_extreme_eigenvalues':[float(eig0.min()),float(eig0.max())],
        'residual_extreme_eigenvalues':[float(eigr.min()),float(eigr.max())],
        'seconds':time.perf_counter()-started}
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
