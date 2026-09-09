"""Oracle 1D latent transport: endpoint agreement is not distribution correctness."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.special import ndtr,ndtri,log_ndtr,ndtri_exp

ROOT=Path(__file__).resolve().parents[1]
def logcdf(x,m):
 return np.logaddexp(log_ndtr(x-m),log_ndtr(x+m))-np.log(2.)
def inverse_unconditional(z,m):
 # Solve only the lower half, using symmetry and log probabilities.
 target=log_ndtr(-np.abs(z));lo=-m-np.abs(z)-10.;hi=np.zeros_like(z)
 for _ in range(80):
  mid=(lo+hi)/2;left=logcdf(mid,m)<target;lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
 lower=(lo+hi)/2
 return np.where(z<0,lower,-lower)
def transport(z,m):
 y=m+z;lower=ndtri_exp(logcdf(-np.abs(y),m))
 return np.where(y<0,lower,-lower)
def main():
 q=(np.arange(8192)+.5)/8192;initial=ndtri(q);rows=[]
 root=ROOT/'experiments/results/terminal_defect_20260908';out=root/'fsg_exact_gaussian_transport.csv'
 for m in [.5,2.,4.]:
  z=initial.copy();target=m+initial
  for k in range(65):
   if k in [0,1,2,4,8,16,32,64]:
    u=inverse_unconditional(z,m);c=m+z;gap=c-u;nextz=transport(z,m)
    # Analytic T' = p_mix(C(z))/phi(T(z)), sigma_data=1.
    logpdf=np.logaddexp(-.5*(c-m)**2,-.5*(c+m)**2)-np.log(2.)-.5*np.log(2*np.pi)
    deriv=np.exp(logpdf+.5*nextz**2+.5*np.log(2*np.pi))
    assert np.all(gap>0) and np.all(nextz>z)
    err=float(np.max(np.abs(inverse_unconditional(nextz,m)-c)));assert err<1e-10
    w2=float(np.mean((u-target)**2))
    if k==1:assert w2<1e-20
    if k>=2:assert w2>1e-5
    rows.append(dict(class_mean=m,iterations=k,quantile_nodes=len(z),conditional_target_w2_squared=w2,endpoint_gap_mse=float(np.mean(gap**2)),endpoint_gap_min=float(gap.min()),decoded_mean=float(u.mean()),decoded_std=float(u.std()),latent_mean=float(z.mean()),latent_std=float(z.std()),mean_local_derivative_squared=float(np.mean(deriv**2)),max_local_derivative=float(deriv.max()),oracle_composition_max_error=err))
   z=transport(z,m)
 frame=pd.DataFrame(rows);frame.to_csv(out,index=False,mode='x')
 for m,part in frame.groupby('class_mean'):
  assert np.all(np.diff(part.endpoint_gap_mse)<0)
  assert np.all(np.diff(part[part.iterations>=1].conditional_target_w2_squared)>0)
 print(frame.to_string(index=False))
 (root/'fsg_exact_gaussian_transport_source.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),grid='8192 deterministic midpoint Gaussian quantiles',scope='exact Gaussian/mixture oracle; squared W2 midpoint quadrature, not neural image quality'),indent=2))
if __name__=='__main__':main()
