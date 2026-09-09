"""Oracle source-mixture calibration followed by continued ordinary AG."""
import json,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr,expit
from experiments.audit_ag_source_condition_bridge import logdensity_score

def cdf(x,t,kind,weight=.5):
    def one(sd):
        v=np.sqrt(sd*sd+t*t)
        return weight*ndtr((x+2)/v)+(1-weight)*ndtr((x-2)/v)
    if kind=='strong':return one(.35)
    if kind=='weak':return one(1.2)
    if kind=='mixture':return .5*(one(.35)+one(1.2))
    return one(float(kind))
def quantile(u,t,kind,weight=.5):
    lo=np.full_like(u,-40.);hi=np.full_like(u,40.)
    assert np.all(cdf(lo,t,kind,weight)<u) and np.all(cdf(hi,t,kind,weight)>u)
    for _ in range(50):
        mid=(lo+hi)/2;left=cdf(mid,t,kind,weight)<u
        lo=np.where(left,mid,lo);hi=np.where(left,hi,mid)
    return (lo+hi)/2
def transport(x,t,s,kind):return quantile(cdf(x,t,kind),s,kind)
def rho(x,t):
    p,_=logdensity_score(x,.35,t);q,_=logdensity_score(x,1.2,t)
    return expit(p-q)
def guided(x,n):
    def v(z,t):
        _,a=logdensity_score(z,.35,t);_,b=logdensity_score(z,1.2,t)
        return -t*(a+.78*(a-b))
    for t,s in zip(np.linspace(3.,0.,n+1)[:-1],np.linspace(3.,0.,n+1)[1:]):
        h=s-t;k1=v(x,t);k2=v(x+h*k1/2,t+h/2);k3=v(x+h*k2/2,t+h/2);k4=v(x+h*k3,s)
        x=x+h*(k1+2*k2+2*k3+k4)/6
    return x
def main():
    start=time.perf_counter();u=(np.arange(8192)+.5)/8192
    initial=quantile(u,3.,'strong')
    recover=transport(transport(initial,3.,2.,'strong'),2.,3.,'strong')
    parity=float(np.max(np.abs(recover-initial)));assert parity<1e-10
    starts={};calibration=[]
    for name in ['ordinary_ag','weak_reference','mixture_reference','weak_posterior_relaxed']:
        z=initial.copy()
        if name!='ordinary_ag':
            for _ in range(2):
                future=transport(z,3.,2.,'strong')
                kind='mixture' if name=='mixture_reference' else 'weak'
                returned=transport(future,2.,3.,kind)
                if name=='weak_posterior_relaxed':z=z+(1-rho(z,3.))*(returned-z)
                else:z=returned
        starts[name]=z
        calibration.append(dict(arm=name,mean_source_posterior_before=float(rho(initial,3.).mean()),
                                mean_source_posterior_after=float(rho(z,3.).mean()),
                                write_mse=float(np.square(z-initial).mean())))
    targets={'underfit':quantile(u,0.,.25),'strong_correct':quantile(u,0.,.35),
             'shared_wrong_mode_mass':quantile(u,0.,.35,.8)}
    rows=[];endpoints={};numerical=[]
    for name,z in starts.items():
        coarse=guided(z.copy(),100);fine=guided(z.copy(),200)
        error=float(np.max(np.abs(coarse-fine)));assert error<1e-4
        numerical.append(dict(arm=name,max_endpoint_100_200_difference=error))
        endpoints[name]=fine
        for target,q in targets.items():
            rows.append(dict(arm=name,target=target,w2_squared=float(np.square(np.sort(fine)-q).mean()),
                             left_mode_mass=float(np.mean(fine<0)),mean=float(fine.mean()),variance=float(fine.var())))
    strong=quantile(u,0.,'strong')
    for target,q in targets.items():rows.append(dict(arm='strong_exact',target=target,w2_squared=float(np.square(strong-q).mean()),left_mode_mass=.5,mean=0.,variance=float(strong.var())))
    out=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908'
    np.savez_compressed(out/'ag_source_calibration_toy.npz',quantiles=u,initial=initial,**endpoints)
    result=dict(complete=True,rows=rows,calibration=calibration,numerical=numerical,reference_roundtrip_max_error=parity,
                seconds=time.perf_counter()-start,protocol=dict(samples=8192,source_prior=.5,gamma=.78,calibrations=2,calibration_interval=[3.,2.],
                continued_guidance=True,initial_law='exact strong sigma=3 marginal, not pure Gaussian',quality_scope='one-dimensional controlled distributions; not neural FID or novelty evidence'))
    (out/'ag_source_calibration_toy.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
