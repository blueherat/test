"""Exact-density test: time confidence can improve while a correct marginal drifts.

Both marginals belong to the SAME linear Gaussian bridge X_t=(1-t)X+tE,
X~N(0,4), E~N(0,1). This tests an isolated oracle TLS update, not a full
TAG reproduction or any learned RAE model.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]


def main():
    started=time.perf_counter()
    t,r=.5,.25
    vt,vr=4*(1-t)**2+t*t,4*(1-r)**2+r*r
    k=1/vr-1/vt
    def logp(x,v):return -.5*(np.log(2*np.pi*v)+x*x/v)
    def posterior_other(x):return expit(logp(x,vr)-logp(x,vt))
    def tls(x):return k*x*posterior_other(x)
    def derivative(x):
        w=posterior_other(x)
        return k*w-k*k*x*x*w*(1-w)
    def confidence(x):return -np.logaddexp(0,logp(x,vr)-logp(x,vt))
    # Integrate in standard-normal coordinates on a fixed wide interval.
    # The omitted |Z|>12 tails have negligible Gaussian-polynomial moments.
    def expect(f):
        return quad(lambda z:f(np.sqrt(vt)*z)*np.exp(-.5*z*z)/np.sqrt(2*np.pi),
                    -12,12,epsabs=1e-12,epsrel=1e-11,limit=300)
    slope,err=expect(lambda x:2*x*tls(x))
    assert slope<0
    rows=[]
    for eta in [.001,.01,.05]:
        # Prove positive derivative globally for these fixed eta:
        # w(1-w)<=exp(-a*x²)/q, a=-k/2, q=sqrt(vt/vr).
        # Thus |TLS'| <= |k| + k²/(a*e*q).
        aa=-k/2;qq=np.sqrt(vt/vr)
        jac_lower=1-eta*(abs(k)+k*k/(aa*np.e*qq))
        assert jac_lower>0
        def mapped(x):return x+eta*tls(x)
        variance,ve=expect(lambda x:mapped(x)**2)
        confidence_gain,ce=expect(lambda x:confidence(mapped(x))-confidence(x))
        # Monotone global change of variables gives KL(T#p_t || p_t).
        kl,ke=expect(lambda x:(mapped(x)**2-x*x)/(2*vt)-np.log1p(eta*derivative(x)))
        second,_=expect(lambda x:tls(x)**2)
        assert abs(variance-(vt+eta*slope+eta*eta*second))<1e-10
        assert variance<vt and confidence_gain>0 and kl>0
        # Independent centered finite-difference check of the analytic derivative.
        xs=np.linspace(-8,8,1001);h=1e-5
        assert np.max(np.abs((tls(xs+h)-tls(xs-h))/(2*h)-derivative(xs)))<1e-9
        rows.append(dict(eta=eta,variance=variance,time_log_posterior_gain=confidence_gain,
                         kl_to_correct_marginal=kl,global_jacobian_lower_bound=jac_lower,
                         quadrature_error_estimates=[ve,ce,ke]))
    result=dict(complete=True,target_data_variance=4,current_time=t,other_time=r,
                current_marginal_variance=vt,other_marginal_variance=vr,
                initial_variance_derivative=slope,derivative_quadrature_error=err,rows=rows,
                seconds=time.perf_counter()-started,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Oracle time-posterior gradient update from an already correct marginal. No full sampler, learned classifier, FID or RAE quality claim.')
    with (ROOT/'experiments/results/terminal_defect_20260908/time_posterior_bias.json').open('x') as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
