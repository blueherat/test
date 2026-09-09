"""Exact-density bridge between AG log odds and implicit source conditioning."""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp,expit

def logdensity_score(x,sigma,noise):
    variance=sigma*sigma+noise*noise
    component=-.5*(x[:,None]-np.array([-2.,2.]))**2/variance-.5*np.log(2*np.pi*variance)-np.log(2.)
    logp=logsumexp(component,axis=1)
    weights=np.exp(component-logp[:,None])
    score=(weights*(-(x[:,None]-np.array([-2.,2.]))/variance)).sum(1)
    return logp,score

def main():
    x=np.linspace(-8,8,32001);gamma=.78;rows=[]
    for noise in [0.,.5,1.,3.]:
        lp,sp=logdensity_score(x,.35,noise);lq,sq=logdensity_score(x,1.2,noise)
        for prior in [.2,.5,.8]:
            a=expit(np.log(prior/(1-prior))+lp-lq)
            sm=a*sp+(1-a)*sq
            score_log_posterior=(1-a)*(sp-sq)
            direct=(sp-sm)
            np.testing.assert_allclose(direct,score_log_posterior,atol=1e-13,rtol=1e-11)
            # Algebraically equivalent adaptive source-CFG, not a proposed estimator.
            adaptive=sp+gamma/(1-a)*direct
            ordinary=sp+gamma*(sp-sq)
            np.testing.assert_allclose(adaptive,ordinary,atol=1e-11,rtol=1e-10)
            eps=1e-5
            def posterior_log(xx):
                p,_=logdensity_score(xx,.35,noise);q,_=logdensity_score(xx,1.2,noise)
                return -np.logaddexp(0,-np.log(prior/(1-prior))-p+q)
            fd=(posterior_log(x+eps)-posterior_log(x-eps))/(2*eps)
            np.testing.assert_allclose(fd,score_log_posterior,atol=5e-8,rtol=1e-6)
            center=len(x)//2
            rows.append(dict(noise=noise,source_prior=prior,max_score_identity_error=float(np.max(np.abs(direct-score_log_posterior))),
                             max_adaptive_equivalence_error=float(np.max(np.abs(adaptive-ordinary))),
                             max_finite_difference_error=float(np.max(np.abs(fd-score_log_posterior))),
                             center_source_posterior=float(a[center]),center_score_gap=float(sp[center]-sq[center]),
                             max_source_posterior=float(a.max())))
    assert rows[1]['center_source_posterior']<1e-5 and rows[1]['center_score_gap']==0
    result=dict(complete=True,gamma=gamma,rows=rows,
                scope='analytic two-component densities and Gaussian smoothing; no neural density assumption, sampler or quality experiment',
                conclusion='source posterior is a possible information carrier; AG equals adaptive implicit-source CFG; zero score gap does not imply high source posterior')
    out=Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908/ag_source_condition_bridge.json'
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
