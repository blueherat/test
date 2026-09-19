"""Offline image-distribution diagnostic, never a deployment proposal."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,brentq
from scipy.special import expit
import torch
from experiments.guidance_pasted_20260912 import common as c
from experiments.guidance_pasted_20260912.audit import REFS
from . import mixture as m,moments

PROTOCOL=c.WORK/'docs/CONTAMINATION_ENDPOINT_FILTER_PROTOCOL_20260912_ZH.md'


def weighted_mmd(k,weight,counts=None):
    if counts is None:counts=np.ones(len(weight))
    q=np.asarray(weight)*counts;n=q.sum()
    denom=n*n-np.square(q).sum()
    assert denom>0
    u=(q@k['ss']@q-np.dot(q*q,np.diag(k['ss'])))/denom
    return u-2*np.dot(q,k['st'])/n+k['tt']


def analyze(model,reference):
    track='ig' if reference=='weak' else 'cfg'
    root=m.ROOT/model/(track+'_endpoint_filter')
    s,srec=moments.load(model,'strong');w,wrec=moments.load(model,reference)
    with np.load(REFS[model]) as d:mu=d['mu'];cov=d['sigma']
    kappa=m.coefficient(model,track)
    features=np.concatenate((s[:400],w[:400]))
    mean=features.mean(0);std=features.std(0).clip(1e-4)
    x=(features-mean)/std;y=np.r_[np.ones(400),np.zeros(400)]
    def objective(theta):
        ww,b=theta[:-1],theta[-1];logit=x@ww+b
        p=expit(logit);res=(p-y)/len(y)
        loss=np.mean(np.logaddexp(0,logit)-y*logit)+.5*np.square(ww).sum()
        grad=np.r_[x.T@res+ww,res.sum()]
        return loss,grad
    result=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',
                    options=dict(maxiter=1000,gtol=1e-6,ftol=1e-12))
    assert result.success,result.message
    theta=result.x
    pred=lambda a:(a-mean)/std@theta[:-1]+theta[-1]
    logits=np.r_[pred(s[400:500]),pred(w[400:500])]
    truth=np.r_[np.ones(100),np.zeros(100)]
    gradient=lambda beta:np.mean((expit(beta*logits)-truth)*logits)
    hi=1.
    while gradient(hi)<0 and hi<1024:hi*=2
    beta=0. if gradient(0)>=0 else brentq(gradient,0,hi)
    l_s=beta*pred(s[500:]);l_w=beta*pred(w[500:])
    logrho=np.log(kappa)-l_s
    accept=1-np.exp(np.minimum(logrho,0))
    invalid=logrho>=0
    ker=moments.kernels(s[500:],s[500:],mu,cov)
    before=weighted_mmd(ker,np.ones(500));after=weighted_mmd(ker,accept)
    # Equal weights must recover the same unweighted off-diagonal estimate.
    np.testing.assert_allclose(before,moments.terms(ker)[0],atol=1e-12,rtol=0)
    rng=np.random.default_rng(2026121361);boot=[]
    for _ in range(500):
        count=np.bincount(rng.integers(0,500,500),minlength=500)
        boot.append(weighted_mmd(ker,accept,count)-weighted_mmd(ker,np.ones(500),count))
    accepted=rng.random(500)<accept
    testlogits=np.r_[l_s,l_w];testy=np.r_[np.ones(500),np.zeros(500)]
    root.mkdir(parents=True,exist_ok=True)
    np.savez(root/'source_data.npz',weights=theta,mean=mean,std=std,beta=beta,
        strong_logit=l_s,reference_logit=l_w,accept_probability=accept,accepted=accepted,
        heldout_ids=np.arange(500,1000),bootstrap_change=np.asarray(boot))
    summary=dict(complete=True,model=model,track=track,kappa=kappa,inverse_temperature=float(beta),
        regularization='mean BCE + .5*l2 squared; unregularized intercept',
        optimizer_success=bool(result.success),optimizer_iterations=int(result.nit),
        source_audit_bce=float(np.mean(np.logaddexp(0,testlogits)-testy*testlogits)),
        source_audit_brier=float(np.mean((expit(testlogits)-testy)**2)),
        source_audit_accuracy=float(((testlogits>=0)==testy).mean()),
        mean_accept_probability=float(accept.mean()),ideal_accept_probability=1-kappa,
        predicted_negative_accept_fraction=float(invalid.mean()),realized_accepted=int(accepted.sum()),
        effective_sample_size=float(accept.sum()**2/np.square(accept).sum()),
        mmd2_before=float(before),mmd2_after_weighted=float(after),mmd2_change=float(after-before),
        change_resampling_2p5_97p5=np.quantile(boot,[.025,.975]).tolist(),
        density_ratio_only_in_inception_feature_space=True,deployment_method=False,
        source_records=[srec,wrec],source_sha256=c.sha(Path(__file__)),
        protocol_sha256=c.sha(PROTOCOL),artifact_sha256=c.sha(root/'source_data.npz'))
    c.atomic(root/'summary.json',summary)
    print(model,track,'endpoint filter',before,'->',after,'accept',accept.mean(),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    for model in c.MODELS:
        for reference in ('weak','null'):analyze(model,reference)
