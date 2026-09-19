"""Numerical weak-reference diagnostics with known densities and score errors.

Density curves here are instantaneous normalized potentials, NOT claimed
terminal distributions of a time-varying guided diffusion. The separate ODE
experiment integrates a consistent linear Gaussian path with exact GMM scores.
"""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.integrate import cumulative_trapezoid, quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('docs/data/weak_reference_20260914/toy')


def mixture(x, means, variances, weights):
    x=np.asarray(x)
    d=x[...,None]-np.asarray(means)
    v=np.asarray(variances)
    logits=np.log(weights)-.5*np.log(2*np.pi*v)-.5*d*d/v
    logp=logsumexp(logits,axis=-1)
    score=(np.exp(logits-logp[...,None])*(-d/v)).sum(-1)
    return np.exp(logp),score


def normalize(logp,x):
    p=np.exp(logp-logp.max())
    return p/np.trapz(p,x)


def induced(score,x):
    return normalize(cumulative_trapezoid(score,x,initial=0),x)


def smooth(value,tau,dx):
    return gaussian_filter1d(value,np.sqrt(tau)/dx,mode='nearest',truncate=8.)


def references(score,x,tau=.09,rho=.09):
    dx=x[1]-x[0];p=induced(score,x)
    q=smooth(p,tau,dx)
    weak=np.gradient(np.log(q.clip(1e-200)),dx,edge_order=2)
    mean=np.trapz(x*p,x);var=np.trapz((x-mean)**2*p,x)
    a=np.sqrt(var/(var+tau)); mapped=mean+(x-mean)/a
    # This is exact affine change-of-variable after convolution, up to grid error.
    moment=np.interp(mapped,x,weak)/a
    return {'density':score-weak,'log':score-smooth(score,tau,dx),
            'moment':score-moment,'band':smooth(score,rho,dx)-smooth(score,rho+tau,dx)}


def snapshots():
    x=np.linspace(-10,10,12001)
    p,s=mixture(x,[-2,2],[.25,.25],[.5,.5])
    cases={
        'exact_mixture':(p,s,s),
        'density_oversmoothed':(p,s,mixture(x,[-2,2],[.4,.4],[.5,.5])[1]),
        'score_oversmoothed':(p,s,smooth(s,.15,x[1]-x[0])),
        'smooth_plus_ripple':(p,s,smooth(s,.15,x[1]-x[0])+.25*np.sin(25*x)),
        'spurious_center_peak':(p,s,mixture(x,[-2,0,2],[.25,.04,.25],[.45,.1,.45])[1]),
    }
    pg,sg=mixture(x,[0],[1],[1])
    cases['exact_gaussian']=(pg,sg,sg)
    cases['gaussian_variance_inflation']=(pg,sg,-x/1.5)
    pu,su=mixture(x,[-1.5,1.2],[.12,.65],[.2,.8])
    cases['unequal_mixture']=(pu,su,mixture(x,[-1.5,1.2],[.22,.75],[.2,.8])[1])
    rows=[];curves={'x':x}
    for name,(truth,true_score,model) in cases.items():
        error=model-true_score; risk=np.trapz(truth*error**2,x)
        residuals=references(model,x)
        curves[name+'_true']=truth;curves[name+'_base']=induced(model,x)
        for kind,g in residuals.items():
            alignment=np.trapz(truth*error*g,x);energy=np.trapz(truth*g*g,x)
            weight=max(0,min(3,-alignment/max(energy,1e-20)))
            for omega in (1.,weight):
                pred=model+omega*g
                risk_new=np.trapz(truth*(pred-true_score)**2,x)
                assert abs(risk_new-(risk+2*omega*alignment+omega**2*energy))<1e-8
                q=induced(pred,x)
                rows.append(dict(case=name,method=kind,weight_rule='fixed_1' if omega==1 else 'oracle_capped_3',
                    omega=omega,baseline_score_risk=risk,score_risk=risk_new,
                    alignment=alignment,correction_energy=energy,
                    mean=np.trapz(x*q,x),variance=np.trapz((x-np.trapz(x*q,x))**2*q,x),
                    low_density_mass=np.trapz(q*(truth<.01*truth.max()),x),
                    kl_true_to_potential=np.trapz(truth*(np.log(truth.clip(1e-200))-np.log(q.clip(1e-200))),x)))
            curves[name+'_'+kind]=induced(model+g,x)
    write_csv(OUT/'snapshot_risk.csv',rows)
    np.savez(OUT/'snapshot_curves.npz',**curves)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,name in zip(axes.flat,['exact_gaussian','density_oversmoothed','score_oversmoothed',
                                'smooth_plus_ripple','spurious_center_peak','unequal_mixture']):
        for key,color in [('true','black'),('base','.55'),('density','#c55a33'),('log','#146fa1'),('band','#488458')]:
            ax.plot(x,curves[name+'_'+key],label=key,color=color,lw=1.4)
        ax.set_xlim(-4,4);ax.set_title(name.replace('_',' '),fontsize=10)
        ax.set_xlabel('x');ax.set_ylabel('Normalized potential density')
    axes[0,0].legend(fontsize=8)
    fig.savefig(OUT/'potentials.png',dpi=180);plt.close(fig)
    return rows


def exact_checks():
    table=[]
    for x in [.25,1.75,2.25]:
        score=lambda y:float(mixture(y,[-2,2],[.25,.25],[.5,.5])[1])
        expectation=quad(lambda e:score(x+.3*e)*np.exp(-e*e/2)/np.sqrt(2*np.pi),
                         -12,12,epsabs=1e-12,epsrel=1e-12,limit=300)[0]
        table.append(dict(x=x,density=float(score(x)-mixture(x,[-2,2],[.34,.34],[.5,.5])[1]),
                          log=score(x)-expectation))
    write_csv(OUT/'attachment_point_check.csv',table)
    cov=np.array([[2.,.4],[.4,.2]]);tau=.09
    eig,Q=np.linalg.eigh(cov)
    A=(Q*np.sqrt(eig/(eig+tau)))@Q.T
    err=np.max(np.abs(A@(cov+tau*np.eye(2))@A.T-cov))
    assert err<1e-12
    rng=np.random.default_rng(1401);x=rng.normal(size=(100,2))
    precision=np.linalg.inv(cov);mean=np.array([.5,-.3])
    sp=-(x-mean)@precision
    mapped=mean+(x-mean)@np.linalg.inv(A).T
    sq=(-(mapped-mean)@np.linalg.inv(cov+tau*np.eye(2)))@np.linalg.inv(A)
    assert np.max(np.abs(sp-sq))<1e-10
    return dict(moment_covariance_error=err,moment_gaussian_score_error=float(np.max(np.abs(sp-sq))),points=table)


def calibrate():
    """Direction 3: fit scalar weights on noisy DSM targets, test independently."""
    rng=np.random.default_rng(1405);u=.3;xgrid=np.linspace(-12,12,12001);dx=xgrid[1]-xgrid[0]
    _,true=mixture(xgrid,[-2,2],[.25+u]*2,[.5,.5])
    cases={'score_oversmoothed':smooth(true,.3,dx),'high_frequency_error':true+.25*np.sin(25*xgrid),'spurious_center_peak':
        mixture(xgrid,[-2,0,2],[.25+u,.04+u,.25+u],[.45,.1,.45])[1]}
    rows=[]
    for case,model in cases.items():
        residual=references(model,xgrid)['log']
        banks=[]
        for n in [8192,65536]:
            x0=rng.choice([-2.,2.],size=n)+.5*rng.normal(size=n)
            eps=rng.normal(size=n);x=x0+np.sqrt(u)*eps;target=-eps/np.sqrt(u)
            banks.append((np.interp(x,xgrid,model),np.interp(x,xgrid,residual),target,
                          mixture(x,[-2,2],[.25+u]*2,[.5,.5])[1]))
        model_fit,g_fit,target_fit,_=banks[0]
        weight=float(np.clip(-np.mean((model_fit-target_fit)*g_fit)/np.mean(g_fit*g_fit),0,3))
        m,g,y,oracle=banks[1]
        for name,w in [('baseline',0),('fixed_1',1),('dsm_calibrated',weight)]:
            rows.append(dict(case=case,method=name,omega=w,train_n=8192,test_n=65536,
                test_dsm_loss=float(np.mean((m+w*g-y)**2)),test_true_score_risk=float(np.mean((m+w*g-oracle)**2))))
    write_csv(OUT/'calibration.csv',rows)
    return rows


def ode():
    """Deterministic quantile particles: actual guided SiT-form probability ODE."""
    n=4096;steps=400;eps=ndtri((np.arange(n)+.5)/n)
    nodes,weights=np.polynomial.hermite.hermgauss(16);weights=weights/np.sqrt(np.pi)
    rows=[];endpoints={}
    for variance in [.25,.4]:
        for kind in ['baseline','log','density','moment']:
            z=eps.copy()
            for k in range(1,steps):
                t=k/steps;b=1-t;h=1/steps
                # First step from t=0 has v=-eps (mixture mean zero), exact field limit.
                if k==1:z=eps*(1-1/steps)
                means=np.array([-2.,2.])*t;v=variance*t*t+b*b
                _,s=mixture(z,means,[v,v],[.5,.5]);g=np.zeros_like(s);tau=(.2*b)**2
                if kind=='log':
                    smooth_s=sum(w*mixture(z+np.sqrt(2*tau)*a,means,[v,v],[.5,.5])[1] for a,w in zip(nodes,weights))
                    g=s-smooth_s
                if kind=='density':g=s-mixture(z,means,[v+tau]*2,[.5,.5])[1]
                if kind=='moment':
                    marginal_variance=(4+variance)*t*t+b*b
                    A=np.sqrt(marginal_variance/(marginal_variance+tau))
                    g=s-mixture(z/A,means,[v+tau]*2,[.5,.5])[1]/A
                z=z+h*(z+b*(s+g))/t
            target_p,_=mixture(np.linspace(-8,8,32001),[-2,2],[.25,.25],[.5,.5])
            axis=np.linspace(-8,8,32001);cdf=cumulative_trapezoid(target_p,axis,initial=0)
            expected=np.interp((np.arange(n)+.5)/n,cdf,axis)
            rows.append(dict(model_variance=variance,method=kind,n=n,steps=steps,
                quantile_w2=float(np.sqrt(np.mean((np.sort(z)-expected)**2))),
                bridge_mass=float(np.mean(np.abs(z)<1)),within_mode_variance=float(np.mean((np.abs(z)-2)**2)),
                mean=float(np.mean(z))))
            endpoints[f'{variance}_{kind}']=z
    write_csv(OUT/'ode.csv',rows);np.savez(OUT/'ode_endpoints.npz',**endpoints)
    return rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=exact_checks();snapshots();calibration=calibrate();flows=ode()
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(checks=checks,calibration=calibration,ode=flows),indent=2))

if __name__=='__main__':main()
