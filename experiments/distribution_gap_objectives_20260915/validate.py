"""CPU-only algebra/counterexample checks; no image model quality experiment."""

import json
from pathlib import Path
import numpy as np
import torch

OUT = Path(__file__).resolve().parents[2] / 'docs/data/distribution_gap_objectives_20260915'


def run():
    torch.set_num_threads(2)
    torch.manual_seed(2026091507)
    rng = np.random.default_rng(2026091507)
    result = {'kind': 'mathematical_validation_not_image_training'}

    # Equal-variance Gaussian family: P=N(0,1), G=N(1,1), R=N(m,1).
    m = np.linspace(-4, 3, 70001)
    squared, metric = {}, {}
    for weight in (.2, .5, .8):
        loss = m*m-weight*(m-1)**2
        actual = float(m[np.argmin(loss)])
        expected = -weight/(1-weight)
        assert abs(actual-expected)<1.1e-4
        squared[str(weight)] = {'grid_minimizer': actual, 'analytic_minimizer': expected}
        loss = np.abs(m)-weight*np.abs(m-1)
        actual = float(m[np.argmin(loss)])
        assert abs(actual)<1e-8
        metric[str(weight)] = {'grid_minimizer': actual}
    result['squared_distance_repulsion'] = squared
    result['true_metric_repulsion'] = metric
    # Even the metric theorem need not preserve the closest feasible model.
    candidates = np.array([.1,-.2])
    losses = np.abs(candidates)-.5*np.abs(candidates-1)
    assert int(np.argmin(losses)) != int(np.argmin(np.abs(candidates)))
    result['restricted_family_counterexample'] = {'means': candidates.tolist(), 'losses': losses.tolist()}

    # Triangle bound for Euclidean probability embeddings on random finite supports.
    p, g, r = rng.dirichlet(np.ones(5),size=(3,2000))
    dpr=np.linalg.norm(p-r,axis=1)
    dgr=np.linalg.norm(g-r,axis=1)
    dpg=np.linalg.norm(p-g,axis=1)
    margin=dpr-.7*dgr+.7*dpg-(1-.7)*dpr
    assert margin.min()>-1e-12
    result['metric_triangle_bound_min_slack']=float(margin.min())

    # KL value AND gradient: density ratio is held fixed during the generator step.
    p=torch.tensor([.2,.3,.5],dtype=torch.float64)
    g=torch.tensor([.6,.3,.1],dtype=torch.float64)
    theta=torch.randn(3,dtype=torch.float64,requires_grad=True)
    r=theta.softmax(0)
    true_loss=(r*(r.log()-p.log())).sum()
    hpg=(p/g).log()
    hrg=(r.detach()/g).log()
    surrogate=(r*(hrg-hpg)).sum()
    actual_grad=torch.autograd.grad(surrogate,theta,retain_graph=True)[0]
    true_grad=torch.autograd.grad(true_loss,theta)[0]
    torch.testing.assert_close(actual_grad,true_grad,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(surrogate,true_loss,rtol=1e-12,atol=1e-12)
    result['kl_anchor_identity']={'value_abs_error':float(abs(surrogate-true_loss).detach()),
                                  'gradient_max_abs_error':float((actual_grad-true_grad).abs().max())}
    tilted={}
    for beta in (.5,1.,2.):
        target=((p.log()+(beta-1)*g.log())/beta).softmax(0)
        if beta==1:torch.testing.assert_close(target,p)
        tilted[str(beta)]=target.tolist()
    result['kl_target_by_beta']=tilted
    result['static_ratio_reward_only']={'p':p.tolist(),'g':g.tolist(),
        'reward':hpg.tolist(),'maximizing_delta_atom':int(hpg.argmax()),
        'real_distribution_is_not_reward_maximizer':float((p*hpg).sum())<float(hpg.max())}

    # With shared four-way logits, all ratio cycles telescope algebraically.
    logits=torch.randn(100,4,dtype=torch.float64)
    hpg=logits[:,0]-logits[:,1]
    hrg=logits[:,3]-logits[:,1]
    hpr=logits[:,0]-logits[:,3]
    error=float((hpg-hrg-hpr).abs().max())
    assert error<1e-12
    result['shared_logit_cycle_max_abs_error']=error

    # Kernel mean geometry, here checked in a finite Hilbert feature space.
    mu_p,mu_g,mu_r=rng.normal(size=(3,23))
    a,b=mu_p-mu_g,mu_r-mu_g
    gain=2*a@b-b@b
    expected=np.dot(a,a)-np.dot(mu_p-mu_r,mu_p-mu_r)
    assert np.allclose(gain,expected,rtol=1e-12,atol=1e-12)
    result['directional_gain_identity_abs_error']=float(abs(gain-expected))

    # Does a paired strong reference help estimation? One explicit example only.
    # psi(x)=[x,x^2], P=N(0,1), G=N(1,1), R=N(.8,1), paired via shared epsilon.
    trials,n,mean=20000,24,.8
    eps=rng.standard_normal((trials,n))
    xr,xg=eps+mean,eps+1
    fr=np.stack((xr,xr*xr),axis=-1)
    fg=np.stack((xg,xg*xg),axis=-1)
    deriv=np.stack((np.ones_like(xr),2*xr),axis=-1)
    a=np.array([-1.,-1.])
    d=fr-fg
    e=fr-np.array([0.,1.])
    def ugradient(values):
        total=(deriv.sum(1)*values.sum(1)).sum(1)
        diagonal=(deriv*values).sum((1,2))
        return 2*(total-diagonal)/(n*(n-1))
    paired=ugradient(d)-2*(deriv.mean(1)*a).sum(1)
    ordinary=ugradient(e)
    truth=2*mean+4*mean**3
    result['paired_control_variate_example']={'trials':trials,'batch':n,'true_gradient':truth,
        'paired_mean':float(paired.mean()),'ordinary_mean':float(ordinary.mean()),
        'paired_std':float(paired.std(ddof=1)),'ordinary_std':float(ordinary.std(ddof=1)),
        'variance_ratio':float(paired.var()/ordinary.var()),
        'warning':'One quadratic-feature Gaussian example; no universal variance or image-quality claim.'}
    for values in (paired,ordinary):
        assert abs(values.mean()-truth)<6*values.std(ddof=1)/np.sqrt(trials)
    vpaired=2*((d.mean(1)-a)*deriv.mean(1)).sum(1)
    vordinary=2*(e.mean(1)*deriv.mean(1)).sum(1)
    result['naive_squared_batch_mean_gradient_bias']={
        'paired_observed':float(vpaired.mean()-truth),'paired_expected':8*(mean-1)/n,
        'ordinary_observed':float(vordinary.mean()-truth),'ordinary_expected':8*mean/n}

    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    plot()
    print(json.dumps(result,indent=2))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=np.linspace(-2.5,2,1000)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,squared in zip(axes,(True,False)):
        y=m*m-.5*(m-1)**2 if squared else np.abs(m)-.5*np.abs(m-1)
        best=-1 if squared else 0
        best_y=best*best-.5*(best-1)**2 if squared else abs(best)-.5*abs(best-1)
        ax.plot(m,y,color='#2474a6',lw=2)
        ax.axvline(0,color='#159a77',ls='--',label='Real P: mean 0')
        ax.axvline(1,color='#989898',ls=':',label='Strong G: mean 1')
        ax.scatter([best],[best_y],color='#dc6144',s=50,zorder=5,label='Loss minimizer')
        ax.set_title('Squared W2: overshoots to mean -1' if squared else 'W2 metric: minimum remains at P')
        ax.set_xlabel('Mean m of R = N(m, 1)')
        ax.set_ylabel('Attraction to P - 0.5 x repulsion from G')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Same reference, same weight: the definition of distance changes the answer',fontsize=11)
    fig.savefig(OUT/'metric_vs_squared.png',dpi=170)
    fig.savefig(OUT/'metric_vs_squared.pdf')
    plt.close(fig)


if __name__=='__main__':
    run()
