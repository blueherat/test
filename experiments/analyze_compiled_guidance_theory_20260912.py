"""Exact finite-posterior identities and explicit limits; no image-quality fit."""
from pathlib import Path
import json
import numpy as np
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/data/reference_compilation_20260912'


def posterior(z,points,prior,t):
    z=np.asarray(z);points=np.asarray(points)
    likelihood=-((z[...,None,:]-t*points)**2).sum(-1)/(2*(1-t)**2)
    return softmax(likelihood+np.log(prior),axis=-1)


def power(p,q,a):
    return softmax((1+a)*np.log(p)-a*np.log(q),axis=-1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    points=np.array([[-1.],[1.]])
    p=np.array([.25,.75]);q=np.array([.5,.5]);a=1.25;t=.5
    x=np.linspace(-2,2,801)[:,None]
    pp,qp=posterior(x,points,p,t),posterior(x,points,q,t)
    r0=power(p,q,a);exact=posterior(x,points,r0,t);combined=power(pp,qp,a)
    identity=float(np.max(np.abs(exact-combined)))
    assert identity<1e-14
    m_p=(pp@points)[:,0];m_q=(qp@points)[:,0]
    m_linear=(1+a)*m_p-a*m_q;m_probability=(combined@points)[:,0]
    assert np.max(m_linear)>1 and np.max(np.abs(m_probability))<=1
    # Full joint posterior combination and coordinate-wise combination differ.
    points2=np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
    p2=np.array([.4,.1,.1,.4]);q2=np.full(4,.25);z2=np.array([[.25,-.1]])
    pp2=posterior(z2,points2,p2,.4)[0];qp2=posterior(z2,points2,q2,.4)[0]
    joint=power(pp2,qp2,a);joint_mean=joint@points2;local_means=[]
    for coordinate in range(2):
        mask=points2[:,coordinate]>0
        pi=np.array([pp2[~mask].sum(),pp2[mask].sum()]);qi=np.array([qp2[~mask].sum(),qp2[mask].sum()])
        local_means.append(float(power(pi,qi,a)@np.array([-1.,1.])))
    marginal_error=float(np.linalg.norm(joint_mean-local_means))
    assert marginal_error>.01
    # Equal-variance Gaussian posteriors close under probability powers.
    mu_p,mu_q,var=1.,-.4,.7
    gaussian_mean=(1+a)*mu_p-a*mu_q
    grid=np.linspace(-12,12,100001)
    logp=-(grid-mu_p)**2/(2*var);logq=-(grid-mu_q)**2/(2*var)
    numerical=softmax((1+a)*logp-a*logq)@grid
    assert abs(gaussian_mean-numerical)<1e-12
    records=dict(passed=True,full_posterior_commutation_max_error=identity,
        extrapolated_mean_max=float(m_linear.max()),probability_mean_max=float(m_probability.max()),
        tempered_endpoint_weights=r0.tolist(),joint_posterior_mean=joint_mean.tolist(),
        marginal_combination_mean=local_means,marginal_joint_mean_error=marginal_error,
        gaussian_equal_variance_mean_error=float(abs(gaussian_mean-numerical)),
        scope='exact finite clean support and constant strength; not the patch-readout image sampler')
    (OUT/'posterior_identity_checks.json').write_text(json.dumps(records,indent=2))
    np.savetxt(OUT/'posterior_mean_curves.csv',np.column_stack([x[:,0],m_p,m_q,m_linear,m_probability]),
        delimiter=',',header='noisy_state,conditional_mean,null_mean,linear_guidance,probability_guidance',comments='')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    ax=axes[0];ax.axhspan(-1,1,color='.94')
    ax.plot(x[:,0],m_p,label='Conditional mean',color='.35')
    ax.plot(x[:,0],m_linear,label='Extrapolate means',color='#b4513b')
    ax.plot(x[:,0],m_probability,label='Combine posteriors, then average',color='#275c83')
    ax.axhline(1,color='.6',ls=':',lw=1);ax.axhline(-1,color='.6',ls=':',lw=1)
    ax.set(xlabel='Noisy observation z',ylabel='Predicted clean value',title='Full clean candidates: {-1, +1}')
    ax.legend(fontsize=8,loc='lower right')
    ax=axes[1];positions=np.arange(2);width=.34
    ax.bar(positions-width/2,joint_mean,width,label='Combine full joint posterior',color='#275c83')
    ax.bar(positions+width/2,local_means,width,label='Combine coordinate marginals',color='#b4513b')
    ax.axhline(0,color='.7',lw=.8);ax.set(xticks=positions,xticklabels=['Coordinate 1','Coordinate 2'],
        ylabel='Predicted clean mean',title='Marginals do not preserve the joint identity')
    ax.legend(fontsize=8)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for suffix in ('png','pdf','svg'):fig.savefig(OUT/f'posterior_mean_identity.{suffix}',dpi=180)
    plt.close(fig);print(json.dumps(records))


if __name__=='__main__':main()
