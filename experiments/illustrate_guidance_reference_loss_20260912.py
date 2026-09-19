"""Analytic population examples of restricted reference loss; no model probes."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/data/guidance_reference_loss_20260912'


def density(z,m,sigma=1.):
    z=np.asarray(z)
    logp=logsumexp(np.stack((-.5*((z-m)/sigma)**2,-.5*((z+m)/sigma)**2)),axis=0)
    return np.exp(logp-np.log(2*sigma*np.sqrt(2*np.pi)))


def score(z,m,sigma=1.):
    return -(z-m*np.tanh(m*z/(sigma*sigma)))/(sigma*sigma)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.1),layout='constrained')
    records=[];curves=[]
    for ax,m,a in zip(axes,(2.,.8),(0.,2.)):
        variance=1+m*m
        ezscore=quad(lambda z:z*score(z,m)*density(z,m),-np.inf,np.inf,epsabs=1e-11)[0]
        ez2=quad(lambda z:z*z*density(z,m),-np.inf,np.inf,epsabs=1e-11)[0]
        coefficient=ezscore/ez2
        np.testing.assert_allclose(ezscore,-1.,atol=1e-10)
        np.testing.assert_allclose(coefficient,-1/variance,atol=1e-10)
        fourth=quad(lambda z:z**4*density(z,m),-np.inf,np.inf,epsabs=1e-10)[0]
        cumulant=fourth-3*variance**2
        np.testing.assert_allclose(cumulant,-2*m**4,atol=1e-9)
        z=np.linspace(-6 if m==2 else -3.6,6 if m==2 else 3.6,2001)
        p=density(z,m);q=np.exp(-z*z/(2*variance))/np.sqrt(2*np.pi*variance)
        ax.plot(z,p,color='#1c658c',lw=2.3,label='Strong / exact target p')
        ax.plot(z,q,color='#bd8633',lw=2.1,ls='--',label='Affine-score fit q')
        if a:
            def raw(x):
                lp=float(logsumexp([-.5*(x-m)**2,-.5*(x+m)**2])-np.log(2*np.sqrt(2*np.pi)))
                lq=-x*x/(2*variance)-.5*np.log(2*np.pi*variance)
                return np.exp((1+a)*lp-a*lq)
            normalizer=quad(raw,-np.inf,np.inf,epsabs=1e-11)[0]
            powered=np.array([raw(x)/normalizer for x in z])
            ax.plot(z,powered,color='#98516b',lw=2,label=r'Fixed-time $p^{3}/q^{2}$ (normalized)')
            original_curvature=m*m-1
            powered_curvature=(1+a)*(m*m-1)+a/variance
            threshold=m**-4-1
            assert original_curvature<0 and powered_curvature>0 and a>threshold
            ax.set_title('Even an exact unimodal p can be split',fontsize=12)
            ax.text(.03,.94,f'Center log-curvature: {original_curvature:.3f} → {powered_curvature:.3f}',
                transform=ax.transAxes,va='top',fontsize=9)
        else:
            powered=np.full_like(z,np.nan);normalizer=None;original_curvature=None;powered_curvature=None;threshold=None
            ax.set_title('A broad reference need not be Gaussian blur',fontsize=12)
            ax.text(.03,.94,r'Same variance; different fourth cumulant',transform=ax.transAxes,va='top',fontsize=9)
        records.append(dict(mean=m,sigma=1.,score_fit_coefficient=coefficient,expected_coefficient=-1/variance,
            fourth_cumulant_p=cumulant,fourth_cumulant_q=0.,extra_guidance=a,
            original_center_log_curvature=original_curvature,powered_center_log_curvature=powered_curvature,
            splitting_threshold=threshold,powered_normalizer=normalizer))
        curves.extend(dict(mean=m,x=float(x),p=float(vp),q=float(vq),powered=None if np.isnan(vg) else float(vg))
            for x,vp,vq,vg in zip(z,p,q,powered))
        ax.set_xlabel('State x');ax.set_ylabel('Density');ax.grid(alpha=.14);ax.set_ylim(bottom=0)
        ax.legend(loc='upper right' if m==2 else 'lower center',fontsize=8,frameon=False)
    fig.suptitle('Restricted score fitting: exact one-dimensional examples',fontsize=14)
    for suffix in ('png','pdf','svg'):fig.savefig(OUT/f'score_projection.{suffix}',dpi=190)
    plt.close(fig)
    with (OUT/'score_projection_curves.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['mean','x','p','q','powered']);writer.writeheader();writer.writerows(curves)
    result=dict(passed=True,records=records,gaussian_convolution_preserves_fourth_cumulant=True,
        caveat='Fixed-noise-level normalized score potentials, not actual guided ODE endpoint densities. Not an identification of an IG head.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'score_projection_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
