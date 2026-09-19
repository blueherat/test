from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.sit_bayes_risk_reference_20260912.checks import action,analytic_checks

OUT=Path(__file__).resolve().parents[1]/'docs/data/bayes_risk_reference_20260912'


def posterior(sigma,y=.2):
    means=np.array([-1.5,.7]);variances=np.array([.35,.55]);prior=np.array([.35,.65])
    v=variances+sigma**2
    density=prior*np.exp(-.5*(y-means)**2/v)/np.sqrt(2*np.pi*v)
    weights=density/density.sum();pm=(sigma**2*means+variances*y)/v;pv=variances*sigma**2/v
    mean=weights@pm;variance=weights@(pv+(pm-mean)**2)
    third=weights@((pm-mean)**3+3*(pm-mean)*pv)
    delta=2*np.sqrt(variance)*np.sinh(np.arcsinh(third/(2*variance**1.5))/3)
    scores=(means-y)/v
    q1=weights@scores;q2=weights@(scores**2-1/v);q3=weights@(scores**3-3*scores/v)
    score_second=q3-3*q1*q2+2*q1**3
    assert abs(third-sigma**6*score_second)<1e-12
    return delta,score_second,variance,third


def main():
    OUT.mkdir(parents=True,exist_ok=True);checks=analytic_checks()
    edge=np.geomspace(1e-12,.01,80)
    probabilities=np.unique(np.concatenate([np.linspace(0.,1.,401),edge,1-edge]));curves=[]
    for p in probabilities:
        mean=2*p-1;fourth=action([-1,1],[1-p,p]);guided=mean+.8*(mean-fourth)
        curves.append(dict(probability=p,square_action=mean,quartic_action=fourth,guided_action=guided))
    sigmas=np.geomspace(.02,.5,80);small=[]
    for sigma in sigmas:
        delta,second,variance,third=posterior(sigma)
        small.append(dict(sigma=sigma,delta=delta,score_second=second,variance=variance,third=third,
            velocity_gap=abs(delta)*(1+sigma)/sigma,leading_velocity_gap=abs(sigma**3*(1+sigma)*second/3)))
    checks.update(low_noise_relative_asymptotic_error=abs(small[0]['velocity_gap']/small[0]['leading_velocity_gap']-1),
        guided_overshoots_two_point_support=bool(max(abs(r['guided_action']) for r in curves)>1),
        quartic_point_nine=action([-1,1],[.1,.9]))
    assert checks['low_noise_relative_asymptotic_error']<.01 and checks['guided_overshoots_two_point_support']
    for name,rows in [('two_point_curves',curves),('low_noise_curves',small)]:
        with (OUT/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'theory_checks.json').write_text(json.dumps(checks,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11.5,4))
    ax=axes[0]
    for key,label,color in [('square_action','Squared-loss action','.3'),('quartic_action','Quartic-loss action','#275c83'),('guided_action','Extrapolation, a = 0.8','#ac5a36')]:
        ax.plot(probabilities,[r[key] for r in curves],label=label,color=color,lw=1.8)
    for y in (-1,1):ax.axhline(y,color='.6',ls=':',lw=1)
    ax.set(xlabel='Posterior probability of +1',ylabel='Predicted clean coordinate',title='Two candidates: exact odds flattening')
    ax.legend(frameon=False,fontsize=8)
    axes[1].loglog(sigmas,[r['velocity_gap'] for r in small],color='#275c83',label='Exact Bayes-action gap')
    axes[1].loglog(sigmas,[r['leading_velocity_gap'] for r in small],color='.35',ls='--',label='Low-noise leading term')
    axes[1].set(xlabel='Effective Gaussian noise sigma',ylabel='Absolute FM velocity gap',title='Smooth prior: gap vanishes at low noise')
    axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(OUT/f'bayes_actions.{ext}',dpi=170)
    plt.close(fig);print(json.dumps(checks))


if __name__=='__main__':main()
