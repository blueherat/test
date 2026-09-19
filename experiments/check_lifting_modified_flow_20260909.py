"""Independent CPU checks of fractional lifting's second-order flow expansion.

These tests characterize the operator, not image quality or research success.
SciPy DOP853 evaluates smooth nonlinear, explicitly time-dependent exact-flow
proxies, independently of the GPU Heun/Euler implementations being sampled.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUT=Path(__file__).resolve().parents[1]/'docs/data/lifting_modified_flow_20260909'
AS=np.array([[.3,-.6],[.2,-.4]])
AW=np.array([[-.1,.5],[.65,.2]])
BS=np.array([.2,-.3]);BW=np.array([-.15,.1])
CS=np.array([.1,-.05]);CW=np.array([-.05,.07])


def strong(t,x):return np.tanh(AS@x+BS*t)+CS
def weak(t,x):return np.tanh(AW@x+BW*t)+CW
def gap(t,x):return strong(t,x)-weak(t,x)


def gap_jacobian(t,x):
    return (1-np.tanh(AS@x+BS*t)**2)[:,None]*AS-(1-np.tanh(AW@x+BW*t)**2)[:,None]*AW


def flow(field,x,t,u):
    result=solve_ivp(field,(t,u),x,method='DOP853',rtol=2e-13,atol=1e-14)
    assert result.success and np.isfinite(result.y).all()
    return result.y[:,-1]


def lift(x,t,h,alpha,m):
    z=x.copy()
    for _ in range(m):
        target=flow(strong,z,t,t+h)
        inverse=flow(weak,target,t+h,t)
        z=z+(alpha/m)*(inverse-z)
    return flow(strong,z,t,t+h)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(202609949)
    states=rng.uniform(-1,1,(8,2));times=rng.uniform(.15,.65,8)
    alphas=[.25,.5,1.,1.5,2.];hs=[.08,.04,.02,.01,.005]
    rows=[]
    for direction in (1,-1):
        for index,(x,t) in enumerate(zip(states,times)):
            jdd=gap_jacobian(t,x)@gap(t,x)
            for alpha in alphas:
                for m in (1,2):
                    coefficient=.5*(alpha-alpha*alpha/m)
                    for h_abs in hs:
                        h=direction*h_abs
                        actual=lift(x,t,h,alpha,m)
                        exact=flow(lambda t,z:strong(t,z)+alpha*gap(t,z),x,t,t+h)
                        difference=actual-exact
                        predicted=h*h*coefficient*jdd
                        rows.append(dict(direction=direction,state=index,alpha=alpha,m=m,h=h_abs,
                            predicted_coefficient=coefficient,
                            estimated_coefficient=float(difference@jdd/(h*h*(jdd@jdd))),
                            raw_error=float(np.linalg.norm(difference)),
                            corrected_error=float(np.linalg.norm(difference-predicted))))
    with (OUT/'nonlinear_checks.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    summaries=[]
    for direction in (1,-1):
        for alpha in alphas:
            for m in (1,2):
                subset=[r for r in rows if r['direction']==direction and r['alpha']==alpha and r['m']==m]
                averages={h:{kind:np.mean([r[kind] for r in subset if r['h']==h])
                              for kind in ('raw_error','corrected_error','estimated_coefficient')} for h in hs}
                corrected_order=float(np.log2(averages[.01]['corrected_error']/averages[.005]['corrected_error']))
                raw_order=float(np.log2(averages[.01]['raw_error']/averages[.005]['raw_error']))
                expected_raw_order=3 if alpha==m else 2
                summaries.append(dict(direction=direction,alpha=alpha,m=m,
                    predicted_coefficient=.5*(alpha-alpha*alpha/m),
                    estimated_at_smallest_h=averages[.005]['estimated_coefficient'],
                    corrected_order=corrected_order,raw_order=raw_order,
                    expected_raw_order=expected_raw_order,
                    passed=bool(corrected_order>2.85 and abs(raw_order-expected_raw_order)<.15)))
    strengths=np.linspace(0,2,201)
    a,b,target_std=.8,.4,1.8
    ratio=a/b
    std_lift=a*(1+strengths*(ratio-1))
    effective=np.log1p(strengths*(ratio-1))/np.log(ratio)
    std_matched=a*ratio**effective
    std_ig=a*ratio**strengths
    equivalence_error=float(np.max(np.abs(std_lift-std_matched)))
    np.savez(OUT/'gaussian_reparameterization.npz',alpha=strengths,effective_alpha=effective,
             standard_deviation_lifting=std_lift,standard_deviation_ig=std_ig,
             standard_deviation_matched_ig=std_matched,target_standard_deviation=target_std)
    result=dict(passed=all(r['passed'] for r in summaries) and equivalence_error<1e-12,
        trials=len(rows),seed=202609949,states=states.tolist(),times=times.tolist(),
        strong_matrix=AS.tolist(),weak_matrix=AW.tolist(),strong_time=BS.tolist(),weak_time=BW.tolist(),
        strong_offset=CS.tolist(),weak_offset=CW.tolist(),summaries=summaries,
        gaussian_exact_equivalence_max_error=equivalence_error,
        minimum_corrected_order=min(r['corrected_order'] for r in summaries),
        scope='Smooth exact-flow operator checks; no neural-model, image-quality, or novelty claim.',
        research_goal_achieved=False)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for m,color in [(1,'#2a628f'),(2,'#c35a26')]:
        values=[r for r in summaries if r['direction']==1 and r['m']==m]
        axes[0].plot(alphas,[r['predicted_coefficient'] for r in values],color=color,label=f'm={m}, predicted')
        axes[0].scatter(alphas,[r['estimated_at_smallest_h'] for r in values],color=color,s=24)
    axes[0].axhline(0,color='#666666',lw=.7);axes[0].set(xlabel='Total extra strength',ylabel='Second-order coefficient',title='Nonlinear time-dependent fields')
    axes[0].legend(frameon=False,fontsize=9)
    for m,alpha,color in [(1,.5,'#2a628f'),(1,1.,'#c35a26'),(2,2.,'#34775d')]:
        subset=[r for r in rows if r['direction']==1 and r['m']==m and r['alpha']==alpha]
        values=[np.mean([r['raw_error'] for r in subset if r['h']==h]) for h in hs]
        axes[1].loglog(hs,values,'o-',ms=3,color=color,label=f'm={m}, strength={alpha:g}')
    axes[1].set(xlabel='Signed-time interval magnitude',ylabel='Difference from exact IG flow',title='Order increases at strength = m')
    axes[1].legend(frameon=False,fontsize=9)
    axes[2].plot(strengths,(std_ig-target_std)**2,color='#2a628f',label='IG, same nominal strength')
    axes[2].plot(strengths,(std_lift-target_std)**2,color='#c35a26',label='Lifting')
    axes[2].plot(strengths,(std_matched-target_std)**2,'--',color='#34775d',label='IG, exact matched strength')
    axes[2].set(xlabel='Nominal extra strength',ylabel='Squared Gaussian Wasserstein distance',title='Pointwise gains can be reparameterization')
    axes[2].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'operator_checks.png',dpi=180);fig.savefig(OUT/'operator_checks.pdf');plt.close(fig)
    print(json.dumps(dict(passed=result['passed'],trials=len(rows),
        minimum_corrected_order=result['minimum_corrected_order'],
        gaussian_equivalence_error=equivalence_error)),flush=True)
    assert result['passed'],summaries


if __name__=='__main__':main()
