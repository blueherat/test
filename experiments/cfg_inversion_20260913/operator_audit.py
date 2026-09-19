"""CPU exact-flow audit for conditional-inverse / high-CFG compositions.

Reproduce: python experiments/cfg_inversion_20260913/operator_audit.py
Uses matrix exponentials for affine time-dependent fields. No GPU/model calls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm, expm_frechet
from numpy.polynomial.legendre import leggauss


def norm(x):
    return float(np.linalg.norm(x))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='/home/zhoushunyu/data/eqvae/experiments/cfg_inversion_20260913/operator_audit')
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    # Augment physical time and a constant: Y=(x_1,x_2,t,1).
    # A=(1,A_spatial), G=(0,g_spatial), B=A+gamma*G.
    A=np.zeros((4,4));G=np.zeros((4,4))
    A[:2,:2]=[[.2,.8],[-.5,-.1]]; A[:2,2]=[.3,-.2];A[:2,3]=[.15,-.1];A[2,3]=1.
    G[:2,:2]=[[.4,-.3],[.7,.2]];G[:2,2]=[-.1,.25];G[:2,3]=[-.2,.05]
    gamma=.6;theta=.4;B=A+gamma*G
    y=np.array([.4,-.7,.3,1.]);I=np.eye(4);rows=[]
    for h in [.08,.04,.02,.01,.005]:
        C=expm(h*A);H=expm(h*B);Ci=expm(-h*A)
        r=Ci @ H @ y
        a=(A@y)[:2];g=(G@y)[:2];d=gamma*g
        ja=A[:2,:2];jg=G[:2,:2];gt=G[:2,2]
        bracket=jg@a-ja@g
        first=y[:2]+h*d
        second=first+h*h/2*(gamma*gt+gamma*bracket+gamma*gamma*jg@g)
        # Reverse order is based at end time, and its input is C(y).
        z=C@y;rend=H@Ci@z
        ae=(A@z)[:2];ge=(G@z)[:2];be=jg@ae-ja@ge
        reverse_second=z[:2]+h*gamma*ge+h*h/2*(-gamma*gt-gamma*be+gamma*gamma*jg@ge)
        bad=(I-h*A)@(I+h*B)@y
        selfbad=(I-h*A)@(I+h*A)@y
        unbiased=bad-selfbad+y
        discrete=np.linalg.solve(I+h*A,(I+h*B)@y)
        discrete_second=y[:2]+h*d-h*h*ja@d
        reinjected_A=C@r
        repeated_H=H@r
        effective=expm(h*(2*B-A))@y
        lifted=C@(y+theta*(r-y))
        effective_partial=expm(h*(A+theta*gamma*G))@y
        partial_lead=h*h/2*theta*(1-theta)*gamma*gamma*jg@g
        # A time-dependent pure-gap comparator advances the external clock.
        Dclock=gamma*G.copy();Dclock[2,3]=1.
        puregap=expm(h*Dclock)@y
        puregap_diff=r[:2]-puregap[:2]
        expected_comm=h*h/2*gamma*bracket
        rows.append(dict(h=h,loop_first_error=norm(r[:2]-first),loop_second_error=norm(r[:2]-second),
                         reverse_second_error=norm(rend[:2]-reverse_second),
                         naive_self_cycle=norm(selfbad[:2]-y[:2]),
                         naive_self_cycle_formula_error=norm(selfbad[:2]-y[:2]+h*h*(A@A@y)[:2]),
                         discrete_inverse_second_error=norm(discrete[:2]-discrete_second),
                         self_bias_subtraction_vs_discrete=norm(unbiased[:2]-discrete[:2]),
                         exact_cancellation_error=norm(reinjected_A-H@y),
                         repeated_high_vs_effective_error=norm(repeated_H[:2]-effective[:2]),
                         partial_lifting_error=norm(lifted[:2]-effective_partial[:2]),
                         partial_lifting_formula_remainder=norm(lifted[:2]-effective_partial[:2]-partial_lead),
                         loop_minus_pure_gap_commutator_remainder=norm(puregap_diff-expected_comm),
                         loop_clock_error=abs(float(r[2]-y[2])),repeated_clock_error=abs(float(repeated_H[2]-(y[2]+h)))))
    with (out/'orders.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    orderkeys=['loop_first_error','loop_second_error','reverse_second_error','naive_self_cycle',
               'discrete_inverse_second_error','self_bias_subtraction_vs_discrete','repeated_high_vs_effective_error',
               'partial_lifting_error','partial_lifting_formula_remainder','loop_minus_pure_gap_commutator_remainder']
    orders={k:float(np.log(rows[-2][k]/rows[-1][k])/np.log(2)) for k in orderkeys}
    h=.12;C=expm(h*A);H=expm(h*B);Ci=expm(-h*A)
    derivative=Ci@expm_frechet(h*B,h*G,compute_expm=False)@y
    eps=1e-5
    fd=Ci@(expm(h*(A+(gamma+eps)*G))-expm(h*(A+(gamma-eps)*G)))@y/(2*eps)
    derivative_zero=Ci@expm_frechet(h*A,h*G,compute_expm=False)@y
    nodes,weights=leggauss(48)
    integrated=np.zeros(4)
    for node,weight in zip(nodes,weights):
        s=h*(node+1)/2
        integrated += weight*h/2*(expm(-s*A)@G@expm(s*A)@y)
    R=Ci@H;residual=(R@y-y)[:2];jac=R[:2,:2]-np.eye(2)
    loss_gradient=jac.T@residual
    loss_fd=[]
    for j in range(2):
        delta=np.zeros(4);delta[j]=eps
        rp=(R@(y+delta)-(y+delta))[:2];rm=(R@(y-delta)-(y-delta))[:2]
        loss_fd.append((np.dot(rp,rp)-np.dot(rm,rm))/(4*eps))
    # Nonlinear Euler inverse: A(x)=a*x^2, g(x)=b.
    aa=.4;bb=.8;x=.6;hh=.2;gg=.7
    inverse=lambda strength:(np.sqrt(1+4*hh*aa*(x+hh*(aa*x*x+strength*bb)))-1)/(2*hh*aa)
    u=inverse(gg);implicit_derivative=hh*bb/(1+2*hh*aa*u)
    implicit_fd=(inverse(gg+eps)-inverse(gg-eps))/(2*eps)
    # Compatible Gaussian FM: C~N(0,1), X|C=c~N(c,1), target c=0.
    # Exact conditional variance=1, exact null variance=2.
    t=.2;delta=.15;cfgextra=.8
    variance=lambda s,v:(1-s)**2+v*s*s
    ca=np.sqrt(variance(t+delta,1)/variance(t,1))
    un=np.sqrt(variance(t+delta,2)/variance(t,2))
    high=ca**(1+cfgextra)/un**cfgextra
    compatible=dict(time=t,h=delta,gamma=cfgextra,conditional_flow_scale=float(ca),null_flow_scale=float(un),
                    high_flow_scale=float(high),exact_loop_scale=float(high/ca),
                    loop_displacement_at_x1=float(high/ca-1),
                    full_generation_conditional_variance=1.,full_generation_high_variance=float(2**(-cfgextra)),
                    endpoint_power_target_variance=float(1/(1+cfgextra/2)),
                    interpretation='Both conditional and null fields are exact. Cross-field loop displacement is not model error.')
    Fa=expm(.12*(A+.2*G));Fb=expm(.12*(A+.7*G));Fc=expm(.12*(A+1.3*G))
    Rba=np.linalg.solve(Fb,Fa);Rab=np.linalg.solve(Fa,Fb);Rcb=np.linalg.solve(Fc,Fb);Rca=np.linalg.solve(Fc,Fa)
    cocycle=dict(same_reference_identity_error=norm(np.linalg.solve(Fa,Fa)-I),
                 reverse_reference_identity_error=norm(Rab@Rba-I),
                 chained_reference_error=norm(Rcb@Rba-Rca),
                 convention='R_ba=F_b^{-1}F_a; all F share the same physical interval and augmented state.')
    checks=dict(orders=orders,constant_gamma=gamma,constant_theta=theta,
                finite_gamma_derivative_error=norm((derivative-fd)[:2]),
                gamma_zero_pullback_integral_error=norm((derivative_zero-integrated)[:2]),
                cycle_loss_gradient_error=norm(loss_gradient-np.array(loss_fd)),
                nonlinear_euler_inverse=dict(output=float(u),derivative=float(implicit_derivative),fd=float(implicit_fd),error=abs(float(implicit_derivative-implicit_fd))),
                exact_cancellation_max=max(r['exact_cancellation_error'] for r in rows),
                compatible_gaussian=compatible,A_augmented=A.tolist(),gap_augmented=G.tolist(),initial_state=y.tolist(),
                reference_cocycle=cocycle,
                cpu_only=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert checks['exact_cancellation_max']<1e-12
    assert checks['finite_gamma_derivative_error']<1e-9
    assert checks['gamma_zero_pullback_integral_error']<1e-12
    assert checks['cycle_loss_gradient_error']<1e-10
    assert abs(orders['loop_second_error']-3)<.1
    assert abs(orders['repeated_high_vs_effective_error']-3)<.1
    assert abs(orders['partial_lifting_formula_remainder']-3)<.1
    assert cocycle['chained_reference_error']<1e-12
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
