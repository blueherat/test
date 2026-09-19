"""CPU-only analytic checks of CFG invariants and their counterexamples.

No fitted model, image-quality metric, or parameter search is used here.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parents[1]
OUT = WORK/'docs/data/cfg_invariants_20260913'
CTRL = WORK/'readings/fsg_ctrl_mp_comparison_20260911/code/cfg_ctrl/pipeline/common_cfg_ctrl.py'


def normalize(p):
    return p/p.sum()


def mixture(z, t):
    priors = np.array([.2, .35, .45])
    means = t*np.array([[-1.2, .1], [.7, 1.1], [1., -.8]])
    cov0 = np.array([[[.6, .1], [.1, .9]], [[1.1, -.2], [-.2, .7]], [[.8, .25], [.25, .8]]])
    cov = t*t*cov0 + (1-t)**2*np.eye(2)[None]
    precision = np.linalg.inv(cov)
    delta = z[None]-means
    logp = np.log(priors)-.5*np.linalg.slogdet(cov)[1]-.5*np.einsum('ki,kij,kj->k', delta, precision, delta)
    weights = normalize(np.exp(logp-logp.max()))
    scores = -np.einsum('kij,kj->ki', precision, delta)
    null = weights@scores
    gaps = scores-null[None]
    covariance = np.einsum('k,ki,kj->ij', weights, gaps, gaps)
    jac_null = np.einsum('k,kij->ij', weights, -precision)+covariance
    jac_gaps = -precision-jac_null[None]
    return weights, scores, null, gaps, jac_gaps, covariance


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(cpu_only=True, is_quality_experiment=False)
    # A. Fixed-state CFG algebra.
    u, c = np.array([.3, -.5]), np.array([2.3, .5])
    cfg = lambda w, v, base: base+w*(v-base)
    a, b = 1.5, 2.
    A, shift = np.array([[2., .3], [-.4, 1.2]]), np.array([-.7, .4])
    g = c-u
    perpendicular = np.array([g[1], -g[0]])/np.linalg.norm(g)
    result['algebra'] = dict(
        common_complement_error=float(abs(perpendicular@(cfg(b,c,u)-u))),
        affine_equivariance_error=float(np.max(np.abs(cfg(b,A@c+shift,A@u+shift)-(A@cfg(b,c,u)+shift)))),
        multiplicative_composition_error=float(np.max(np.abs(cfg(b,cfg(a,c,u),u)-cfg(a*b,c,u)))),
        idempotence_counterexample_distance=float(np.linalg.norm(cfg(b,cfg(b,c,u),u)-cfg(b,c,u))))
    assert max(result['algebra'][k] for k in ('common_complement_error','affine_equivariance_error','multiplicative_composition_error')) < 1e-12
    assert result['algebra']['idempotence_counterexample_distance'] > 1

    # B. Exact finite-distribution conditioning and likelihood fibers.
    prior = np.array([.10, .20, .15, .05, .30, .20])
    likelihood = np.array([.8, .8, .2, .2, .05, .05])
    event = np.array([1., 1., 1., 0., 0., 0.])
    hard = normalize(prior*event)
    soft = normalize(prior*likelihood)
    twice = normalize(soft*likelihood)
    fiber_rows = []
    maximum_error = 0.
    for w in (0., 1., 2., 4.):
        tilted = normalize(prior*likelihood**w)
        for k in range(3):
            idx = slice(2*k,2*k+2)
            original_fiber = normalize(prior[idx])
            new_fiber = normalize(tilted[idx])
            maximum_error = max(maximum_error, float(np.max(np.abs(original_fiber-new_fiber))))
            fiber_rows.append(dict(w=w,fiber=k,first_conditional_mass=new_fiber[0],second_conditional_mass=new_fiber[1],total_fiber_mass=tilted[idx].sum()))
    result['conditioning'] = dict(hard_idempotence_error=float(np.max(np.abs(normalize(hard*event)-hard))),
        repeated_soft_likelihood_l1=float(np.abs(twice-soft).sum()),
        hard_positive_power_max_error=float(max(np.max(np.abs(normalize(prior*event**w)-hard)) for w in (.5,1.,2.,4.))),
        likelihood_fiber_conditional_max_error=maximum_error,
        prior=prior.tolist(),likelihood=likelihood.tolist(),soft_once=soft.tolist(),soft_twice=twice.tolist())
    assert maximum_error < 1e-12
    assert result['conditioning']['repeated_soft_likelihood_l1'] > .1
    # Same terminal tilt, passed through one fixed class-independent corruption.
    # Rows are clean states; columns are two possible noisy observations.
    kernel=np.array([[.8,.2],[.6,.4],[.5,.5],[.35,.65],[.2,.8],[.1,.9]])
    noisy=prior@kernel
    clean_given_noisy=prior[:,None]*kernel/noisy[None,:]
    q=likelihood@clean_given_noisy
    h2=(likelihood**2)@clean_given_noisy
    variance=(likelihood[:,None]**2*clean_given_noisy).sum(axis=0)-q*q
    exact_tilt_noisy=normalize(prior*likelihood**2)@kernel
    noisy_via_conditional_moment=normalize(noisy*h2)
    naive_noisy_tilt=normalize(noisy*q*q)
    result['terminal_tilt_corruption']=dict(
        tower_max_error=float(np.max(np.abs(noisy*q-(prior*likelihood)@kernel))),
        w2_variance_identity_max_error=float(np.max(np.abs(h2-(q*q+variance)))),
        exact_noisy_tilt_max_error=float(np.max(np.abs(exact_tilt_noisy-noisy_via_conditional_moment))),
        naive_power_of_mean_l1=float(np.abs(exact_tilt_noisy-naive_noisy_tilt).sum()),
        h2=h2.tolist(),q_squared=(q*q).tolist(),posterior_variance=variance.tolist(),
        note='Finite class-independent corruption checks the conditional-expectation algebra, not an ODE terminal-distribution claim.')
    assert result['terminal_tilt_corruption']['exact_noisy_tilt_max_error'] < 1e-12
    assert result['terminal_tilt_corruption']['naive_power_of_mean_l1'] > .01

    # C. Exact, mutually compatible class-conditional Gaussian FM marginals.
    probability_rows=[]
    for t in (.15,.5,.9):
        for x in (-1.1,.2,1.3):
            for y in (-.7,.4):
                z=np.array([x,y])
                weights,scores,null,gaps,jac_gaps,covariance=mixture(z,t)
                first=weights@gaps
                second=np.einsum('k,kij->ij',weights,jac_gaps)+covariance
                h=1e-5
                finite_jac=np.stack([(mixture(z+np.eye(2)[i]*h,t)[3]-mixture(z-np.eye(2)[i]*h,t)[3])/(2*h) for i in range(2)],axis=-1)
                probability_rows.append(dict(t=t,x=x,y=y,weighted_gap_max=float(np.max(np.abs(first))),
                    second_order_identity_max=float(np.max(np.abs(second))),
                    jacobian_finite_difference_error=float(np.max(np.abs(finite_jac-jac_gaps)))))
    result['bayes'] = dict(states=len(probability_rows),
        first_order_max=max(r['weighted_gap_max'] for r in probability_rows),
        second_order_max=max(r['second_order_identity_max'] for r in probability_rows),
        independent_finite_difference_max=max(r['jacobian_finite_difference_error'] for r in probability_rows))
    assert result['bayes']['second_order_max'] < 1e-12
    assert result['bayes']['independent_finite_difference_max'] < 1e-7
    # Perturb conditional scores by B(z-z*) and leave null unchanged: at z*
    # values still agree with ground truth, but derivatives are incompatible.
    weights,_,_,gaps,jac_gaps,covariance=mixture(np.array([.2,.4]),.5)
    B=np.diag([.2,-.15])
    residual=np.einsum('k,kij->ij',weights,jac_gaps+B[None])+covariance
    result['bayes']['pointwise_first_order_zero_but_second_order_wrong'] = dict(
        first_order_max=float(np.max(np.abs(weights@gaps))),
        second_order_residual=residual.tolist(),expected_B=B.tolist())
    np.testing.assert_allclose(residual,B,atol=1e-12)
    guided_residual=2*np.einsum('k,kij->ij',weights,jac_gaps)+4*covariance
    np.testing.assert_allclose(guided_residual,2*covariance,atol=1e-12)
    result['bayes']['w2_guided_gap_residual_max']=float(np.max(np.abs(guided_residual)))
    result['bayes']['guided_gap_must_not_be_tested_against_original_posterior']=True

    # D. Source-repository CFG-Ctrl: exact coordinate/memory behavior.
    spec=importlib.util.spec_from_file_location('cfg_ctrl_invariant_audit',CTRL)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    controller=module.CFGCtrlMixin()
    params=module.CFGCtrlParams(smc_cfg_enable=True,smc_cfg_lambda=.05,smc_cfg_K=.3)
    def control(pos,neg,state=None):
        if state is None: state=module.CFGCtrlState()
        output=controller._cfg_ctrl_apply(noise_pred_posi=torch.tensor(pos,dtype=torch.float64),
            noise_pred_nega=torch.tensor(neg,dtype=torch.float64),cfg_scale=2.,progress_id=0,params=params,state=state)
        return output.numpy(),state
    zero=np.zeros(2); raw=np.array([2.,1.]); direction=np.array([1.,-2.])/np.sqrt(5)
    theta=.7; Q=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    direct,state=control(raw,zero)
    rotated,_=control(Q@raw,Q@zero)
    scaled,_=control(2*raw,zero)
    memory_output,_=control(zero,zero,state)
    result['controller'] = dict(source=str(CTRL),source_sha256=hashlib.sha256(CTRL.read_bytes()).hexdigest(),
        raw_gap=raw.tolist(),first_output=direct.tolist(),
        common_complement_leakage=float(direction@direct),
        rotation_reexpression_error=float(np.linalg.norm(Q.T@rotated-direct)),
        same_K_scale_reexpression_error=float(np.linalg.norm(scaled/2-direct)),
        zero_raw_gap_with_nonzero_history_output=memory_output.tolist(),
        interpretation='Differences from standard CFG algebra; not a quality verdict or universal physical impossibility.')
    assert abs(result['controller']['common_complement_leakage']) > .1
    assert result['controller']['rotation_reexpression_error'] > .1

    # E. Exact ODE endpoint invariance versus posterior-mean prediction.
    trajectory=[]
    for t in (0.,.25,.5,.75,1.):
        scale=np.sqrt(t*t+(1-t)**2)
        z=scale*1.2
        trajectory.append(dict(t=t,z=z,posterior_clean_mean=t*z/(scale*scale),remaining_endpoint=z/scale))
    result['trajectory'] = dict(rows=trajectory,
        endpoint_max_error=float(max(abs(r['remaining_endpoint']-1.2) for r in trajectory)),
        posterior_clean_mean_is_constant=False)
    # At pure noise, identical scores do not imply identical FM velocities.
    result['pure_noise_endpoint'] = dict(z=.4,conditional_mean=1.,unconditional_mean=0.,
        conditional_score=-.4,unconditional_score=-.4,
        conditional_velocity=.6,unconditional_velocity=-.4,velocity_gap=1.)
    # F. Local unchanged coordinate can change later through common dynamics.
    # vu=(0,x1), vc=(1,x1), both start at (0,0).
    result['local_vs_trajectory'] = dict(local_gap=[1.,0.],conditional_endpoint=[1.,.5],cfg_w2_endpoint=[2.,1.],
        explanation='The second velocity component is shared at the same state; first-coordinate intervention changes its later input.')
    # G. An exact posterior can lose its potential under norm-based feedback.
    # p(c|x)=.5 exp(-.5*(x1^2+2*x2^2)) is valid for a binary label and any
    # smooth base density. Its log-posterior gradient is (-x1,-2*x2).
    z=np.array([1.,1.]); h=1e-5
    def posterior_gap(x): return -np.array([x[0],2*x[1]])
    def normalized(x):
        d=posterior_gap(x)
        return d/np.linalg.norm(d)
    def value_control(x):
        ell=np.log(.5)-.5*(x[0]**2+2*x[1]**2)
        return np.exp(ell)*posterior_gap(x)
    def jacobian(f):
        return np.stack([(f(z+h*np.eye(2)[i])-f(z-h*np.eye(2)[i]))/(2*h) for i in range(2)],axis=-1)
    j_norm,j_value=jacobian(normalized),jacobian(value_control)
    result['potential_feedback'] = dict(
        normalized_gap_antisymmetric_01=float((j_norm-j_norm.T)[0,1]),
        normalized_gap_expected_01=float(2/(5**1.5)),
        posterior_value_feedback_antisymmetric_max=float(np.max(np.abs(j_value-j_value.T))))
    assert abs(result['potential_feedback']['normalized_gap_antisymmetric_01']-2/(5**1.5)) < 1e-8
    assert result['potential_feedback']['posterior_value_feedback_antisymmetric_max'] < 1e-8
    # H. Density preservation is stricter than total probability conservation.
    rotation=np.array([[0.,-1.],[1.,0.]])
    points=np.array([[1.,1.],[2.,-.5],[-.2,.8]])
    rotation_defects=[np.trace(rotation)+(rotation@x)@(-x) for x in points]
    radial_defects=[2-x@x for x in points]
    result['density_preservation']=dict(standard_gaussian=True,
        rotation_weighted_divergence_max=float(np.max(np.abs(rotation_defects))),
        radial_weighted_divergence=radial_defects,
        note='Both ODEs push forward normalized measures; only rotation preserves this reference Gaussian density.')
    for name,rows in [('likelihood_fibers',fiber_rows),('bayes_identities',probability_rows),('endpoint_vs_posterior',trajectory)]:
        with (OUT/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
