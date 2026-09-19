"""Reproducible counterexamples and identities for the pasted FSG analysis.

These CPU calculations check mathematical claims, not neural image quality.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

WORK = Path('/home/zhoushunyu/eqvae')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/sit_fsg_pasted_followup_20260910')


def affine_flow(a, b, dt, x):
    dimension = len(x)
    augmented = np.zeros((dimension + 1, dimension + 1))
    augmented[:dimension, :dimension] = a
    augmented[:dimension, -1] = b
    return (expm(dt * augmented) @ np.r_[x, 1.])[:dimension]


def checks():
    result = {}
    # U=0 and C(t)=cos(2*pi*t) are smooth, invertible translation flows.
    # At time 0 both terminal maps equal x, but the suffix C map changes.
    times = np.linspace(0, 1, 4097)
    conditional_path = np.sin(2 * np.pi * times) / (2 * np.pi)
    suffix_residual = -conditional_path
    path_energy = np.trapezoid(conditional_path ** 2, times)
    assert abs(suffix_residual[0]) < 1e-15
    assert abs(suffix_residual[1024] + 1 / (2 * np.pi)) < 1e-15
    assert abs(path_energy - 1 / (8 * np.pi ** 2)) < 1e-12
    result['endpoint_equality_without_invariance'] = dict(
        v_u='0', v_c='cos(2*pi*t)', horizon=[0, 1], residual_t0=0.,
        residual_t025=float(suffix_residual[1024]),
        path_energy=float(path_energy), unconditional_endpoint_change=0.,
        implication='Rebound is a changed conditional counterfactual, not loss of the original U terminal output.')

    # R=C(t,x)-U(t,x), differentiated along the unconditional flow.
    rng = np.random.default_rng(202610089)
    errors = []
    for _ in range(128):
        ac, au = rng.normal(size=(2, 3, 3)) * .15
        bc, bu = rng.normal(size=(2, 3)) * .2
        x = rng.normal(size=3)
        t = rng.uniform(.1, .8)
        epsilon = 1e-5

        def residual(time, state):
            return (affine_flow(ac, bc, 1-time, state)
                    - affine_flow(au, bu, 1-time, state))

        plus = residual(t+epsilon, affine_flow(au, bu, epsilon, x))
        minus = residual(t-epsilon, affine_flow(au, bu, -epsilon, x))
        actual = (plus-minus)/(2*epsilon)
        expected = expm(ac*(1-t)) @ ((au-ac) @ x + bu-bc)
        errors.append(float(np.linalg.norm(actual-expected)))
        assert errors[-1] < 1e-8
    result['material_derivative_identity'] = dict(
        cases=128, max_error=max(errors),
        equation='(partial_t + v_u dot grad) R = D_x Phi_c(t,1) (v_u-v_c)',
        conditions='C1 flows with a common fixed terminal time; invertibility is needed only for the zero-gap corollary.')

    # A3's inverse response solves a frozen target; it need not solve R=0.
    c = lambda x: 3*x+1
    u = lambda x: x
    x, delta = 0., 1.
    assert u(x+delta) == c(x)
    assert abs(c(x+delta)-u(x+delta)) > abs(c(x)-u(x))
    result['frozen_target_is_not_same_start_root'] = dict(
        conditional='3*x+1', unconditional='x', x=x, inverse_delta=delta,
        frozen_target_error_after=0., same_start_residual_before=1.,
        same_start_residual_after=3., true_root_delta=-.5)

    # Spectral sensitivity amplifies vectors; loss gradients pull back by J^T.
    j = np.array([[2., 1.], [0., 3.]])
    residual = np.array([1., -2.])
    pullback = j.T @ residual
    inverse = np.linalg.solve(j, residual)
    assert not np.allclose(pullback, inverse)
    result['vector_and_covector'] = dict(jacobian=j.tolist(),
        gradient_pullback=pullback.tolist(), target_matching_inverse=inverse.tolist())

    # Same laws can disagree under a particular shared-noise coupling.
    normal = rng.normal(size=4096)
    normal = np.r_[normal, -normal]
    sample_c, sample_u = normal, -normal
    fixed_coupling = np.mean((sample_c-sample_u)**2)
    optimal_empirical = np.mean((np.sort(sample_c)-np.sort(sample_u))**2)
    assert fixed_coupling > 3.5 and optimal_empirical == 0.
    result['coupling_upper_bound_not_wasserstein_equality'] = dict(
        conditional='xi', unconditional='-xi', shared_noise_mse=float(fixed_coupling),
        empirical_w2_squared=float(optimal_empirical), laws_identical=True)

    # A positive intervention margin does not certify the target's semantics.
    target_future, null_future, rival_future = -1., -1., 1.
    margin = (null_future-rival_future)**2-(null_future-target_future)**2
    assert margin == 4. and target_future < 0
    result['contrastive_bad_root'] = dict(target_event='terminal > 0',
        conditional_future=target_future, unconditional_future=null_future,
        rival_future=rival_future, residual=0., margin=margin,
        target_success=False, neighborhood_agreement_also_zero=True)

    # The singular-value assumption in CFG-Ctrl does not imply Eq. (26).
    gamma = np.array([[-1.]])
    s = np.array([1.])
    k = 2.
    drift_bound = .5  # Strictly positive as required by the published assumption.
    actual_vdot = float(s @ gamma @ (-k*np.sign(s)))
    claimed_upper = (drift_bound-k*np.linalg.svd(gamma, compute_uv=False)[-1])*np.linalg.norm(s)
    assert actual_vdot > claimed_upper and actual_vdot > 0
    spd = np.array([[1., -2.], [-2., 5.]])
    s2 = np.array([100., 1.])
    assert np.linalg.eigvalsh(spd).min() > 0
    spd_vdot = float(s2 @ spd @ (-np.sign(s2)))
    assert spd_vdot > 0
    result['cfg_ctrl_sign_control_counterexample'] = dict(
        gamma=gamma.tolist(), s=s.tolist(), min_singular=1., k=k,
        drift_bound=drift_bound, actual_drift=0., actual_vdot=actual_vdot, claimed_upper_bound=float(claimed_upper),
        even_spd_gamma=spd.tolist(), spd_s=s2.tolist(), spd_actual_vdot=spd_vdot,
        scope='Refutes the stated singular-value-to-Lyapunov inequality, not the reported image experiments.')

    # If Q is P conditioned on a terminal class event, its distances from P
    # quantify remaining class uncertainty under that common joint law.
    distances = []
    for _ in range(64):
        p = rng.dirichlet(np.ones(8))
        event = np.arange(8) < 3
        success = p[event].sum()
        q = np.where(event, p/success, 0.)
        tv = .5*np.abs(p-q).sum()
        kl = np.sum(q[event]*np.log(q[event]/p[event]))
        assert abs(tv-(1-success)) < 1e-12
        assert abs(kl+math.log(success)) < 1e-12
        distances.append([float(success), float(tv), float(kl)])
    result['conditional_kernel_distances'] = dict(cases=64,
        tv='1-P_u(target|state)', kl='-log P_u(target|state)',
        assumption='P_c is the actual P_u kernel conditioned on the terminal class, not two arbitrary neural samplers.',
        examples=distances[:3])

    # Correct score / reverse-SDE conversion for X_t=t*Y+(1-t)*epsilon.
    conversion_errors = []
    for t in (.125, .25, .5, .75):
        variance = t*t + (1-t)**2
        x = rng.normal(size=32)
        score = -x/variance
        velocity = (2*t-1)*x/variance
        recovered_score = (t*velocity-x)/(1-t)
        reverse_drift = 2*velocity-x/t
        drift_from_score = velocity + ((1-t)/t)*score
        conversion_errors.append(float(np.max(np.abs(recovered_score-score))))
        assert np.allclose(recovered_score, score)
        assert np.allclose(reverse_drift, drift_from_score)
    result['linear_interpolant_sde_conversion'] = dict(
        max_score_error=max(conversion_errors), score='(t*v-x)/(1-t)',
        reverse_drift='2*v-x/t', diffusion_squared='2*(1-t)/t',
        domain='t>0; the planned SDE tail starts at t=1/8 to avoid the singular initial endpoint.')
    return dict(passed=True, checks=result, no_neural_quality_claim=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    result = checks()
    source = Path(__file__).resolve()
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    path = ROOT/'theory_checks.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
