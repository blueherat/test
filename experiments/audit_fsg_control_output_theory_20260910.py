"""Independent numerical checks of output selection and control identities.

CPU-only mathematical examples. These checks neither train a model nor claim
improved neural generation quality.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.special import expit, ndtr

WORK = Path('/home/zhoushunyu/eqvae')
ROOT = Path('/home/zhoushunyu/data/eqvae/experiments/fsg_control_output_theory_20260910')


def flow(a, b, h, x):
    n = len(x)
    augmented = np.zeros((n+1, n+1))
    augmented[:n, :n] = a
    augmented[:n, -1] = b
    return (expm(h*augmented) @ np.r_[x, 1.])[:n]


def checks():
    rng = np.random.default_rng(202610092)
    out = {}
    # Two different derivatives: changing look-ahead versus advancing the
    # controlled state while keeping the terminal time fixed.
    ac, au = rng.normal(size=(2, 3, 3))*.2
    bc, bu = rng.normal(size=(2, 3))*.3
    x = rng.normal(size=3)
    gap = (ac-au) @ x+bc-bu
    horizons = (1e-2, 1e-3, 1e-4, 1e-5)
    errors = [float(np.linalg.norm((flow(ac, bc, h, x)-flow(au, bu, h, x))/h-gap))
              for h in horizons]
    assert all(b < a*.12 for a, b in zip(errors, errors[1:]))
    out['horizon_derivative'] = dict(time_direction='noise to data, increasing t',
        formula='partial_h (Phi_c(t,t+h)-Phi_u(t,t+h)) at h=0 = v_c-v_u',
        horizons=horizons, errors=errors)

    fixed_errors, rolling_errors, omitted_boundary_errors, semantic_errors = [], [], [], []
    for _ in range(128):
        ac, au = rng.normal(size=(2, 3, 3))*.2
        bc, bu = rng.normal(size=(2, 3))*.3
        x = rng.normal(size=3); t = rng.uniform(.1, .6)
        b = rng.normal(size=(3, 2)); command = rng.normal(size=2)*.2
        intervention = b @ command
        gap = (ac-au) @ x+bc-bu
        epsilon = 1e-5
        plus = flow(au, bu+intervention, epsilon, x)
        minus = flow(au, bu+intervention, -epsilon, x)
        jc, ju = expm(ac*(1-t)), expm(au*(1-t))

        def residual(time, state, terminal=1.):
            return flow(ac, bc, terminal-time, state)-flow(au, bu, terminal-time, state)

        actual = (residual(t+epsilon, plus)-residual(t-epsilon, minus))/(2*epsilon)
        expected = -jc @ gap+(jc-ju) @ intervention
        fixed_errors.append(float(np.linalg.norm(actual-expected)))

        horizon = .3
        jc_h, ju_h = expm(ac*horizon), expm(au*horizon)
        end_c, end_u = flow(ac, bc, horizon, x), flow(au, bu, horizon, x)
        boundary = ac @ end_c+bc-(au @ end_u+bu)
        rolling_actual = (residual(t+epsilon, plus, t+epsilon+horizon)
                          -residual(t-epsilon, minus, t-epsilon+horizon))/(2*epsilon)
        incomplete = -jc_h @ gap+(jc_h-ju_h) @ intervention
        rolling_errors.append(float(np.linalg.norm(rolling_actual-incomplete-boundary)))
        omitted_boundary_errors.append(float(np.linalg.norm(rolling_actual-incomplete)))

        # A fixed semantic readout of the U terminal map is passively invariant.
        l = rng.normal(size=(2, 3))
        offset = rng.normal(size=2)
        target = np.array([.6, -.2])

        def semantic(time, state):
            return np.tanh(l @ flow(au, bu, 1-time, state)+offset)-target

        end_u = flow(au, bu, 1-t, x)
        derivative_psi = (1-np.tanh(l @ end_u+offset)**2)[:, None]*l
        semantic_actual = (semantic(t+epsilon, plus)-semantic(t-epsilon, minus))/(2*epsilon)
        semantic_expected = derivative_psi @ ju @ intervention
        semantic_errors.append(float(np.linalg.norm(semantic_actual-semantic_expected)))
    assert max(fixed_errors) < 1e-8
    assert max(rolling_errors) < 1e-8 and max(omitted_boundary_errors) > .1
    assert max(semantic_errors) < 1e-8
    out['controlled_fixed_terminal_residual'] = dict(cases=128, max_error=max(fixed_errors),
        formula='Rdot = -J_C(v_c-v_u) + (J_C-J_U) B u',
        consequence='The control response is (J_C-J_U)B, not J_C alone.')
    out['moving_terminal_boundary'] = dict(cases=128, max_correct_error=max(rolling_errors),
        max_error_without_boundary=max(omitted_boundary_errors),
        extra_term='Tprime(t) * [v_c(C,T)-v_u(U,T)]')
    out['fixed_semantic_target_output'] = dict(cases=128, max_error=max(semantic_errors),
        formula='edot = D Psi(U) J_U B u for e=Psi(U_t(x))-y_ref, fixed T,y_ref',
        consequence='No passive drift under the same deterministic U flow.')

    # Local consistency and terminal consistency are incomparable at one state.
    times = np.linspace(0, 1, 8193)
    local_zero_gap = times
    terminal_zero_gap = np.cos(2*np.pi*times)
    assert local_zero_gap[0] == 0 and abs(np.trapezoid(local_zero_gap, times)-.5) < 1e-12
    assert terminal_zero_gap[0] == 1 and abs(np.trapezoid(terminal_zero_gap, times)) < 1e-12
    out['no_local_endpoint_total_order'] = dict(
        local_zero_endpoint_nonzero=dict(v_c='t', v_u='0', initial_gap=0., terminal_residual=.5),
        endpoint_zero_local_nonzero=dict(v_c='cos(2*pi*t)', v_u='0', initial_gap=1., terminal_residual=0.),
        future_sensitivity_at_initial_time_zero_does_not_imply_current_residual_zero=True)

    # A full invertible endpoint Jacobian changes the metric, not the zero set.
    singular_bounds = []
    for _ in range(128):
        j = expm(rng.normal(size=(3, 3))*.3); g = rng.normal(size=3)
        least = np.linalg.svd(j, compute_uv=False)[-1]
        margin = np.linalg.norm(j @ g)-least*np.linalg.norm(g)
        assert margin >= -1e-12
        singular_bounds.append(float(least))
    out['invertible_future_jacobian_zero_set'] = dict(
        cases=128, min_singular_across_cases=min(singular_bounds),
        implication='J_C*g=0 iff g=0 for invertible J_C; approximate weighting can still differ.',
        scope='Smooth finite-time ODE flow. Semantic projection or singular limits need separate analysis.')

    # Residual vanishes because the remaining interval vanishes, with no
    # guidance, no state controllability of R and an incorrect terminal output.
    t = np.linspace(0, 1, 1025)
    residual = 1-t; value = .5*residual**2; derivative = -residual
    assert np.all(derivative <= -2*value+1e-14)
    initial_x = -2.
    for state in rng.normal(size=128):
        assert abs(((state+.7)-state)-.7) < 1e-14
    out['terminal_collapse_and_uncontrollable_residual'] = dict(
        v_c='1', v_u='0', terminal_time=1., residual='1-t',
        V='(1-t)^2/2', Vdot='-(1-t)', decay_bound='Vdot <= -2V on [0,1]',
        state_jacobian_of_R=0., initial_x=initial_x, actual_null_terminal=initial_x,
        target_event='terminal > 0', target_success=False,
        horizon_normalized_residual=1.,
        consequence='Decay of this full-future residual does not certify task success.')

    # Exact Gaussian conditional-expectation velocities have a legitimate
    # nonzero class/null gap even without model approximation error.
    relation_errors = []
    for schedule in ('linear', 'vp_angle'):
        for time in (.125, .25, .5, .75, .875):
            if schedule == 'linear':
                alpha, sigma, adot, sdot = time, 1-time, 1., -1.
            else:
                theta = .1+time*1.2
                alpha, sigma = math.cos(theta), math.sin(theta)
                adot, sdot = -math.sin(theta), math.cos(theta)
            mean, variance_data = 1., .7
            variance = alpha**2*variance_data+sigma**2
            x = rng.normal(size=64)
            prob = expit(2*alpha*mean*x/variance)
            def conditional_velocity(label):
                expected_data = label*mean+alpha*variance_data/variance*(x-alpha*label*mean)
                expected_noise = sigma/variance*(x-alpha*label*mean)
                return adot*expected_data+sdot*expected_noise
            positive, negative = conditional_velocity(1), conditional_velocity(-1)
            null = prob*positive+(1-prob)*negative
            log_prob_gradient = 2*alpha*mean/variance*(1-prob)
            coefficient = sigma*sigma*adot/alpha-sigma*sdot
            relation_errors.append(float(np.max(np.abs(positive-null-coefficient*log_prob_gradient))))
    assert max(relation_errors) < 1e-12
    out['perfect_model_class_gap'] = dict(
        identity='v_c-v_u = (sigma^2*alpha_prime/alpha - sigma*sigma_prime) * grad log p_t(c|x)',
        assumption='Canonical conditional-expectation velocity for the same Gaussian interpolation and joint law.',
        cases=10, max_error=max(relation_errors),
        exact_example=dict(class_means=[-1, 1], within_class_variance=1., t=.5, x=0.,
                           conditional_velocity=1., null_velocity=0., posterior_target=.5,
                           model_approximation_error=0.))

    # Consistent epsilon and velocity parameterizations differ by a signed,
    # time-dependent factor; matching errors and their decay rates is different.
    errors = []
    for _ in range(128):
        alpha, sigma = rng.uniform(.1, 1., size=2)
        adot, sdot = rng.normal(size=2)
        delta_eps = rng.normal(size=8)
        delta_data = -sigma/alpha*delta_eps
        actual = adot*delta_data+sdot*delta_eps
        expected = (sdot-adot*sigma/alpha)*delta_eps
        errors.append(float(np.max(np.abs(actual-expected))))
    assert max(errors) < 1e-12
    t = np.linspace(0, 1, 21)
    assert np.all(np.diff(np.exp(-t)) < 0) and np.all(np.diff(np.exp(2*t)*np.exp(-t)) > 0)
    out['parameterization_and_decay'] = dict(
        cases=128, max_velocity_conversion_error=max(errors),
        epsilon_factor='sigma_prime - alpha_prime*sigma/alpha',
        scaled_error_derivative='(k e)dot = kdot e + k edot',
        rate_counterexample='e=exp(-t), k=exp(2t), k*e=exp(t)',
        warning='Zero-set equivalence does not allow copying gains, time signs or Lyapunov rates.')

    # Same one-time DDPM marginals, different cross-time couplings.
    abar_s, abar_t = .8, .5
    shared = math.sqrt((1-abar_s)*(1-abar_t))
    markov = math.sqrt(abar_t/abar_s)*(1-abar_s)
    assert abs(shared-markov) > .1
    bridge = np.array([[2., 1.], [0., 1.]])
    covariance = bridge @ bridge.T
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    circle = np.stack((np.cos(times*2*np.pi), np.sin(times*2*np.pi)))
    ellipse = eigenvectors.T @ bridge @ circle
    assert np.max(np.abs((ellipse**2/eigenvalues[:, None]).sum(0)-1)) < 1e-12
    out['vp_marginal_coupling_is_not_ddpm_path'] = dict(
        alpha_bar_s=abar_s, alpha_bar_t=abar_t,
        covariance_using_one_shared_epsilon=shared, covariance_true_forward_markov=markov,
        fixed_pair_state_plane_ellipse_axes=np.sqrt(eigenvalues).tolist(),
        consequence='A coefficient-circle coupling is not an individual Markov DDPM noise trajectory.')

    # Linear training interpolations can yield an exactly curved marginal ODE.
    target_variances = np.array([4., .25]); noise = np.ones(2)
    trajectory = lambda time: np.sqrt(time*time*target_variances+(1-time)**2)*noise
    first, middle, last = trajectory(0.), trajectory(.5), trajectory(1.)
    cross = float(np.linalg.det(np.stack((middle-first, last-first))))
    assert abs(cross) > .1
    out['straight_bridges_curved_exact_marginal_flow'] = dict(
        training_path='X_t=t*Y+(1-t)*epsilon with independent Gaussian endpoints',
        target_variances=target_variances.tolist(), initial=first.tolist(),
        midpoint=middle.tolist(), terminal=last.tolist(), noncollinearity_determinant=cross,
        exact_velocity='v_i=(t*variance_i-(1-t))/(t^2*variance_i+(1-t)^2)*x_i',
        model_approximation_error=0.)

    # A bound on a disturbance is not a zero-tracking theorem.
    gain, disturbance, end = 2., .4, 20.
    steady = disturbance/gain
    actual = steady*(1-math.exp(-gain*end))
    assert abs(actual-steady) < 1e-12 and actual > 0.
    commands = np.linspace(-3, 3, 1001)
    minimax = np.maximum(np.abs(commands-1), np.abs(commands+1))
    assert minimax.min() >= 1.
    out['bounded_disturbance_and_unobserved_ideal'] = dict(
        scalar_system='edot=-k*e+d, constant d', k=gain, d=disturbance,
        steady_tracking_error=steady, error_at_t20=actual,
        learned_field='v_hat=0 in both worlds', compatible_ideal_fields=['+1', '-1'],
        error_bound_in_both_worlds=1., minimax_unidentified_bias_error=1.,
        scope='Ideal-field errors hidden behind the same learned sampler are not measured physical disturbances.')

    gain = 2.
    scale = math.exp(-gain)
    out['stable_state_can_collapse_correct_distribution'] = dict(
        desired_distribution='N(0,1)', control='xdot=-2*x', initial_distribution='N(0,1)',
        terminal_variance=scale**2, terminal_w2_squared=(1-scale)**2,
        state_exponentially_stable=True, correct_target_law=False)

    # A bounded-input CLF inequality has an explicit feasibility condition.
    errors = []; feasible = 0
    for _ in range(256):
        e, drift = rng.normal(size=(2, 3))
        response = rng.normal(size=(3, 2))
        q = response.T @ e
        bound, disturbance_bound, rate = .7, .2, 1.
        v = .5*np.dot(e, e)
        minimum = np.dot(e, drift)+disturbance_bound*np.linalg.norm(e)-bound*np.linalg.norm(q)
        minimizing_input = -bound*q/np.linalg.norm(q)
        achieved = np.dot(e, drift+response @ minimizing_input)+disturbance_bound*np.linalg.norm(e)
        errors.append(abs(float(minimum-achieved)))
        feasible += minimum <= -rate*v
    assert max(errors) < 1e-12
    out['bounded_input_output_feasibility'] = dict(
        cases=256, max_error=max(errors), feasible_random_examples=int(feasible),
        formula='e^T*a + D*||e|| - Umax*||(D_x e B)^T e|| <= -lambda*V',
        scope='Illustrates the exact pointwise robust CLF feasibility test, not a neural controllability claim.')

    # For Brownian motion conditioned on a positive terminal point, the
    # committor is harmonic for the passive generator and improves in expectation
    # under the exact Doob transform; its paths need not be monotone.
    errors = []; rates = []
    for time in (.1, .3, .6, .8):
        for state in (-1., -.5, 0., .5, 1.):
            h = 1-time; z = state/math.sqrt(h)
            density = math.exp(-z*z/2)/math.sqrt(2*math.pi)
            probability = ndtr(z)
            qt = .5*state*density/(h**1.5)
            qx = density/math.sqrt(h)
            qxx = -state*density/(h**1.5)
            errors.append(abs(qt+.5*qxx))
            conditioned_drift = qx/probability
            rate = qt+conditioned_drift*qx+.5*qxx
            assert abs(rate-qx*qx/probability) < 1e-12
            assert rate > 0
            rates.append(float(rate))
    assert max(errors) < 1e-12
    out['conditional_success_committor'] = dict(
        event='Brownian X_1>0', q='NormalCDF(x/sqrt(1-t))',
        passive_equation='(partial_t+L_u)q=0',
        conditioned_drift='b_c=b_u+a*grad(log q)',
        conditioned_probability_drift='grad(q)^T a grad(q)/q >= 0',
        cases=20, max_passive_generator_error=max(errors), min_conditioned_drift=min(rates),
        scope='Common true SDE and exact terminal event; not a claim about a clean-image classifier or arbitrary neural C/U pair.')
    return dict(passed=True, checks=out, no_neural_quality_claim=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    result = checks()
    result['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = ROOT/'theory_checks.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
