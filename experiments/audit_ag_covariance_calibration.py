"""Exact Gaussian flows: does noncommuting error geometry help calibration?"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def root(a):
    val, vec = np.linalg.eigh((a+a.T)/2)
    assert val.min() > 0
    return (vec*np.sqrt(val))@vec.T


def w2(a, b):
    r = root(a)
    return max(0., float(np.trace(a+b-2*root(r@b@r))))


def flow(cov, high, low):
    return root(cov+low**2*np.eye(2))@np.linalg.inv(root(cov+high**2*np.eye(2)))


def guided(strong, weak, delta=0., early=False, tight=False):
    eye = np.eye(2)
    state = eye.copy()
    for high, low in [(3., 2.), (2., 0.)]:
        gamma = .78+(delta if high == 3. or not early else 0.)
        def rhs(t, flat):
            generator = t*((1+gamma)*np.linalg.inv(strong+t*t*eye)-gamma*np.linalg.inv(weak+t*t*eye))
            return (generator@flat.reshape(2, 2)).ravel()
        sol = solve_ivp(rhs, (high, low), state.ravel(), method='DOP853',
                        rtol=1e-12 if tight else 1e-10, atol=1e-13 if tight else 1e-11)
        assert sol.success
        state = sol.y[:, -1].reshape(2, 2)
    return state


def main():
    true = np.diag([.25**2, 1.])
    angle = np.pi/4
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rows = []
    for aligned in [True, False]:
        error = np.diag([.36, .04])
        if not aligned:
            error = rotate@error@rotate.T
        for amplitude in [.25, 1., 4.]:
            strong = true+amplitude*error
            weak = true+2*amplitude*error
            initial = strong+9*np.eye(2)
            calibration = flow(weak, 2., 3.)@flow(strong, 3., 2.)
            calibration = calibration@calibration
            baseline = guided(strong, weak)
            candidate = baseline@calibration
            cov_base = baseline@initial@baseline.T
            cov_candidate = candidate@initial@candidate.T
            roundtrip = flow(strong, 2., 3.)@flow(strong, 3., 2.)
            assert np.max(np.abs(roundtrip-np.eye(2))) < 1e-12
            exact = guided(strong, weak, delta=-.78)
            assert np.max(np.abs(exact@initial@exact.T-strong)) < 1e-8
            fine = guided(strong, weak, tight=True)@calibration
            assert np.max(np.abs(fine-candidate)) < 1e-8
            changes = float(np.trace((candidate-baseline)@initial@(candidate-baseline).T))
            fixed = guided(strong, weak, delta=2., early=True)
            fixed_diff = fixed-candidate
            fixed_residual = float(np.trace(fixed_diff@initial@fixed_diff.T))
            if aligned:
                assert np.max(np.abs(fixed_diff)) < 1e-8
            controls = []
            for early in [False, True]:
                def loss(delta):
                    diff = guided(strong, weak, delta, early)-candidate
                    return float(np.trace(diff@initial@diff.T))
                fit = minimize_scalar(loss, bounds=(0., 4.), method='bounded', options={'xatol': 1e-7})
                assert fit.success
                control = guided(strong, weak, fit.x, early)
                cov_control = control@initial@control.T
                controls.append(dict(early_only=early, delta=float(fit.x), paired_displacement_explained=1-fit.fun/changes,
                                     w2_to_true=w2(cov_control, true), w2_to_candidate=w2(cov_control,cov_candidate),
                                     boundary_solution=bool(fit.x<1e-5 or fit.x>4-1e-5)))
            row = dict(aligned=aligned, error_amplitude=amplitude, strong_w2=w2(strong,true),
                       ordinary_ag_w2=w2(cov_base,true), calibrated_ag_w2=w2(cov_candidate,true),
                       controls=controls, covariance_commutator_norm=float(np.linalg.norm(strong@weak-weak@strong)),
                       fixed_delta_2_explained=1-fixed_residual/changes,
                       fixed_delta_2_map_max_error=float(np.max(np.abs(fixed_diff))),
                       calibrated_covariance=cov_candidate.tolist())
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = dict(complete=True, rows=rows, protocol=dict(true_covariance=true.tolist(),
                  error_eigenvalues=[.36,.04], error_amplitudes=[.25,1.,4.], rotations=[0.,45.],
                  strong='true + amplitude * error', weak='true + 2 * amplitude * error',
                  initial='exact strong sigma=3 marginal', calibration='strong 3->2 then weak 2->3, twice',
                  continued_ag_gamma=.78, scope='Oracle two-dimensional Gaussian mechanism check; no neural or novelty claim. Scalar controls fit paired endpoints, not true quality. All six cases reported.'))
    path = Path(__file__).resolve().parents[1]/'experiments/results/terminal_defect_20260908/ag_covariance_calibration.json'
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
