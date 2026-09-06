"""CPU-only algebra checks for a paper reading; no model, dataset or sampling run."""
import time

started, cpu_started = time.perf_counter(), time.process_time()
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad, solve_ivp

rng = np.random.default_rng(20260906)
d = 4
C = rng.normal(size=(d, d)); C = C @ C.T + np.eye(d)
U = rng.normal(size=(d, d)); U = U @ U.T + np.eye(d)
mu_c, mu_u, x = (rng.normal(size=d) for _ in range(3))
sigma = 1.3
I = np.eye(d)
A_c = np.linalg.solve(C + sigma**2 * I, C)
A_u = np.linalg.solve(U + sigma**2 * I, U)
D_c = mu_c + A_c @ (x-mu_c)
D_u = mu_u + A_u @ (x-mu_u)
split = (A_c-A_u) @ (x-mu_c) + (I-A_u) @ (mu_c-mu_u)
decomposition_error = float(np.max(np.abs(D_c-D_u-split)))
assert decomposition_error < 1e-12

# Appendix B: two commuting covariance modes, one positive and one negative.
c, u = np.array([4., 1.]), np.array([1., 3.])
mu_c, mu_u = np.array([.5, -.2]), np.array([-.1, .3])
x_T = np.array([2., -1.])
T, s, gamma = 8., .4, .75
def ode(q, value):
    ac, au = c/(c+q*q), u/(u+q*q)
    dc, du = mu_c+ac*(value-mu_c), mu_u+au*(value-mu_u)
    return (value-dc-gamma*(dc-du))/q

numeric = solve_ivp(ode, (T, s), x_T, method='DOP853', rtol=1e-11, atol=1e-12).y[:, -1]
h = (c+s*s)/(c+T*T) * (u+T*T)/(u+s*s)
multiplier = np.sqrt((c+s*s)/(c+T*T)) * h**(gamma/2)
b = []
for ci, ui in zip(c, u):
    p = lambda q: np.sqrt(ci+q*q)*((ci+q*q)/(ui+q*q))**(gamma/2)
    b.append(p(s)*quad(lambda q: q/((ui+q*q)*p(q)), s, T, epsabs=1e-12, epsrel=1e-12)[0])
b = np.asarray(b)
closed = mu_c+multiplier*(x_T-mu_c)+gamma*b*(mu_c-mu_u)
ode_error = float(np.max(np.abs(numeric-closed)))
assert ode_error < 1e-9 and h[0] > 1 and h[1] < 1

# A deterministic common translation changes no centered covariance.
cohort = rng.normal(size=(20, 2))*multiplier
translated = cohort+gamma*b*(mu_c-mu_u)
covariance_error = float(np.max(np.abs(np.cov(cohort, rowvar=False)-np.cov(translated, rowvar=False))))
assert covariance_error < 1e-12

# Exact symmetric two-atom Bayes denoiser: D(0)=0 is not sufficient for D=J_D*x.
q = .8
denoised = float(np.tanh(q))
jacobian_times_input = float(q/(np.cosh(q)**2))
assert abs(denoised-jacobian_times_input) > .1

result = {
    'scope': 'small deterministic CPU algebra verification; no learned models, dataset, image generation, GPU, or guidance parameter selection',
    'seed': 20260906,
    'noncommuting_covariance_commutator_norm': float(np.linalg.norm(C@U-U@C)),
    'exact_linear_gap_decomposition_max_abs_error': decomposition_error,
    'commuting_ode': {'conditional_eigenvalues': c.tolist(), 'unconditional_eigenvalues': u.tolist(),
        'sigma_start': T, 'sigma_end': s, 'fixed_gamma_for_identity_check': gamma,
        'h': h.tolist(), 'CPC_relative_scaling': (h**(gamma/2)).tolist(),
        'appendix_B_b': b.tolist(), 'closed_solution': closed.tolist(),
        'numeric_solution': numeric.tolist(), 'max_abs_error': ode_error},
    'common_translation_centered_covariance_max_abs_error': covariance_error,
    'stationary_origin_does_not_imply_local_homogeneity': {
        'prior': 'equal atoms at -1,+1, Gaussian noise sigma=1',
        'D_at_zero': 0., 'x': q, 'D_x': denoised, 'Jacobian_times_x': jacobian_times_input},
    'RAEv2_single_dense_FP32_jacobian_bytes': (1024*16*16)**2*4,
    'timing_scope': 'starts before numpy/scipy imports, ends before JSON output; no subprocesses',
    'wall_seconds_before_output': time.perf_counter()-started,
    'cpu_seconds_before_output': time.process_time()-cpu_started,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'all_assertions_passed': True,
}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
