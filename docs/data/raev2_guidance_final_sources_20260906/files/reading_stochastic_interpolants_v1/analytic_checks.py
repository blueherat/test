"""Exact scalar checks of the SI JMLR formulas; no sampling/model/GPU."""
import json
import math
from pathlib import Path
import time

started = time.perf_counter()
# x0,x1,z iid N(0,1), I=(1-t)x0+t*x1, gamma^2=2t(1-t).
# rho_t=N(0,1), b=0, score=-x; learned b=1, learned score=-x+1.
times = [0., .1, .5, .9, 1.]
variances = [(1-t)**2+t*t+2*t*(1-t) for t in times]
terminal_mean = 2*(1-math.exp(-1))
terminal_kl = .5*terminal_mean**2
delta_velocity_loss = delta_score_loss = .5
printed_bound = .5*delta_velocity_loss+.5*delta_score_loss
correct_bound = delta_velocity_loss+delta_score_loss
assert max(abs(v-1) for v in variances) < 3e-16
assert printed_bound < terminal_kl < correct_bound
# Reverse progress u=1-t, epsilon(t)=1+t, rho_t=N(0,1), b=0.
# Correct reversed OU drift is -(2-u)x. At u=0 and V=1:
correct_reversed_variance_derivative = -2*2+2*2
printed_reversed_variance_derivative = -2*2+2*1
assert correct_reversed_variance_derivative == 0
assert printed_reversed_variance_derivative == -2
result = {
    'kind': 'analytic deterministic scalar calculations, not an RAE experiment',
    'gaussian_excess_risk_constant': {
        'times': times, 'target_variances': variances,
        'epsilon': 1, 'exact_b': 0, 'exact_score': '-x',
        'learned_b': 1, 'learned_score': '-x+1',
        'learned_forward_SDE': 'dX=(-X+2)dt+sqrt(2)dW, X0~N(0,1)',
        'terminal_mean': terminal_mean, 'terminal_variance': 1,
        'KL_target_to_learned': terminal_kl,
        'delta_Lb': delta_velocity_loss, 'delta_Ls': delta_score_loss,
        'paper_equation_2_45_RHS': printed_bound,
        'corrected_RHS_and_Lemma22_bound': correct_bound,
        'printed_bound_violated': True,
    },
    'nonsymmetric_reverse_epsilon': {
        'epsilon_original': '1+t', 'epsilon_reversed': '2-u',
        'initial_variance_derivative_correct': correct_reversed_variance_derivative,
        'initial_variance_derivative_using_paper_2_35_noise_argument': printed_reversed_variance_derivative,
    },
    'calls': {'GPU': 0, 'model_forward': 0, 'training': 0, 'samples_generated': 0},
    'compute_wall_seconds_excluding_imports_and_write': time.perf_counter()-started,
}
output = Path(__file__).with_name('analytic_checks.json')
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
