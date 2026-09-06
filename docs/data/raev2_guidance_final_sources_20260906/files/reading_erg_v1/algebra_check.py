"""CPU-only checks of fixed finite-dimensional identities; no model or tuning."""
import time

started_wall, started_cpu = time.perf_counter(), time.process_time()
import hashlib
import json
import math
from pathlib import Path


def dot(x, y):
    return sum(a*b for a, b in zip(x, y))


def softmax(logits):
    pivot = max(logits)
    exp = [math.exp(a-pivot) for a in logits]
    total = sum(exp)
    return [a/total for a in exp]


def energy(q, keys, alpha, c):
    logits = [c*dot(k, q) for k in keys]
    pivot = max(logits)
    lse = pivot + math.log(sum(math.exp(a-pivot) for a in logits))
    return .5*dot(q, q)-alpha*lse/c


def grad(q, keys, alpha, c):
    p = softmax([c*dot(k, q) for k in keys])
    return [q[d]-alpha*sum(pj*k[d] for pj, k in zip(p, keys))
            for d in range(len(q))]


keys = [[1., 0.], [-1., .5], [.2, 1.], [0., -1.]]
q, alpha, c, epsilon = [.3, -.4], .8, 1.2, 1e-6
g = grad(q, keys, alpha, c)
fd = []
for d in range(len(q)):
    plus, minus = q.copy(), q.copy()
    plus[d] += epsilon
    minus[d] -= epsilon
    fd.append((energy(plus, keys, alpha, c)-energy(minus, keys, alpha, c))/(2*epsilon))
gradient_max_error = max(abs(a-b) for a, b in zip(g, fd))
assert gradient_max_error < 1e-8
descent = []
for gamma in [1., 1.5]:
    successor = [a-gamma*b for a, b in zip(q, g)]
    actual = energy(successor, keys, alpha, c)-energy(q, keys, alpha, c)
    bound = -gamma*(1-gamma/2)*dot(g, g)
    assert actual <= bound + 1e-12
    descent.append({'gamma_for_identity_check_only': gamma,
                    'actual_energy_change': actual, 'proven_upper_bound': bound})

# Arbitrary query initialization still permits the tied-value result above.
# Independent V example: K=(+-e1), V=(+-e2), c=alpha=1.
independent_q = [.5, .2]
field = [independent_q[0], independent_q[1]-math.tanh(independent_q[0])]
jacobian = [[1., 0.], [-1/math.cosh(independent_q[0])**2, 1.]]
curl = jacobian[1][0]-jacobian[0][1]
assert abs(curl) > .7

# CCCP need not find a global minimum, even in the valid tied-value case.
sym_keys = [[-2.], [2.]]
stationary_gradient = grad([0.], sym_keys, 1., 1.)[0]
stationary_energy = energy([0.], sym_keys, 1., 1.)
lower_energy = energy([2.], sym_keys, 1., 1.)
assert stationary_gradient == 0. and lower_energy < stationary_energy

logits, tau, values = [-1., .3, 2.], .7, [[1., 0.], [0., 2.], [-1., .5]]
def entropy(t):
    return -sum(p*math.log(p) for p in softmax([t*a for a in logits]))

def retrieved(t):
    p = softmax([t*a for a in logits])
    return [sum(pj*v[d] for pj, v in zip(p, values)) for d in range(2)]

p = softmax([tau*a for a in logits])
mean_logit = dot(p, logits)
entropy_derivative = -tau*sum(pj*(a-mean_logit)**2 for pj, a in zip(p, logits))
entropy_fd = (entropy(tau+epsilon)-entropy(tau-epsilon))/(2*epsilon)
retrieval_derivative = [sum(pj*v[d]*(a-mean_logit)
                            for pj, v, a in zip(p, values, logits)) for d in range(2)]
retrieval_fd = [(a-b)/(2*epsilon)
                for a, b in zip(retrieved(tau+epsilon), retrieved(tau-epsilon))]
assert abs(entropy_derivative-entropy_fd) < 1e-8
assert max(abs(a-b) for a, b in zip(retrieval_derivative, retrieval_fd)) < 1e-8

result = {
    'scope': 'Fixed algebra examples only; not diffusion/model experiments, parameter selection, or proof by testing.',
    'gradient_max_error': gradient_max_error,
    'tied_value_descent_examples': descent,
    'independent_value_field': field,
    'independent_value_jacobian': jacobian,
    'independent_value_curl': curl,
    'nonglobal_stationary_example': {'q': 0., 'gradient': stationary_gradient,
                                   'energy': stationary_energy, 'energy_at_q_2': lower_energy},
    'entropy_derivative': {'exact': entropy_derivative, 'finite_difference': entropy_fd},
    'retrieval_derivative': {'exact': retrieval_derivative, 'finite_difference': retrieval_fd},
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'all_checks_passed': True,
    'model_calls': 0, 'gpu_calls': 0,
    'cost_scope': 'Python process after initial time import through checks/hash; excludes process startup, JSON write and print.',
    'wall_seconds': time.perf_counter()-started_wall,
    'process_cpu_seconds': time.process_time()-started_cpu,
}
Path(__file__).with_name('algebra_results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
