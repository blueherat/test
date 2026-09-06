"""Fixed CPU algebra witnesses; no diffusion model, data, tuning or GPU."""
import time
wall = time.perf_counter()
cpu = time.process_time()
import json
import math
from pathlib import Path
import numpy as np


def softmax(a):
    e = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def hessian(a):
    p = softmax(a)
    return np.outer(p, p) - np.diag(p)


a = np.array([4., -4.])
B = np.array([[.75, .25], [.25, .75]])
H = hessian(a)
H_composite = B.T @ hessian(B @ a) @ B
direction = np.array([1., -1.]) / math.sqrt(2)
Q = np.array([[1., 0.], [0., 2.], [2., -1.]])
K = np.array([[1., 0.], [-1., 0.], [0., 1.]])
J = np.ones((3, 3)) / 3
attention = softmax((J @ Q) @ K.T)
original_attention = softmax(Q @ K.T)
geometric = np.exp(np.log(original_attention).mean(0))
geometric /= geometric.sum()
candidate = np.array([.2, .3, .5])
reverse_objective = lambda r: float((r * (np.log(r)-np.log(original_attention))).sum(-1).mean())
kl_to_barycenter = float((candidate * (np.log(candidate)-np.log(attention[0]))).sum())
row_constant = np.tile(np.array([.5, -.2]), (3, 1))
reflection_blur = np.array([[.5, .5, 0.], [.25, .5, .25], [0., .5, .5]])
x = np.array([1., 0., 0.])
out = {
    'scope': 'Fixed mathematical witnesses, not an RAE experiment or paper result.',
    'softmax_energy': {
        'a': a.tolist(), 'B': B.tolist(),
        'original_lse': float(np.logaddexp.reduce(a)),
        'blurred_lse': float(np.logaddexp.reduce(B @ a)),
        'H_times_ones_max_abs': float(np.abs(H @ np.ones(2)).max()),
        'original_mean_zero_direction_curvature_abs': abs(float(direction @ H @ direction)),
        'composite_mean_zero_direction_curvature_abs': abs(float(direction @ H_composite @ direction)),
        'meaning': 'Averaging can increase curvature in a nonconstant direction despite lowering log-sum-exp. H has a constant null direction; use the exact composite Hessian B.T H(Ba) B.'
    },
    'uniform_query': {
        'logit_blur_equivalence_max_abs': float(np.abs(J @ (Q @ K.T) - (J @ Q) @ K.T).max()),
        'attention': attention.tolist(),
        'all_attention_rows_equal_max_abs': float(np.abs(attention - attention[:1]).max()),
        'attention_is_uniform_over_keys': bool(np.allclose(attention, 1/3)),
        'query_mean_preservation_max_abs': float(np.abs((J @ Q).mean(0)-Q.mean(0)).max()),
        'reverse_kl_geometric_barycenter_max_abs': float(np.abs(geometric-attention[0]).max()),
        'reverse_kl_optimality_identity_residual': reverse_objective(candidate)-reverse_objective(attention[0])-kl_to_barycenter,
        'arithmetic_mean_attention': original_attention.mean(0).tolist(),
        'different_from_forward_kl_barycenter': not bool(np.allclose(attention[0], original_attention.mean(0))),
        'query_projection_pythagorean_residual': float(np.square(Q-row_constant).sum()-np.square(Q-J@Q).sum()-np.square(J@Q-row_constant).sum()),
    },
    'finite_reflection_padding': {
        'input': x.tolist(), 'output': (reflection_blur @ x).tolist(),
        'input_mean': float(x.mean()), 'output_mean': float((reflection_blur @ x).mean()),
        'meaning': 'Normalized symmetric local kernel plus reflection padding need not preserve the global spatial mean; the exact full-query average does.'
    },
    'cost': {'wall_seconds': time.perf_counter()-wall, 'cpu_seconds': time.process_time()-cpu,
             'scope': 'Includes imports and computation; excludes final JSON serialization, process exit, downloads and reading.'}
}
assert out['softmax_energy']['blurred_lse'] < out['softmax_energy']['original_lse']
assert out['softmax_energy']['composite_mean_zero_direction_curvature_abs'] > out['softmax_energy']['original_mean_zero_direction_curvature_abs']
assert out['uniform_query']['logit_blur_equivalence_max_abs'] < 1e-14
assert out['uniform_query']['all_attention_rows_equal_max_abs'] < 1e-14
assert not out['uniform_query']['attention_is_uniform_over_keys']
assert out['uniform_query']['reverse_kl_geometric_barycenter_max_abs'] < 1e-14
assert abs(out['uniform_query']['reverse_kl_optimality_identity_residual']) < 1e-14
assert abs(out['uniform_query']['query_projection_pythagorean_residual']) < 1e-14
(Path(__file__).parent/'algebra_results.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
