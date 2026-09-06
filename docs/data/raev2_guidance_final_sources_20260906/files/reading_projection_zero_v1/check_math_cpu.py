"""CPU-only algebra checks from the APG / CFG-Zero* reading.

These are mathematical counterexamples and coordinate identities, not RAE experiments.
"""
import time
start, cpu_start = time.perf_counter(), time.process_time()
from pathlib import Path
import hashlib
import json
import os
import numpy as np

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
out = Path(__file__).resolve().parent

# CFG-Zero* Eq.9 printed bound, even after repairing omega'=omega-1.
c, u, target, alpha, scale = np.array([1.]), np.array([1.]), np.array([0.]), 1., .1
guided = c + alpha * (c - scale * u)
lhs = float(np.sum((guided - target) ** 2))
printed_rhs = float(np.sum(c ** 2) + np.sum(target ** 2) + alpha * np.sum((c - scale * u) ** 2))
assert lhs > printed_rhs

# A valid upper-bound minimizer can still increase the true field error.
c2, u2, target2 = np.array([2., 1.]), np.array([1., 0.]), np.array([3., 2.])
scale2 = float(c2 @ u2 / (u2 @ u2))
ordinary, projected = c2 + (c2 - u2), c2 + (c2 - scale2 * u2)
ordinary_error = float(np.sum((ordinary - target2) ** 2))
projected_error = float(np.sum((projected - target2) ** 2))
assert ordinary_error == 0 and projected_error == 1

# Exact algebra under an affine clean-to-dataward-velocity conversion.
rng = np.random.default_rng(202609106)
z, full = rng.normal(size=(2, 37))
base = .7 * full + .3 * rng.normal(size=37)
t, alpha = .3, .78
conditional_velocity, weak_velocity = (full - z) / t, (base - z) / t
scale3 = float(conditional_velocity @ weak_velocity / (weak_velocity @ weak_velocity))
velocity_star = conditional_velocity + alpha * (conditional_velocity - scale3 * weak_velocity)
clean_from_velocity = z + t * velocity_star
official_clean = full + alpha * (full - base)
clean_identity = official_clean + alpha * (1 - scale3) * (base - z)
np.testing.assert_allclose(clean_from_velocity, clean_identity, atol=2e-15, rtol=2e-15)
projection = (conditional_velocity @ weak_velocity) * weak_velocity / (weak_velocity @ weak_velocity)
velocity_decomposition = projection + (1 + alpha) * (conditional_velocity - projection)
np.testing.assert_allclose(velocity_star, velocity_decomposition, atol=2e-15, rtol=2e-15)
naive_clean_scale = float(full @ base / (base @ base))
naive_clean = full + alpha * (full - naive_clean_scale * base)
assert np.linalg.norm(naive_clean - clean_from_velocity) > .01

# APG's projection removes the first-order radial increment, not finite norm growth.
f, gap = np.array([1., 0.]), np.array([0., 1.])
apg = f + (gap - (gap @ f) * f / (f @ f))
assert float(apg @ apg) == 2 and float(f @ f) == 1

result = {
    'scope': 'Independent finite-dimensional algebra only, no saved RAE banks, GPU, models or FID.',
    'cfgzero_printed_bound_counterexample': {
        'c': c.tolist(), 'u': u.tolist(), 'target': target.tolist(), 'alpha': 1.,
        'scale': scale, 'lhs': lhs, 'printed_rhs': printed_rhs},
    'upper_bound_argmin_not_true_error_argmin': {
        'c': c2.tolist(), 'u': u2.tolist(), 'target': target2.tolist(), 'alpha': 1.,
        'scale_star': scale2, 'ordinary_true_error_squared': ordinary_error,
        'optimized_scale_true_error_squared': projected_error,
        'ordinary_surrogate': float(np.sum((c2 - u2) ** 2)),
        'optimized_surrogate': float(np.sum((c2 - scale2 * u2) ** 2))},
    'raev2_affine_velocity_conversion': {
        'time': t, 'alpha': alpha, 'scale_star': scale3,
        'max_clean_identity_error': float(np.max(np.abs(clean_from_velocity - clean_identity))),
        'max_velocity_projection_identity_error': float(np.max(np.abs(velocity_star - velocity_decomposition))),
        'naive_clean_projection_difference_norm': float(np.linalg.norm(naive_clean - clean_from_velocity))},
    'apg_not_norm_preserving': {'before_norm_squared': float(f @ f), 'after_norm_squared': float(apg @ apg)},
    'passed': True, 'gpu_calls': 0, 'model_calls': 0, 'fid_calls': 0,
    'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'wall_seconds': time.perf_counter() - start, 'cpu_seconds': time.process_time() - cpu_start,
}
with (out / 'math_checks.json').open('x') as handle:
    json.dump(result, handle, indent=2, allow_nan=False)
    handle.write('\n')
print(json.dumps(result, indent=2))
