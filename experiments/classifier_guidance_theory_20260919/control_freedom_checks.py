"""Small CPU algebra checks; not GAN training or image-model benchmarks."""
import argparse
import json
from pathlib import Path

import numpy as np


def js(p, q):
    m = (p + q) / 2
    def kl(x):
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / m[mask])))
    return (kl(p) + kl(q)) / 2


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260920)

    # With unrestricted W, a fixed nonzero coefficient can represent any field.
    strong = rng.normal(size=(16, 5))
    desired_field = rng.normal(size=(16, 5))
    alpha = 0.8
    inverse_weak = strong + (strong - desired_field) / alpha
    inverse_error = float(np.max(np.abs(strong + alpha * (strong - inverse_weak) - desired_field)))
    assert inverse_error < 1e-12

    # Fixed reference r=(1,0): two times x two states. These are local fields,
    # not full trajectory optima, and the reference head is NOT trainable here.
    desired = np.array([[0.4, 0.3], [1.2, -0.3], [1.0, 0.3], [1.8, -0.3]])
    fixed = np.tile([desired[:, 0].mean(), 0.0], (4, 1))
    time = np.array([[0.8, 0.0], [0.8, 0.0], [1.4, 0.0], [1.4, 0.0]])
    state = np.column_stack([desired[:, 0], np.zeros(4)])
    errors = {name: float(np.mean(np.sum((value - desired) ** 2, axis=1)))
              for name, value in [('fixed_scalar', fixed), ('time_scalar', time),
                                  ('state_scalar', state), ('full_vector', desired)]}
    assert np.allclose(list(errors.values()), [0.34, 0.25, 0.09, 0.0])

    # A same-size direct residual head is not generally a superset of weak-head
    # extrapolation: S=(1,0), H={(0,theta)}, a=1 gives two parallel affine sets.
    def affine_error(target, base):
        direction = np.array([0., 1.])
        coefficient = np.dot(target-base, direction) / np.dot(direction, direction)
        return float(np.sum((base+coefficient*direction-target)**2))
    family_errors = {}
    for name, target in [('target_(2,1)', np.array([2., 1.])),
                         ('target_(1,1)', np.array([1., 1.]))]:
        family_errors[name] = dict(
            old_extrapolation_squared_error=affine_error(target, np.array([2., 0.])),
            direct_residual_squared_error=affine_error(target, np.array([1., 0.])))
    assert list(family_errors['target_(2,1)'].values()) == [0., 1.]
    assert list(family_errors['target_(1,1)'].values()) == [1., 0.]

    # Hamiltonian H(u)=p.u + lambda/2 ||u||^2. Scalar control has an exact
    # missing-direction cost at a fixed state, reference direction, and p.
    p, r, lam = np.array([1.0, 2.0]), np.array([1.0, 0.0]), 2.0
    full_u = -p / lam
    best_a = -np.dot(p, r) / (lam * np.dot(r, r))
    scalar_u = best_a * r
    H = lambda u: float(np.dot(p, u) + lam * np.dot(u, u) / 2)
    perpendicular_p = p - r * np.dot(p, r) / np.dot(r, r)
    missing_cost = np.dot(perpendicular_p, perpendicular_p) / (2 * lam)
    assert np.isclose(H(scalar_u) - H(full_u), missing_cost)

    # Random coefficients can conceal bad individual components when the
    # discriminator sees only the pooled output distribution.
    data, q0, q1 = np.array([.5, .5]), np.array([1., 0.]), np.array([0., 1.])
    mixture = (q0 + q1) / 2
    assert js(data, mixture) == 0 and js(data, q0) > 0

    # f=x1, a=x2 -> a*grad f=(x2,0): not a conservative score field.
    # A state gate also has a nonzero input-Jacobian term: v=x+x^2.
    x, h = .7, 1e-5
    v = lambda x: x + x*x
    finite_difference = (v(x+h) - v(x-h)) / (2*h)
    assert abs(finite_difference - (1+2*x)) < 1e-9

    result = dict(
        scope='CPU algebra and finite arrays; no learned image-model results.',
        unrestricted_fixed_scale=dict(a=alpha, field_max_abs_error=inverse_error),
        fixed_reference_local_field_mse=errors,
        finite_head_families=family_errors,
        hamiltonian=dict(best_signed_scalar=float(best_a), scalar_value=H(scalar_u),
                         full_value=H(full_u), missing_direction_cost=float(missing_cost)),
        random_coefficient=dict(pooled_js=js(data, mixture),
                                mean_component_js=(js(data, q0)+js(data, q1))/2),
        state_gate=dict(example='a=x2, f=x1, a*grad(f)=(x2,0)', curl=-1.,
                        true_input_derivative=float(finite_difference),
                        derivative_if_gate_is_detached=1.),
    )
    (output / 'control_freedom_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
