"""CPU checks of scale identifiability and calibration; no model training.

The Gaussian examples use exact population KL between equal-covariance
Gaussians, not an estimated GAN loss or an image-quality experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260919)
    initial = rng.normal(size=(128, 2))

    def strong(x, t):
        return np.tanh(x @ np.array([[0.3, -0.2], [0.1, 0.4]])) + t * 0.1

    def weak(x, t):
        return np.sin(x @ np.array([[0.2, 0.1], [-0.3, 0.2]])) - t * 0.2

    a, b = 0.8, 1.3

    def field(x, t, reparameterized):
        s, w = strong(x, t), weak(x, t)
        f = (6 / 7 if t < 0.25 else 1.0) if t < 0.5 else 0.0
        if reparameterized:
            w = s + (a / b) * (w - s)
            return s + b * f * (s - w)
        return s + a * f * (s - w)

    x, y = initial.copy(), initial.copy()
    for t in np.arange(64) / 64:
        x += field(x, t, False) / 64
        y += field(y, t, True) / 64
    gauge_error = float(np.max(np.abs(x - y)))
    assert gauge_error < 1e-12

    # S=(1,0), W_theta=(0,theta), target N((2,1), I).
    # At training scale a>0: theta*=-1/a; G_b=(1+b,b/a).
    def capacity_map(a):
        return a * (a + 1) / (a * a + 1)

    def capacity_loss(theta, b):
        return 0.5 * ((b - 1) ** 2 + (-b * theta - 1) ** 2)

    theta = -1 / a
    best_b = capacity_map(a)
    head_gradient = a * (a * theta + 1)
    scale_gradient = (a - 1) + theta * (a * theta + 1)
    assert abs(head_gradient) < 1e-12
    assert capacity_loss(theta, best_b) < capacity_loss(theta, a)
    iterations = [a]
    for _ in range(12):
        iterations.append(capacity_map(iterations[-1]))
    eps = 1e-5
    fixed_point_slope = (capacity_map(1 + eps) - capacity_map(1 - eps)) / (2 * eps)
    # At (theta,a)=(-1,1), Hessian [[1,-1],[-1,2]] gives slope 1/2.
    assert abs(fixed_point_slope - 0.5) < 1e-9

    # S=0, W_theta=-theta, target N(1,1), weak-head penalty lambda*theta^2/2.
    # theta*(a)=a/(a^2+lambda); best sampling b=1/theta*=a+lambda/a.
    lam = 0.16
    regularized_theta = a / (a * a + lam)
    regularized_b = 1 / regularized_theta
    grid = np.linspace(0.25, 2.0, 176)
    assert np.all(grid + lam / grid > grid)
    rows = [dict(train_a=float(v), identity=float(v),
                 capacity_best_b=float(capacity_map(v)),
                 regularized_best_b=float(v + lam / v)) for v in grid]
    with (output / 'calibration_maps.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = dict(
        scope='Analytic population Gaussian examples and synthetic vector fields; CPU only.',
        gauge=dict(a=a, b=b, euler_64_endpoint_max_abs_difference=gauge_error,
                   limitation='Transformed W contains S and need not fit the existing shallow head.'),
        restricted_capacity=dict(train_a=a, trained_theta=theta, head_gradient=head_gradient,
                                 scale_gradient=scale_gradient, best_b=best_b,
                                 population_kl_at_train_a=capacity_loss(theta, a),
                                 population_kl_at_best_b=capacity_loss(theta, best_b),
                                 fixed_point=1.0, slope=fixed_point_slope,
                                 calibration_iterations=iterations),
        regularized=dict(train_a=a, penalty_lambda=lam, trained_theta=regularized_theta,
                         best_b=regularized_b, positive_finite_fixed_point=False,
                         explanation='F(a)=a+lambda/a > a for all a>0.'),
        feature_blindness=dict(real_mean=[0, 0], fake_mean=[0, 5], covariance='identity',
                               feature='first coordinate', feature_kl=0.0, image_kl=12.5),
    )
    (output / 'checks.json').write_text(json.dumps(result, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ['capacity_best_b', 'regularized_best_b'],
                               ['Restricted head: stable fixed point', 'Head penalty: no finite fixed point']):
        ax.plot(grid, [row[key] for row in rows], label='Best sampling scale F(a)', linewidth=2)
        ax.plot(grid, grid, '--', color='gray', label='F(a) = a')
        ax.set(xlabel='Training scale a', ylabel='Best sampling scale b', title=title)
        ax.scatter([a], [capacity_map(a) if key == 'capacity_best_b' else regularized_b], color='black')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('Exact Gaussian toy examples — not image-model results', fontsize=11)
    fig.savefig(output / 'calibration_maps.png', dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args().output)
