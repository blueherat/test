"""CPU audit of local Heun density responses and doubly robust feedback.

Uses analytic Gaussian laws and Gauss-Hermite integration, not image training.
All counterfactual interventions change one interval only; the upstream law
and downstream maps are held fixed.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqvae_self_guidance_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import multivariate_normal

OUT = Path(__file__).resolve().parents[2] / "docs/research/self_guidance_ram_20260923"
IDENTITY = np.eye(2)
BASE = np.array([[.20, .35], [-.15, -.10]])
TIME = np.array([[.08, -.11], [.07, .05]])
CONTROL = np.array([[.12, .23], [.31, -.16]])
Q = np.array([[1.3, .2], [.2, .8]])
TARGET = np.array([.2, -.3])
SCALES = np.array([.2, -.3, .4, .1])
STEP = 1 / len(SCALES)


def heun(index, scale):
    a0 = BASE + index * STEP * TIME + scale * CONTROL
    a1 = BASE + (index + 1) * STEP * TIME + scale * CONTROL
    matrix = IDENTITY + STEP / 2 * (a0 + a1) + STEP**2 / 2 * a1 @ a0
    derivative = STEP * CONTROL + STEP**2 / 2 * (CONTROL @ a0 + a1 @ CONTROL)
    return matrix, derivative


def gaussian_rule(covariance, order=36):
    nodes, weights = hermgauss(order)
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    points = np.sqrt(2) * np.stack([first.ravel(), second.ravel()], axis=1)
    points = points @ np.linalg.cholesky(covariance).T
    weights = np.outer(weights, weights).ravel() / np.pi
    return points, weights


def quadratic(points, matrix):
    return np.einsum("bi,ij,bj->b", points, matrix, points)


def basis(points):
    x, y = points.T
    return np.stack([np.ones_like(x), x, y, x*x, x*y, y*y], axis=1)


def basis_direction(points, direction):
    x, y = points.T
    dx, dy = direction.T
    return np.stack([np.zeros_like(x), dx, dy, 2*x*dx, y*dx+x*dy, 2*y*dy], axis=1)


def main():
    matrices = [heun(i, a)[0] for i, a in enumerate(SCALES)]
    covariances = [IDENTITY]
    for matrix in matrices:
        covariances.append(matrix @ covariances[-1] @ matrix.T)
    interval = 1
    matrix, derivative = heun(interval, SCALES[interval])
    upstream = covariances[interval]
    current = covariances[interval + 1]
    suffix = IDENTITY.copy()
    for subsequent in matrices[interval + 1:]:
        suffix = subsequent @ suffix
    value_matrix = suffix.T @ Q @ suffix
    value_linear = suffix.T @ Q @ TARGET
    constant = -.5 * TARGET @ Q @ TARGET

    def value(points):
        return -.5 * quadratic(points, value_matrix) + points @ value_linear + constant

    def value_expectation(covariance):
        return -.5 * np.trace(value_matrix @ covariance) + constant

    covariance_derivative = derivative @ upstream @ matrix.T + matrix @ upstream @ derivative.T
    exact_gradient = -.5 * np.trace(value_matrix @ covariance_derivative)
    points, weights = gaussian_rule(current)
    upstream_points, upstream_weights = gaussian_rule(upstream)
    vector_matrix = derivative @ np.linalg.inv(matrix)
    eta = -np.trace(vector_matrix) + quadratic(points, np.linalg.inv(current) @ vector_matrix)
    score_gradient = float(weights @ (value(points) * eta))
    assert abs(score_gradient - exact_gradient) < 1e-12
    features = basis(points)
    gram = features.T @ (weights[:, None] * features)
    moment = weights @ basis_direction(points, points @ vector_matrix.T)
    coefficients = np.linalg.solve(gram, moment)
    riesz_error = float(np.sqrt(weights @ (features @ coefficients - eta)**2))
    assert riesz_error < 1e-12

    epsilons = [.4, .2, .1, .05, .025]
    finite_results = []
    for epsilon in epsilons:
        plus = heun(interval, SCALES[interval] + epsilon)[0]
        minus = heun(interval, SCALES[interval] - epsilon)[0]
        plus_covariance = plus @ upstream @ plus.T
        minus_covariance = minus @ upstream @ minus.T
        contrast = (value_expectation(plus_covariance) - value_expectation(minus_covariance)) / (2*epsilon)
        eta_finite = (
            multivariate_normal.pdf(points, cov=plus_covariance)
            - multivariate_normal.pdf(points, cov=minus_covariance)
        ) / (2*epsilon*multivariate_normal.pdf(points, cov=current))
        density_contrast = float(weights @ (value(points)*eta_finite))
        assert abs(density_contrast - contrast) < 1e-11
        finite_results.append(dict(epsilon=epsilon, contrast=float(contrast),
                                   derivative_error=float(abs(contrast-exact_gradient))))
    errors = np.array([entry["derivative_error"] for entry in finite_results])
    assert np.max(np.abs(errors[:-1] / errors[1:] - 4)) < 1e-6

    epsilon = .1
    plus = heun(interval, SCALES[interval]+epsilon)[0]
    minus = heun(interval, SCALES[interval]-epsilon)[0]
    plus_covariance = plus @ upstream @ plus.T
    minus_covariance = minus @ upstream @ minus.T
    eta_finite = (
        multivariate_normal.pdf(points, cov=plus_covariance)
        - multivariate_normal.pdf(points, cov=minus_covariance)
    ) / (2*epsilon*multivariate_normal.pdf(points, cov=current))
    exact_contrast = (value_expectation(plus_covariance)-value_expectation(minus_covariance))/(2*epsilon)

    def error_function(p):
        return p[:, 0]**2 + .4*p[:, 0]*p[:, 1] - .3*p[:, 1]

    eta_error_function = .2 + .7*points[:, 0]**2 - .4*points[:, 1]**2
    cases = []
    for value_error, response_error in [(0., .4), (.4, 0.), (.4, .4), (.2, .2), (.1, .1)]:
        h = lambda p: value(p) + value_error*error_function(p)
        plugin = float(upstream_weights @ (h(upstream_points @ plus.T)-h(upstream_points @ minus.T))/(2*epsilon))
        response_hat = eta_finite + response_error*eta_error_function
        residual = float(weights @ ((value(points)-h(points))*response_hat))
        estimate = plugin + residual
        predicted_bias = float(weights @ (
            -value_error*error_function(points)*response_error*eta_error_function))
        assert abs(estimate-exact_contrast-predicted_bias) < 1e-11
        cases.append(dict(value_error=value_error, response_error=response_error,
                          estimate=estimate, actual_bias=estimate-exact_contrast,
                          product_bias=predicted_bias))

    rotation = np.array([[0., -1.], [1., 0.]])
    gauss_points, gauss_weights = gaussian_rule(IDENTITY)
    true_rotation_response = quadratic(gauss_points, rotation)
    wrong_precision = np.diag([1., 2.])
    wrong_rotation_response = quadratic(gauss_points, wrong_precision @ rotation)
    wrong_score_gradient = float(gauss_weights @ (
        gauss_points[:, 0]*gauss_points[:, 1]*wrong_rotation_response))
    assert np.max(abs(true_rotation_response)) < 1e-13
    assert abs(wrong_score_gradient-1.) < 1e-12

    result = dict(
        kind="CPU Gaussian/Heun population identities; not image training",
        interval=interval, exact_discrete_gradient=float(exact_gradient),
        exact_density_response_gradient=score_gradient,
        polynomial_Riesz_fit_L2_error=riesz_error,
        Riesz_coefficients=coefficients.tolist(),
        local_finite_difference=finite_results,
        doubly_robust_finite_contrast=dict(
            epsilon=epsilon, true_contrast=float(exact_contrast), cases=cases),
        incorrect_marginal_score=dict(
            actual_density="N(0,I)", substituted_score="-diag(1,2)x",
            control="rotation Jx", reward="x0*x1",
            correct_gradient=0., wrong_score_gradient=wrong_score_gradient),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"local_riesz_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].loglog(epsilons, errors, "o-", label="Local Heun finite-difference bias")
    axes[0].set(xlabel="Local perturbation epsilon", ylabel="Absolute gradient error",
                title="Finite-step derivative error is quadratic")
    both = cases[2:]
    axes[1].loglog([c["value_error"] for c in both],
                  [abs(c["actual_bias"]) for c in both], "o-", label="Both nuisance errors shrink")
    axes[1].set(xlabel="Error amplitude in both learned functions", ylabel="Absolute DR bias",
                title="Bias is a product of nuisance errors")
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.savefig(OUT/"local_riesz_audit.png", dpi=180)
    fig.savefig(OUT/"local_riesz_audit.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
