"""CPU audit of probability-current transport in normalized implicit tilt matching.

This is a mathematical diagnostic, not a training implementation.  The old
two-dimensional flow has standard Gaussian endpoints and a nonzero rotational
component.  A normalized exponential tilt broadens the second terminal axis.
The inferred ITM field transports the old solenoidal probability current instead
of forcing canonicalization.  No GPU or model checkpoint is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ndtri
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/self_guidance_ram_20260923/tilt_flux_audit.json"
OMEGA = 0.8
TARGET_VARIANCE = np.array([1.0, 2.0])


def path(t: float):
    old_var = (1 - t) ** 2 + t**2
    new_var = (1 - t) ** 2 + t**2 * TARGET_VARIANCE
    old_coef = (2 * t - 1) / old_var
    new_coef = (t * TARGET_VARIANCE - (1 - t)) / new_var
    return old_var, new_var, old_coef, new_coef


def components(x: np.ndarray, t: float):
    old_var, new_var, old_coef, new_coef = path(t)
    log_h = np.log(old_var) - 0.5 * np.log(new_var).sum()
    log_h = log_h + 0.5 * (x**2 * (1 / old_var - 1 / new_var)).sum(axis=-1)
    h = np.exp(log_h)
    rotation = OMEGA * np.stack((-x[..., 1], x[..., 0]), axis=-1)
    return old_coef * x, new_coef * x, rotation, h


def velocity(x: np.ndarray, t: float, kind: str):
    old_canonical, new_canonical, rotation, h = components(x, t)
    if kind == "correct":
        return new_canonical + rotation / h[..., None]
    if kind == "naive":
        return new_canonical + rotation
    if kind == "old":
        return old_canonical + rotation
    raise ValueError(kind)


def rk4(x: np.ndarray, steps: int, kind: str):
    x = x.copy()
    dt = 1 / steps
    for step in range(steps):
        t = step * dt
        k1 = velocity(x, t, kind)
        k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt, kind)
        k3 = velocity(x + 0.5 * dt * k2, t + 0.5 * dt, kind)
        k4 = velocity(x + dt * k3, t + dt, kind)
        x += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def main():
    rng = np.random.default_rng(20260923)
    points = rng.normal(size=(256, 2))
    t = 0.61
    old_var, new_var, _, _ = path(t)
    old_canonical, new_canonical, rotation, h = components(points, t)
    correct = new_canonical + rotation / h[:, None]
    old = old_canonical + rotation

    # Conditional population normal equation for normalized ITM.
    expected_weighted_u = h[:, None] * new_canonical
    implicit_residual = correct - old - (
        expected_weighted_u - old_canonical - (h - 1)[:, None] * correct
    )

    # Div(q^+ * d^+) / q^+, checked by finite differences.
    eps = 1e-5
    divergence = np.zeros(points.shape[0])
    for axis in range(2):
        offset = np.zeros_like(points)
        offset[:, axis] = eps
        plus = components(points + offset, t)
        minus = components(points - offset, t)
        d_plus = plus[2] / plus[3][:, None]
        d_minus = minus[2] / minus[3][:, None]
        divergence += (d_plus[:, axis] - d_minus[:, axis]) / (2 * eps)
    new_score = -points / new_var
    correct_stein = divergence + (new_score * rotation / h[:, None]).sum(axis=1)
    naive_stein = (new_score * rotation).sum(axis=1)

    old_density = np.exp(-0.5 * (points**2).sum(axis=1) / old_var) / (2 * np.pi * old_var)
    new_density = np.exp(-0.5 * (points**2 / new_var).sum(axis=1)) / (
        2 * np.pi * np.sqrt(new_var.prod())
    )
    flux_error = new_density[:, None] * rotation / h[:, None] - old_density[:, None] * rotation

    # Independent population ODE integration; scrambled Sobol reduces moment noise.
    normal = ndtri(qmc.Sobol(d=2, scramble=True, seed=20260923).random_base2(m=15))
    end_correct = rk4(normal, 256, "correct")
    sample_cov = end_correct.T @ end_correct / len(normal)
    # Linear naive field allows exact covariance integration without sampling noise.
    rotation_matrix = np.array([[0.0, -OMEGA], [OMEGA, 0.0]])

    def covariance_rhs(time, flat):
        cov = flat.reshape(2, 2)
        matrix = np.diag(path(time)[3]) + rotation_matrix
        return (matrix @ cov + cov @ matrix.T).reshape(-1)

    naive_cov = solve_ivp(covariance_rhs, (0.0, 1.0), np.eye(2).reshape(-1), rtol=1e-11, atol=1e-12).y[:, -1].reshape(2, 2)
    # Compare RK4 integration with a doubled step count on a fixed subset.
    subset = normal[:1024]
    integration_error = np.max(np.abs(rk4(subset, 256, "correct") - rk4(subset, 512, "correct")))

    # Constant shift invariance of normalized weights; includes exact zero update.
    rewards = np.array([-1.2, 0.3, 1.1, 2.0])
    normalized = np.exp(rewards - rewards.max())
    normalized /= normalized.mean()
    shifted = np.exp((rewards + 23.0) - (rewards + 23.0).max())
    shifted /= shifted.mean()
    constant_weights = np.exp(np.full(4, 7.0) - 7.0)

    result = {
        "setting": {
            "old_endpoint_covariance": np.eye(2).tolist(),
            "target_endpoint_covariance": np.diag(TARGET_VARIANCE).tolist(),
            "omega": OMEGA,
            "reward": "R(x)=x_2^2/4; normalized exponential weight exp(R)/sqrt(2)",
            "old_field": "canonical old velocity + omega*J*x",
            "correct_field": "canonical tilted velocity + omega*J*x/h_t(x)",
            "h": "tilted_interpolant_density / old_interpolant_density",
        },
        "local_checks": {
            "max_normal_equation_residual": float(np.max(np.abs(implicit_residual))),
            "max_probability_current_error": float(np.max(np.abs(flux_error))),
            "max_corrected_stein_residual_finite_difference": float(np.max(np.abs(correct_stein))),
            "rms_naive_stein_residual": float(np.sqrt(np.mean(naive_stein**2))),
        },
        "endpoint_checks": {
            "sobol_sample_count": len(normal),
            "corrected_endpoint_second_moment": sample_cov.tolist(),
            "corrected_second_moment_max_abs_error": float(np.max(np.abs(sample_cov - np.diag(TARGET_VARIANCE)))),
            "corrected_endpoint_mean": end_correct.mean(axis=0).tolist(),
            "naive_endpoint_covariance_exact_ode": naive_cov.tolist(),
            "naive_covariance_max_abs_error": float(np.max(np.abs(naive_cov - np.diag(TARGET_VARIANCE)))),
            "rk4_256_vs_512_steps_max_endpoint_difference_on_1024_seeds": float(integration_error),
        },
        "normalization_checks": {
            "constant_reward_offset_max_weight_difference": float(np.max(np.abs(normalized - shifted))),
            "constant_reward_max_abs_weight_minus_one": float(np.max(np.abs(constant_weights - 1))),
            "constant_reward_restricted_head_semigradient": 0.0,
        },
        "scope": "CPU Gaussian identity and ODE audit only; no current-model marginal consistency or image-training performance claim.",
    }
    assert result["local_checks"]["max_normal_equation_residual"] < 1e-12
    assert result["local_checks"]["max_probability_current_error"] < 1e-12
    assert result["local_checks"]["max_corrected_stein_residual_finite_difference"] < 1e-7
    assert result["endpoint_checks"]["corrected_second_moment_max_abs_error"] < 5e-3
    assert result["endpoint_checks"]["naive_covariance_max_abs_error"] > 0.1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
