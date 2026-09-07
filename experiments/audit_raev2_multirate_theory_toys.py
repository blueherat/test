#!/usr/bin/env python3
"""CPU analytic counterexamples; no model imports, data loading, or GPU work."""
import argparse
import json
import math
from pathlib import Path


def scalar_split(n, correction, reuse_endpoint=False):
    """V=-y, b=-.3y; modified cheap flow is integrated exactly."""
    h, y, a, beta = 1.0 / n, 1.0, -1.0, -.3
    residual = (a - beta) * y
    for _ in range(n):
        if not reuse_endpoint:
            residual = (a - beta) * y
        predicted = math.exp(beta * h) * y + math.expm1(beta * h) / beta * residual
        residual_end = (a - beta) * predicted
        y = predicted + .5 * h * (residual_end - residual) if correction else predicted
        residual = residual_end
    return abs(y - math.exp(a))


def off_policy_counterexample(lam=20., microsteps=32):
    """V=2t, b=2t+L(y-t^2), R=-L(y-t^2), teacher y=t^2."""
    h, y = 1.0 / microsteps, 0.
    for i in range(microsteps):
        t = i * h
        k1 = 2 * t + lam * (y - t * t)
        predictor = y + h * k1
        k2 = 2 * (t + h) + lam * (predictor - (t + h) ** 2)
        y += .5 * h * (k1 + k2)
    corrected = y - .5 * lam * (y - 1.)
    return {
        "lambda": lam,
        "macro_H": 1.,
        "microsteps": microsteps,
        "teacher_residual_and_material_derivative": 0.,
        "exact_full_endpoint": 1.,
        "full_Heun_endpoint_error_exact_arithmetic": 0.,
        "cheap_Heun_endpoint": y,
        "endpoint_residual_corrected_endpoint": corrected,
        "corrected_abs_error": abs(corrected - 1.),
        "first_microstep_error_formula": -.5 * lam * h ** 3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ns = [20, 40, 80, 160, 320]
    convergence = {}
    for name, correction, reuse in [
        ("held_residual", False, False),
        ("corrected_fresh_anchor", True, False),
        ("corrected_reused_predicted_anchor", True, True),
    ]:
        errors = [scalar_split(n, correction, reuse) for n in ns]
        convergence[name] = {
            "macro_steps": ns,
            "absolute_endpoint_errors": errors,
            "observed_orders": [math.log(errors[i] / errors[i + 1], 2) for i in range(len(ns) - 1)],
        }
    # Over a whole oscillation period, V=M+eps*sin(wt), b_phi=phi*(1+sin(wt)).
    mean, eps, omega = 100., 1., 20.
    phi_value, phi_derivative = (2 * mean + eps) / 3., eps
    fit = {}
    for name, phi in [("value_fit", phi_value), ("material_derivative_fit", phi_derivative)]:
        fit[name] = {
            "phi": phi,
            "period_mean_squared_value_error": (mean - phi) ** 2 + .5 * (eps - phi) ** 2,
            "period_mean_squared_material_derivative_error": .5 * omega ** 2 * (eps - phi) ** 2,
            "constant_residual_part": mean - phi,
        }
    counter = off_policy_counterexample()
    assert .95 < convergence["held_residual"]["observed_orders"][-1] < 1.05
    assert 1.9 < convergence["corrected_fresh_anchor"]["observed_orders"][-1] < 2.1
    assert 1.9 < convergence["corrected_reused_predicted_anchor"]["observed_orders"][-1] < 2.1
    assert fit["material_derivative_fit"]["period_mean_squared_material_derivative_error"] == 0.
    assert fit["material_derivative_fit"]["period_mean_squared_value_error"] > fit["value_fit"]["period_mean_squared_value_error"]
    assert counter["corrected_abs_error"] > 1.
    result = {
        "protocol": "analytic_multirate_toys_cpu_v1",
        "scope": "Analytic mechanism and counterexamples only; no learned-model or FID evidence.",
        "convergence": convergence,
        "same_one_parameter_fit": {"M": mean, "epsilon": eps, "omega": omega, **fit},
        "off_policy_coupling_counterexample": counter,
        "checks_passed": 6,
        "limitations": [
            "The scalar convergence experiment is not a general stability proof.",
            "The corrected reuse formula is not exact FSAL: the reused residual is at the predicted state.",
            "Zero teacher-path derivative loss leaves transverse Jacobians uncontrolled.",
            "Neither error order nor derivative fitting implies improved generated-distribution FID.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
