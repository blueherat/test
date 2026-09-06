"""CPU-only exact-moment audit of the RAEv2 stochastic refresh.

No Monte Carlo sampling or GPU is needed. A diagonal Gaussian oracle makes
the complete discrete sampler affine; its variance follows an exact scalar
recursion, which separates discretization bias from sampling uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from experiments.raev2_stochastic_guidance import (
        full_score_residual,
        gaussian_bridge_variance,
        refresh_after_euler,
        refresh_coefficients,
    )
except ModuleNotFoundError:
    from raev2_stochastic_guidance import (
        full_score_residual,
        gaussian_bridge_variance,
        refresh_after_euler,
        refresh_coefficients,
    )


def gaussian_variance(time: float, data_variance: float) -> float:
    return (1.0 - time) ** 2 * data_variance + time**2


def discrete_variance(
    steps: int,
    data_variance: float,
    eta: float,
    mode: str,
    data_mean: float = 0.0,
) -> dict[str, float | int | str]:
    variance = 1.0
    mean = 0.0
    max_path_relative_error = 0.0
    max_path_mean_error = 0.0
    maximum_clock = 0.0
    total_clock = 0.0
    for step in range(steps):
        time = 1.0 - step / steps
        next_time = 1.0 - (step + 1) / steps
        step_size = time - next_time
        target = gaussian_variance(time, data_variance)
        target_mean = (1.0 - time) * data_mean
        clean_coefficient = (1.0 - time) * data_variance / target
        clean_intercept = data_mean - clean_coefficient * target_mean
        euler_coefficient = 1.0 + step_size * (clean_coefficient - 1.0) / time
        euler_intercept = step_size * clean_intercept / time
        coef = refresh_coefficients(time, next_time, eta=eta)
        residual_coefficient = (
            1.0 / gaussian_bridge_variance(time) - 1.0 / target
        )
        coefficient = euler_coefficient
        intercept = euler_intercept
        noise_variance = 0.0
        if mode in {"balanced", "score_only"}:
            coefficient = (
                coef.decay * euler_coefficient
                + coef.residual_weight * residual_coefficient
            )
            intercept = (
                coef.decay * euler_intercept
                + coef.residual_weight * target_mean / target
            )
        if mode in {"balanced", "noise_only"}:
            noise_variance = coef.noise_std**2
        variance = coefficient**2 * variance + noise_variance
        mean = coefficient * mean + intercept
        expected = gaussian_variance(next_time, data_variance)
        expected_mean = (1.0 - next_time) * data_mean
        max_path_relative_error = max(
            max_path_relative_error, abs(variance / expected - 1.0)
        )
        max_path_mean_error = max(max_path_mean_error, abs(mean - expected_mean))
        maximum_clock = max(maximum_clock, coef.clock_increment)
        total_clock += coef.clock_increment
    return {
        "steps": steps,
        "data_variance": data_variance,
        "data_mean": data_mean,
        "eta": eta,
        "mode": mode,
        "endpoint_variance": variance,
        "endpoint_relative_error": variance / data_variance - 1.0,
        "endpoint_mean": mean,
        "endpoint_mean_error": mean - data_mean,
        "max_path_relative_error": max_path_relative_error,
        "max_path_mean_error": max_path_mean_error,
        "max_clock_increment": maximum_clock,
        "total_clock_increment": total_clock,
    }


def audit_tensor_implementation() -> dict[str, float | bool]:
    generator = torch.Generator(device="cpu").manual_seed(20260906)
    state = torch.randn((4, 3, 8, 8), generator=generator)
    initial_stream_state = generator.get_state().clone()
    refresh_generator = torch.Generator(device="cpu").manual_seed(1920260906)
    noise = torch.randn(state.shape, generator=refresh_generator)
    independent_stream = torch.equal(initial_stream_state, generator.get_state())
    assert independent_stream, "refresh must not consume the initial noise stream"
    time, next_time = 0.8, 0.79
    variance = gaussian_bridge_variance(time)
    full = (1.0 - time) / variance * state
    euler = state + (time - next_time) * (full - state) / time
    identity = refresh_after_euler(
        state, euler, full, time, next_time, eta=0.0
    )
    identity_same_object = identity is euler
    assert identity_same_object, "eta=0 must preserve the exact Euler object"
    residual = full_score_residual(state, full, time)
    coef = refresh_coefficients(time, next_time, eta=0.15)
    actual = refresh_after_euler(
        state, euler, full, time, next_time, eta=0.15, noise=noise
    )
    expected = coef.decay * euler + coef.noise_std * noise
    max_error = (actual - expected).abs().max().item()
    assert max_error < 1e-6
    # Exact Gaussian transport plus the refresh is an exactly invariant
    # Gaussian oracle step at FINITE time separation in real arithmetic.
    next_variance = gaussian_bridge_variance(next_time)
    exact_refreshed_variance = coef.decay**2 * next_variance + coef.noise_std**2
    finite_invariance_error = abs(exact_refreshed_variance - next_variance)
    assert finite_invariance_error < 1e-12
    # A genuinely noncentral anisotropic 3D oracle exercises the residual,
    # rather than relying on its cancellation for unit isotropic data.
    data_variance = torch.tensor([0.25, 1.0, 4.0]).view(1, 3, 1, 1)
    data_mean = torch.tensor([-1.5, 0.4, 2.0]).view(1, 3, 1, 1)
    target_variance = (1.0 - time)**2 * data_variance + time**2
    target_mean = (1.0 - time) * data_mean
    clean_slope = (1.0 - time) * data_variance / target_variance
    full = data_mean + clean_slope * (state - target_mean)
    euler = state + (time - next_time) * (full - state) / time
    actual = refresh_after_euler(
        state, euler, full, time, next_time, eta=0.15, noise=noise
    )
    oracle_residual = -(state - target_mean) / target_variance + state / variance
    expected = (
        coef.decay * euler + coef.residual_weight * oracle_residual
        + coef.noise_std * noise
    )
    anisotropic_error = (actual - expected).abs().max().item()
    assert anisotropic_error < 1e-6
    return {
        "eta_zero_same_tensor_object": identity_same_object,
        "refresh_rng_independent_of_initial_rng": independent_stream,
        "unit_gaussian_score_residual_max": residual.abs().max().item(),
        "tensor_step_max_error": max_error,
        "finite_gaussian_refresh_variance_error": finite_invariance_error,
        "noncentral_anisotropic_tensor_step_max_error": anisotropic_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--eta", type=float, default=0.15)
    args = parser.parse_args()
    implementation = audit_tensor_implementation()
    results = [
        discrete_variance(steps, variance, args.eta, mode, mean)
        for variance, mean in ((0.25, -1.5), (1.0, 0.4), (4.0, 2.0))
        for mode in ("none", "balanced", "score_only", "noise_only")
        for steps in (100, 400, 1600, 6400)
    ]
    for variance in (0.25, 1.0, 4.0):
        balanced = [
            row for row in results
            if row["data_variance"] == variance and row["mode"] == "balanced"
        ]
        assert abs(balanced[-1]["endpoint_relative_error"]) < 0.001
        assert abs(balanced[-1]["endpoint_mean_error"]) < 0.001
        assert abs(balanced[-1]["endpoint_relative_error"]) < (
            abs(balanced[0]["endpoint_relative_error"]) / 20.0
        ), "balanced sampler must converge toward the oracle variance"
        for mode in ("score_only", "noise_only"):
            finest = next(
                row for row in results
                if row["data_variance"] == variance
                and row["mode"] == mode and row["steps"] == 6400
            )
            assert abs(finest["endpoint_relative_error"]) > 0.05, (
                "ablation must fail to preserve the oracle marginal"
            )
    report = {
        "status": "passed",
        "protocol": "exact affine moment recursion, no GPU or Monte Carlo",
        "joint_gaussian_target": {
            "mean": [-1.5, 0.4, 2.0],
            "diagonal_covariance": [0.25, 1.0, 4.0],
        },
        "implementation_checks": implementation,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "implementation_checks": implementation,
        "unit_gaussian_finest": [
            row for row in results
            if row["data_variance"] == 1.0 and row["steps"] == 6400
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
