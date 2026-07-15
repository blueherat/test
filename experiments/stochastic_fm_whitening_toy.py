"""Analytic toy for covariance whitening under stochastic FM targets.

The toy is a local quadratic model around the population-optimal velocity
field.  It isolates deterministic curvature from microscopic target noise,
so its mean and covariance dynamics are available in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_RESIDUAL_VARIANCE = np.logspace(-2.0, 2.0, 8)
DEFAULT_PREDICTABILITY = np.array(
    [0.02, 0.05, 0.10, 0.40, 0.80, 0.98, 0.999, 0.99999],
    dtype=np.float64,
)
DEFAULT_DECODER_GAIN = np.array(
    [8.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0],
    dtype=np.float64,
)


@dataclass(frozen=True)
class WhiteningToy:
    """Closed-form local regression system.

    The raw velocity error is ``J @ e``.  Direction ``i`` has total residual
    variance ``R_i`` and irreducible fraction ``1-rho_i``.  The weighted loss
    uses ``W_gamma = diag(R_i ** -gamma)``.
    """

    residual_variance: np.ndarray
    predictability: np.ndarray
    decoder_gain: np.ndarray
    sensitivity: np.ndarray
    initial_error: np.ndarray
    decoder_metric: np.ndarray
    parameter_dim: int
    seed: int

    @property
    def direction_count(self) -> int:
        return int(self.residual_variance.size)

    @property
    def irreducible_variance(self) -> np.ndarray:
        return (1.0 - self.predictability) * self.residual_variance

    @property
    def architecture(self) -> str:
        return "decoupled" if self.parameter_dim == self.direction_count else "shared"


@dataclass(frozen=True)
class AnalyticOperators:
    weight: np.ndarray
    hessian: np.ndarray
    gradient_noise: np.ndarray
    transition: np.ndarray
    learning_rate: float
    condition_number: float
    gamma: float
    batch_size: float


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def make_toy(
    *,
    parameter_dim: Optional[int] = None,
    residual_variance: Sequence[float] = DEFAULT_RESIDUAL_VARIANCE,
    predictability: Sequence[float] = DEFAULT_PREDICTABILITY,
    decoder_gain: Sequence[float] = DEFAULT_DECODER_GAIN,
    seed: int = 0,
) -> WhiteningToy:
    """Construct a deterministic toy with an orthonormal architecture basis.

    ``parameter_dim == direction_count`` gives a realizable decoupled model.
    A smaller dimension projects all output directions through a shared
    bottleneck, mirroring parameter sharing in a finite network.
    """

    residual = _as_vector(residual_variance, "residual_variance")
    rho = _as_vector(predictability, "predictability")
    gain = _as_vector(decoder_gain, "decoder_gain")
    if not (residual.size == rho.size == gain.size):
        raise ValueError("residual_variance, predictability, and decoder_gain must match")
    if np.any(residual <= 0.0):
        raise ValueError("residual_variance must be strictly positive")
    if np.any((rho < 0.0) | (rho > 1.0)):
        raise ValueError("predictability must lie in [0, 1]")
    if np.any(gain <= 0.0):
        raise ValueError("decoder_gain must be strictly positive")

    direction_count = int(residual.size)
    parameter_dim = direction_count if parameter_dim is None else int(parameter_dim)
    if not 1 <= parameter_dim <= direction_count:
        raise ValueError("parameter_dim must be between 1 and direction_count")

    rng = np.random.default_rng(seed)
    architecture_basis, _ = np.linalg.qr(
        rng.standard_normal((direction_count, parameter_dim)), mode="reduced"
    )
    sensitivity = np.sqrt(residual)[:, None] * architecture_basis
    decoder_metric = (
        sensitivity.T @ (np.square(gain)[:, None] * sensitivity) / direction_count
    )

    if parameter_dim == direction_count:
        initial_error = architecture_basis.T @ np.ones(direction_count, dtype=np.float64)
    else:
        initial_error = np.ones(parameter_dim, dtype=np.float64)
    initial_risk = float(initial_error @ decoder_metric @ initial_error)
    if initial_risk <= 0.0:
        raise ValueError("constructed decoder metric is not positive on the initial error")
    initial_error = initial_error / np.sqrt(initial_risk)

    return WhiteningToy(
        residual_variance=residual.copy(),
        predictability=rho.copy(),
        decoder_gain=gain.copy(),
        sensitivity=sensitivity,
        initial_error=initial_error,
        decoder_metric=decoder_metric,
        parameter_dim=parameter_dim,
        seed=int(seed),
    )


def mode_table(toy: WhiteningToy) -> pd.DataFrame:
    """Return the controlled per-direction signal/noise factors."""

    return pd.DataFrame(
        {
            "direction": np.arange(toy.direction_count),
            "residual_variance_R": toy.residual_variance,
            "predictable_fraction_rho": toy.predictability,
            "irreducible_fraction": 1.0 - toy.predictability,
            "irreducible_variance_N": toy.irreducible_variance,
            "decoder_gain": toy.decoder_gain,
        }
    )


def analytic_operators(
    toy: WhiteningToy,
    *,
    gamma: float,
    batch_size: float,
    learning_rate_fraction: float = 0.4,
    damping: float = 0.0,
) -> AnalyticOperators:
    """Build exact curvature and gradient-noise operators.

    The loss is ``||W_gamma**0.5 (v_hat-v)||^2 / (2d)``.  The stochastic
    covariance contains only microscopic target noise; random-Hessian noise
    is deliberately excluded so the mechanism is identifiable.
    """

    gamma = float(gamma)
    damping = float(damping)
    batch_size = float(batch_size)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    if not 0.0 < learning_rate_fraction < 1.0:
        raise ValueError("learning_rate_fraction must lie in (0, 1)")
    if not (np.isinf(batch_size) or batch_size >= 1.0):
        raise ValueError("batch_size must be positive or np.inf")

    direction_count = toy.direction_count
    weight = np.power(toy.residual_variance + damping, -gamma)
    weighted_sensitivity = weight[:, None] * toy.sensitivity
    hessian = toy.sensitivity.T @ weighted_sensitivity / direction_count

    if np.isinf(batch_size):
        gradient_noise = np.zeros_like(hessian)
    else:
        noise_weight = np.square(weight) * toy.irreducible_variance
        gradient_noise = (
            toy.sensitivity.T @ (noise_weight[:, None] * toy.sensitivity)
            / (direction_count**2 * batch_size)
        )

    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= 0.0:
        raise ValueError("the analytic Hessian must be positive definite")
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    learning_rate = float(2.0 * learning_rate_fraction / eigenvalues[-1])
    transition = np.eye(toy.parameter_dim, dtype=np.float64) - learning_rate * hessian
    return AnalyticOperators(
        weight=weight,
        hessian=hessian,
        gradient_noise=gradient_noise,
        transition=transition,
        learning_rate=learning_rate,
        condition_number=condition_number,
        gamma=gamma,
        batch_size=batch_size,
    )


def analytic_trajectory(
    toy: WhiteningToy,
    *,
    gamma: float,
    batch_size: float,
    steps: int = 500,
    learning_rate_fraction: float = 0.4,
    damping: float = 0.0,
) -> pd.DataFrame:
    """Compute exact mean, covariance, bias, and misadjustment over training."""

    if int(steps) < 1:
        raise ValueError("steps must be at least 1")
    operators = analytic_operators(
        toy,
        gamma=gamma,
        batch_size=batch_size,
        learning_rate_fraction=learning_rate_fraction,
        damping=damping,
    )
    mean_error = toy.initial_error.copy()
    covariance = np.zeros((toy.parameter_dim, toy.parameter_dim), dtype=np.float64)
    rows = []
    for step in range(int(steps) + 1):
        bias = float(mean_error @ toy.decoder_metric @ mean_error)
        misadjustment = float(np.trace(toy.decoder_metric @ covariance))
        rows.append(
            {
                "step": step,
                "bias": max(bias, 0.0),
                "misadjustment": max(misadjustment, 0.0),
                "risk": max(bias + misadjustment, 0.0),
                "gamma": float(gamma),
                "batch_size": float(batch_size),
                "condition_number": operators.condition_number,
                "learning_rate": operators.learning_rate,
                "architecture": toy.architecture,
            }
        )
        mean_error = operators.transition @ mean_error
        covariance = (
            operators.transition @ covariance @ operators.transition.T
            + operators.learning_rate**2 * operators.gradient_noise
        )
        covariance = 0.5 * (covariance + covariance.T)
    return pd.DataFrame(rows)


def run_sweep(
    toy: WhiteningToy,
    *,
    gammas: Iterable[float],
    batch_sizes: Iterable[float],
    steps: int = 500,
    learning_rate_fraction: float = 0.4,
    damping: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a gamma/batch sweep and return all curves plus compact metrics."""

    trajectory_frames = []
    summary_rows = []
    for batch_size in batch_sizes:
        for gamma in gammas:
            trajectory = analytic_trajectory(
                toy,
                gamma=float(gamma),
                batch_size=float(batch_size),
                steps=steps,
                learning_rate_fraction=learning_rate_fraction,
                damping=damping,
            )
            trajectory_frames.append(trajectory)
            risk = trajectory["risk"].to_numpy()
            summary_rows.append(
                {
                    "architecture": toy.architecture,
                    "batch_size": float(batch_size),
                    "gamma": float(gamma),
                    "condition_number": float(trajectory["condition_number"].iloc[0]),
                    "learning_rate": float(trajectory["learning_rate"].iloc[0]),
                    "risk_auc": float(np.trapezoid(risk) / steps),
                    "final_risk": float(risk[-1]),
                    "min_risk": float(np.min(risk)),
                    "final_bias": float(trajectory["bias"].iloc[-1]),
                    "final_misadjustment": float(trajectory["misadjustment"].iloc[-1]),
                }
            )
    return pd.concat(trajectory_frames, ignore_index=True), pd.DataFrame(summary_rows)


def optimal_gamma(summary: pd.DataFrame, metric: str = "risk_auc") -> pd.DataFrame:
    """Select the pre-registered best gamma for each architecture and batch."""

    if metric not in summary.columns:
        raise ValueError(f"unknown metric: {metric}")
    index = summary.groupby(["architecture", "batch_size"], dropna=False)[metric].idxmin()
    columns = [
        "architecture",
        "batch_size",
        "gamma",
        metric,
        "final_risk",
        "condition_number",
    ]
    return summary.loc[index, columns].sort_values(["architecture", "batch_size"]).reset_index(drop=True)


def monte_carlo_validation(
    toy: WhiteningToy,
    *,
    gamma: float,
    batch_size: float,
    steps: int = 200,
    runs: int = 1024,
    learning_rate_fraction: float = 0.4,
    damping: float = 0.0,
    seed: int = 1234,
) -> pd.DataFrame:
    """Validate the recursion by sampling microscopic residual targets directly."""

    if np.isinf(batch_size):
        raise ValueError("Monte Carlo validation requires a finite batch_size")
    if not float(batch_size).is_integer():
        raise ValueError("Monte Carlo validation requires an integer batch_size")
    if int(runs) < 2:
        raise ValueError("runs must be at least 2")
    operators = analytic_operators(
        toy,
        gamma=gamma,
        batch_size=batch_size,
        learning_rate_fraction=learning_rate_fraction,
        damping=damping,
    )
    analytic = analytic_trajectory(
        toy,
        gamma=gamma,
        batch_size=batch_size,
        steps=steps,
        learning_rate_fraction=learning_rate_fraction,
        damping=damping,
    )

    rng = np.random.default_rng(seed)
    errors = np.repeat(toy.initial_error[None, :], int(runs), axis=0)
    target_noise_scale = np.sqrt(toy.irreducible_variance)
    weighted_sensitivity = operators.weight[:, None] * toy.sensitivity
    rows = []
    for step in range(int(steps) + 1):
        run_risk = np.einsum("bi,ij,bj->b", errors, toy.decoder_metric, errors)
        analytic_risk = float(analytic.loc[step, "risk"])
        monte_carlo_risk = float(np.mean(run_risk))
        standard_error = float(np.std(run_risk, ddof=1) / np.sqrt(runs))
        rows.append(
            {
                "step": step,
                "analytic_risk": analytic_risk,
                "monte_carlo_risk": monte_carlo_risk,
                "monte_carlo_se": standard_error,
                "absolute_error": abs(monte_carlo_risk - analytic_risk),
            }
        )
        microscopic_noise = (
            rng.standard_normal(
                (int(runs), int(batch_size), toy.direction_count)
            )
            * target_noise_scale
        )
        batch_target_noise = microscopic_noise.mean(axis=1)
        parameter_noise = (
            batch_target_noise @ weighted_sensitivity / toy.direction_count
        )
        errors = (
            errors @ operators.transition.T
            + operators.learning_rate * parameter_noise
        )
    return pd.DataFrame(rows)


def mechanism_checks(
    decoupled_toy: WhiteningToy,
    shared_toy: WhiteningToy,
    *,
    steps: int = 500,
    learning_rate_fraction: float = 0.4,
) -> pd.DataFrame:
    """Run deterministic checks tied to the intended causal mechanism."""

    gammas = np.linspace(0.0, 1.0, 21)
    batches = (4.0, 16.0, 64.0, 256.0, np.inf)
    _, decoupled_summary = run_sweep(
        decoupled_toy,
        gammas=gammas,
        batch_sizes=batches,
        steps=steps,
        learning_rate_fraction=learning_rate_fraction,
    )
    _, shared_summary = run_sweep(
        shared_toy,
        gammas=gammas,
        batch_sizes=batches,
        steps=steps,
        learning_rate_fraction=learning_rate_fraction,
    )
    decoupled_optimum = optimal_gamma(decoupled_summary)
    shared_optimum = optimal_gamma(shared_summary)
    noiseless_toy = make_toy(
        parameter_dim=decoupled_toy.parameter_dim,
        residual_variance=decoupled_toy.residual_variance,
        predictability=np.ones(decoupled_toy.direction_count),
        decoder_gain=decoupled_toy.decoder_gain,
        seed=decoupled_toy.seed,
    )
    _, noiseless_summary = run_sweep(
        noiseless_toy,
        gammas=gammas,
        batch_sizes=(4.0,),
        steps=steps,
        learning_rate_fraction=learning_rate_fraction,
    )
    noiseless_optimum = float(optimal_gamma(noiseless_summary)["gamma"].iloc[0])

    full_condition = analytic_operators(
        decoupled_toy,
        gamma=1.0,
        batch_size=np.inf,
        learning_rate_fraction=learning_rate_fraction,
    ).condition_number
    full_batch = decoupled_summary[np.isinf(decoupled_summary["batch_size"])]
    full_batch_gamma0 = float(full_batch.loc[np.isclose(full_batch["gamma"], 0.0), "risk_auc"].iloc[0])
    full_batch_gamma1 = float(full_batch.loc[np.isclose(full_batch["gamma"], 1.0), "risk_auc"].iloc[0])
    finite_decoupled = decoupled_optimum[np.isfinite(decoupled_optimum["batch_size"])]
    finite_shared = shared_optimum[np.isfinite(shared_optimum["batch_size"])]

    checks = [
        (
            "full whitening makes deterministic curvature isotropic",
            np.isclose(full_condition, 1.0, atol=1e-10),
            full_condition,
            1.0,
        ),
        (
            "full whitening wins in full-batch finite-budget risk",
            full_batch_gamma1 < full_batch_gamma0,
            full_batch_gamma1,
            full_batch_gamma0,
        ),
        (
            "finite-batch optimum is fractional in the decoupled model",
            bool(np.all(finite_decoupled["gamma"].to_numpy() < 1.0)),
            float(finite_decoupled["gamma"].max()),
            1.0,
        ),
        (
            "finite-batch optimum is fractional with shared parameters",
            bool(np.all(finite_shared["gamma"].to_numpy() < 1.0)),
            float(finite_shared["gamma"].max()),
            1.0,
        ),
        (
            "optimal gamma rises as stochastic noise is reduced",
            bool(np.all(np.diff(finite_decoupled["gamma"].to_numpy()) >= -1e-12)),
            float(finite_decoupled["gamma"].iloc[-1] - finite_decoupled["gamma"].iloc[0]),
            0.0,
        ),
        (
            "removing irreducible target noise restores full whitening",
            np.isclose(noiseless_optimum, 1.0),
            noiseless_optimum,
            1.0,
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "reference"])


__all__ = [
    "AnalyticOperators",
    "DEFAULT_DECODER_GAIN",
    "DEFAULT_PREDICTABILITY",
    "DEFAULT_RESIDUAL_VARIANCE",
    "WhiteningToy",
    "analytic_operators",
    "analytic_trajectory",
    "make_toy",
    "mechanism_checks",
    "mode_table",
    "monte_carlo_validation",
    "optimal_gamma",
    "run_sweep",
]
