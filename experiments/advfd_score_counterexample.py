"""Analytic counterexamples for AdvFD witness gradients versus noised scores.

The construction deliberately separates the value optimized by an adaptive
Frechet representation from the input derivative consumed by the generator.
For finite, disjoint supports, Hermite interpolation lets us prescribe both
independently. Gaussian smoothing then provides well-defined densities and
scores for a matched score-flow control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiscreteDistribution1D:
    atoms: torch.Tensor
    weights: torch.Tensor

    def __post_init__(self) -> None:
        if self.atoms.ndim != 1 or self.weights.shape != self.atoms.shape:
            raise ValueError("atoms and weights must be one-dimensional and aligned")
        if len(self.atoms) == 0:
            raise ValueError("at least one atom is required")
        if torch.any(self.weights <= 0):
            raise ValueError("weights must be positive")
        if not torch.isclose(
            self.weights.sum(), self.weights.new_tensor(1.0), atol=1e-10, rtol=1e-10
        ):
            raise ValueError("weights must sum to one")

    @property
    def mean(self) -> torch.Tensor:
        return (self.weights * self.atoms).sum()

    @property
    def variance(self) -> torch.Tensor:
        return (self.weights * (self.atoms - self.mean).square()).sum()


def moment_matched_disjoint_pair(
    *, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu"
) -> tuple[DiscreteDistribution1D, DiscreteDistribution1D]:
    """Return distinct, disjoint-support distributions with mean 0 and variance 1."""

    real = DiscreteDistribution1D(
        atoms=torch.tensor([-1.0, 1.0], dtype=dtype, device=device),
        weights=torch.tensor([0.5, 0.5], dtype=dtype, device=device),
    )
    fake = DiscreteDistribution1D(
        atoms=torch.tensor([-0.5, 2.0], dtype=dtype, device=device),
        weights=torch.tensor([0.8, 0.2], dtype=dtype, device=device),
    )
    return real, fake


def frechet_distance_1d(
    real_mean: torch.Tensor,
    real_variance: torch.Tensor,
    fake_mean: torch.Tensor,
    fake_variance: torch.Tensor,
) -> torch.Tensor:
    """Squared Gaussian Wasserstein distance in one dimension."""

    if torch.any(real_variance < 0) or torch.any(fake_variance < 0):
        raise ValueError("variances must be non-negative")
    return (real_mean - fake_mean).square() + (
        real_variance.sqrt() - fake_variance.sqrt()
    ).square()


def feature_moments(
    features: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 1 or weights.shape != features.shape:
        raise ValueError("features and weights must be aligned vectors")
    mean = (weights * features).sum()
    variance = (weights * (features - mean).square()).sum()
    return mean, variance


def hermite_polynomial_coefficients(
    nodes: torch.Tensor,
    values: torch.Tensor,
    derivatives: torch.Tensor,
) -> torch.Tensor:
    """Fit the unique degree ``2*n-1`` polynomial with prescribed first jets."""

    if nodes.ndim != 1 or values.shape != nodes.shape or derivatives.shape != nodes.shape:
        raise ValueError("nodes, values, and derivatives must be aligned vectors")
    if len(torch.unique(nodes)) != len(nodes):
        raise ValueError("Hermite nodes must be distinct")
    count = len(nodes)
    degree_count = 2 * count
    powers = torch.arange(degree_count, device=nodes.device, dtype=nodes.dtype)
    value_rows = nodes[:, None].pow(powers[None, :])
    derivative_rows = torch.zeros_like(value_rows)
    derivative_rows[:, 1:] = (
        powers[None, 1:] * nodes[:, None].pow(powers[None, 1:] - 1.0)
    )
    matrix = torch.empty(
        2 * count, degree_count, device=nodes.device, dtype=nodes.dtype
    )
    target = torch.empty(2 * count, device=nodes.device, dtype=nodes.dtype)
    matrix[0::2] = value_rows
    matrix[1::2] = derivative_rows
    target[0::2] = values
    target[1::2] = derivatives
    return torch.linalg.solve(matrix, target)


def polynomial_value(inputs: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(inputs)
    for coefficient in coefficients.flip(0):
        result = result * inputs + coefficient
    return result


def polynomial_derivative(
    inputs: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    if len(coefficients) <= 1:
        return torch.zeros_like(inputs)
    powers = torch.arange(
        1, len(coefficients), device=coefficients.device, dtype=coefficients.dtype
    )
    return polynomial_value(inputs, coefficients[1:] * powers)


def build_advfd_witness(
    amplitude: float,
    *,
    fake_derivatives: tuple[float, float] = (0.0, 0.0),
    real_derivatives: tuple[float, float] = (0.0, 0.0),
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct a smooth scalar feature with fixed values and arbitrary fake jets."""

    real, fake = moment_matched_disjoint_pair(dtype=dtype, device=device)
    nodes = torch.cat((real.atoms, fake.atoms))
    values = torch.tensor(
        [-1.0, 1.0, float(amplitude), float(amplitude)],
        dtype=dtype,
        device=device,
    )
    derivatives = torch.tensor(
        [*real_derivatives, *fake_derivatives], dtype=dtype, device=device
    )
    return hermite_polynomial_coefficients(nodes, values, derivatives)


def paper_regularized_real_whitened_fd_1d(
    real_mean: torch.Tensor,
    real_variance: torch.Tensor,
    fake_mean: torch.Tensor,
    fake_variance: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Equation (33): apply the same regularized real affine map to both sides."""

    denominator = real_variance + float(epsilon)
    real_white_variance = real_variance / denominator
    fake_white_variance = fake_variance / denominator
    mean_term = (fake_mean - real_mean).square() / denominator
    return mean_term + (
        real_white_variance.sqrt() - fake_white_variance.sqrt()
    ).square()


def official_loaded_real_whitened_fd_1d(
    real_mean: torch.Tensor,
    real_variance: torch.Tensor,
    fake_mean: torch.Tensor,
    fake_variance: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Match the released helper, which adds epsilon to fake covariance too."""

    denominator = real_variance + float(epsilon)
    fake_white_variance = (fake_variance + float(epsilon)) / denominator
    mean_term = (fake_mean - real_mean).square() / denominator
    return mean_term + (1.0 - fake_white_variance.sqrt()).square()


def witness_statistics(
    coefficients: torch.Tensor,
) -> dict[str, torch.Tensor]:
    real, fake = moment_matched_disjoint_pair(
        dtype=coefficients.dtype, device=coefficients.device
    )
    real_features = polynomial_value(real.atoms, coefficients)
    fake_features = polynomial_value(fake.atoms, coefficients)
    real_mean, real_variance = feature_moments(real_features, real.weights)
    fake_mean, fake_variance = feature_moments(fake_features, fake.weights)
    return {
        "real_features": real_features,
        "fake_features": fake_features,
        "real_mean": real_mean,
        "real_variance": real_variance,
        "fake_mean": fake_mean,
        "fake_variance": fake_variance,
        "real_derivatives": polynomial_derivative(real.atoms, coefficients),
        "fake_derivatives": polynomial_derivative(fake.atoms, coefficients),
    }


def witness_generator_gradient(
    coefficients: torch.Tensor,
    *,
    epsilon: float,
    normalization_epsilon: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate a frozen witness loss with respect to the fake atom locations."""

    real, fake = moment_matched_disjoint_pair(
        dtype=coefficients.dtype, device=coefficients.device
    )
    fake_atoms = fake.atoms.detach().clone().requires_grad_(True)
    real_features = polynomial_value(real.atoms, coefficients.detach())
    fake_features = polynomial_value(fake_atoms, coefficients.detach())
    real_mean, real_variance = feature_moments(real_features, real.weights)
    fake_mean, fake_variance = feature_moments(fake_features, fake.weights)
    distance = official_loaded_real_whitened_fd_1d(
        real_mean,
        real_variance,
        fake_mean,
        fake_variance,
        epsilon=epsilon,
    )
    loss = distance
    if normalization_epsilon is not None:
        loss = distance / (distance.detach() + float(normalization_epsilon))
    gradient = torch.autograd.grad(loss, fake_atoms)[0]
    return distance.detach(), gradient.detach()


def shared_support_pearson_control(
    *, dtype: torch.dtype = torch.float64
) -> dict[str, torch.Tensor]:
    """A control where real-only standardized mean matching equals Pearson chi-square."""

    p = torch.tensor([0.5, 0.5], dtype=dtype)
    q = torch.tensor([0.8, 0.2], dtype=dtype)
    witness = torch.tensor([-1.0, 1.0], dtype=dtype)
    objective = (q @ witness).square()
    pearson = ((q - p).square() / p).sum()
    return {"objective": objective, "pearson_chi_square": pearson}


def gaussian_mixture_log_prob_and_score(
    states: torch.Tensor,
    distribution: DiscreteDistribution1D,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if states.ndim != 1:
        raise ValueError("states must be a vector")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    residual = states[:, None] - distribution.atoms[None, :]
    log_components = (
        distribution.weights.log()[None, :]
        - 0.5 * math.log(2.0 * math.pi * sigma**2)
        - 0.5 * residual.square() / sigma**2
    )
    log_probability = torch.logsumexp(log_components, dim=1)
    responsibilities = torch.softmax(log_components, dim=1)
    score = (responsibilities * (-residual / sigma**2)).sum(dim=1)
    return log_probability, score


def integration_grid(
    real: DiscreteDistribution1D,
    fake: DiscreteDistribution1D,
    *,
    sigma: float,
    points: int,
    tail_sigmas: float = 10.0,
) -> torch.Tensor:
    if points < 1001 or points % 2 == 0:
        raise ValueError("points must be an odd integer >= 1001")
    left = torch.minimum(real.atoms.min(), fake.atoms.min()) - tail_sigmas * sigma
    right = torch.maximum(real.atoms.max(), fake.atoms.max()) + tail_sigmas * sigma
    return torch.linspace(
        float(left),
        float(right),
        points,
        dtype=real.atoms.dtype,
        device=real.atoms.device,
    )


def noised_reverse_kl_and_score_metrics(
    real: DiscreteDistribution1D,
    fake: DiscreteDistribution1D,
    *,
    sigma: float,
    grid_points: int = 40_001,
    step_factor: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Evaluate KL dissipation and one finite clean-atom score update."""

    grid = integration_grid(real, fake, sigma=sigma, points=grid_points)
    log_real, score_real = gaussian_mixture_log_prob_and_score(
        grid, real, sigma=sigma
    )
    log_fake, score_fake = gaussian_mixture_log_prob_and_score(
        grid, fake, sigma=sigma
    )
    fake_density = log_fake.exp()
    score_delta = score_real - score_fake
    reverse_kl = torch.trapezoid(fake_density * (log_fake - log_real), grid)
    fisher_divergence = torch.trapezoid(
        fake_density * score_delta.square(), grid
    )

    component_directions = []
    for atom in fake.atoms:
        component_density = torch.exp(-0.5 * ((grid - atom) / sigma).square())
        component_density = component_density / (math.sqrt(2.0 * math.pi) * sigma)
        component_directions.append(
            torch.trapezoid(component_density * score_delta, grid)
        )
    direction = torch.stack(component_directions)
    parameterized_dissipation = (fake.weights * direction.square()).sum()
    directional_derivative = -parameterized_dissipation

    step_size = float(step_factor) * sigma**2
    updated_fake = DiscreteDistribution1D(
        atoms=fake.atoms + step_size * direction,
        weights=fake.weights,
    )
    updated_log_fake, _ = gaussian_mixture_log_prob_and_score(
        grid, updated_fake, sigma=sigma
    )
    updated_density = updated_log_fake.exp()
    updated_reverse_kl = torch.trapezoid(
        updated_density * (updated_log_fake - log_real), grid
    )
    return {
        "reverse_kl": reverse_kl,
        "fisher_divergence": fisher_divergence,
        "continuity_kl_derivative": -fisher_divergence,
        "component_directions": direction,
        "parameterized_kl_derivative": directional_derivative,
        "step_size": reverse_kl.new_tensor(step_size),
        "updated_reverse_kl": updated_reverse_kl,
        "reverse_kl_change": updated_reverse_kl - reverse_kl,
        "updated_fake_atoms": updated_fake.atoms,
    }
