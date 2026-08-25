"""Population tools for Fréchet-complementary score experiments.

The central object is the Wasserstein tangent space of distributions with
fixed mean and covariance.  Projecting ``score_p - score_q`` onto this space
produces the steepest reverse-KL descent direction that does not change those
moments to first order.  This is the shape correction complementary to any
objective that depends only on mean and covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from experiments.advfd_cleanroom.core import (
    Moments,
    frechet_from_moments,
    moments_from_mean_and_second,
    symmetric_matrix_inverse_sqrt,
    symmetric_matrix_sqrt,
)


TensorField = Callable[[torch.Tensor, bool], torch.Tensor]


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.mT)


@dataclass(frozen=True)
class GaussianMixture:
    """A low-dimensional Gaussian mixture with a shared full covariance."""

    weights: torch.Tensor
    means: torch.Tensor
    component_covariance: torch.Tensor

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("weights must have shape [components]")
        if self.means.ndim != 2 or len(self.means) != len(self.weights):
            raise ValueError("means must have shape [components, dimension]")
        dimension = self.means.shape[1]
        if self.component_covariance.shape != (dimension, dimension):
            raise ValueError("component covariance has the wrong shape")
        if torch.any(self.weights <= 0):
            raise ValueError("all mixture weights must be positive")
        if not torch.allclose(
            self.weights.sum(),
            torch.ones((), dtype=self.weights.dtype, device=self.weights.device),
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("mixture weights must sum to one")
        eigenvalues = torch.linalg.eigvalsh(_symmetrize(self.component_covariance))
        if torch.any(eigenvalues <= 0):
            raise ValueError("component covariance must be positive definite")

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    @property
    def components(self) -> int:
        return int(len(self.weights))

    def moments(self) -> Moments:
        mean = (self.weights[:, None] * self.means).sum(dim=0)
        centered = self.means - mean
        covariance = self.component_covariance + torch.einsum(
            "k,ki,kj->ij", self.weights, centered, centered
        )
        return moments_from_mean_and_second(
            mean,
            covariance + torch.outer(mean, mean),
        )

    def log_prob_and_score(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 2 or states.shape[1] != self.dimension:
            raise ValueError("states have the wrong shape")
        covariance = self.component_covariance.to(states)
        precision = torch.linalg.inv(covariance)
        log_det = torch.logdet(covariance)
        residual = states[:, None, :] - self.means.to(states)[None, :, :]
        mahalanobis = torch.einsum("nkd,de,nke->nk", residual, precision, residual)
        log_components = (
            self.weights.to(states).log()[None, :]
            - 0.5 * (
                self.dimension * math.log(2.0 * math.pi)
                + log_det
                + mahalanobis
            )
        )
        log_probability = torch.logsumexp(log_components, dim=1)
        responsibilities = torch.softmax(log_components, dim=1)
        component_scores = -torch.einsum("nkd,de->nke", residual, precision)
        score = (responsibilities[..., None] * component_scores).sum(dim=1)
        return log_probability, score

    def quadrature(self, order: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic Gauss-Hermite points and probability weights."""

        if order < 4:
            raise ValueError("quadrature order must be at least four")
        roots_np, weights_np = np.polynomial.hermite.hermgauss(order)
        roots = torch.as_tensor(roots_np, dtype=self.means.dtype, device=self.means.device)
        weights = torch.as_tensor(
            weights_np, dtype=self.means.dtype, device=self.means.device
        )
        if self.dimension == 1:
            grid = math.sqrt(2.0) * roots[:, None]
            grid_weights = weights / math.sqrt(math.pi)
        elif self.dimension == 2:
            grid_x, grid_y = torch.meshgrid(roots, roots, indexing="ij")
            grid = math.sqrt(2.0) * torch.stack(
                (grid_x.flatten(), grid_y.flatten()), dim=1
            )
            grid_weights = torch.outer(weights, weights).flatten() / math.pi
        else:
            raise ValueError(
                "the current quadrature helper supports one or two dimensions"
            )
        root = torch.linalg.cholesky(self.component_covariance)
        points = self.means[:, None, :] + grid[None, :, :] @ root.mT
        probability_weights = self.weights[:, None] * grid_weights[None, :]
        return points.flatten(0, 1), probability_weights.flatten()

    def sample(self, count: int, *, seed: int) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        labels = torch.multinomial(
            self.weights.cpu(), count, replacement=True, generator=generator
        ).to(self.means.device)
        noise = torch.randn(
            count,
            self.dimension,
            generator=generator,
            dtype=self.means.dtype,
            device="cpu",
        ).to(self.means.device)
        root = torch.linalg.cholesky(self.component_covariance)
        return self.means[labels] + noise @ root.mT

    def affine_pushforward(
        self, matrix: torch.Tensor, shift: torch.Tensor
    ) -> "GaussianMixture":
        if matrix.shape != (self.dimension, self.dimension):
            raise ValueError("matrix has the wrong shape")
        if shift.shape != (self.dimension,):
            raise ValueError("shift has the wrong shape")
        return GaussianMixture(
            weights=self.weights.clone(),
            means=self.means @ matrix.mT + shift,
            component_covariance=matrix @ self.component_covariance @ matrix.mT,
        )

    def convolve_isotropic(self, standard_deviation: float) -> "GaussianMixture":
        if standard_deviation < 0:
            raise ValueError("standard deviation must be nonnegative")
        identity = torch.eye(
            self.dimension,
            dtype=self.component_covariance.dtype,
            device=self.component_covariance.device,
        )
        return GaussianMixture(
            weights=self.weights.clone(),
            means=self.means.clone(),
            component_covariance=self.component_covariance
            + standard_deviation**2 * identity,
        )


@dataclass(frozen=True)
class MomentTangentProjection:
    tangent: torch.Tensor
    normal: torch.Tensor
    translation: torch.Tensor
    symmetric_linear: torch.Tensor
    mean_derivative: torch.Tensor
    covariance_derivative: torch.Tensor
    orthogonality_error: float


def weighted_moments(states: torch.Tensor, weights: torch.Tensor) -> Moments:
    if states.ndim != 2 or weights.shape != (len(states),):
        raise ValueError("states/weights have incompatible shapes")
    normalized = weights / weights.sum()
    mean = (normalized[:, None] * states).sum(dim=0)
    second = torch.einsum("n,ni,nj->ij", normalized, states, states)
    return moments_from_mean_and_second(mean, second)


def weighted_inner(
    first: torch.Tensor, second: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    normalized = weights / weights.sum()
    return (normalized[:, None] * first * second).sum()


def solve_symmetric_lyapunov(
    covariance: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    eigenvalue_floor: float = 1e-10,
) -> torch.Tensor:
    """Solve ``C S + S C = R`` for symmetric ``S``."""

    covariance = _symmetrize(covariance)
    right_hand_side = _symmetrize(right_hand_side)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(eigenvalue_floor)
    rotated = eigenvectors.mT @ right_hand_side @ eigenvectors
    denominator = eigenvalues[:, None] + eigenvalues[None, :]
    solution = eigenvectors @ (rotated / denominator) @ eigenvectors.mT
    return _symmetrize(solution)


def project_onto_fixed_moment_tangent(
    states: torch.Tensor,
    field: torch.Tensor,
    weights: torch.Tensor,
) -> MomentTangentProjection:
    """Project a field onto the fixed-mean/fixed-covariance tangent space.

    The normal space consists of ``a + S (x-m)`` with symmetric ``S``.  The
    Lyapunov solve is required because simply fitting an arbitrary affine map
    removes skew fields that already preserve covariance.
    """

    if states.shape != field.shape or weights.shape != (len(states),):
        raise ValueError("states, field, and weights have incompatible shapes")
    normalized = weights / weights.sum()
    moments = weighted_moments(states, normalized)
    centered = states - moments.mean
    translation = (normalized[:, None] * field).sum(dim=0)
    centered_field = field - translation
    cross = torch.einsum("n,ni,nj->ij", normalized, centered, centered_field)
    symmetric_linear = solve_symmetric_lyapunov(
        moments.covariance, cross + cross.mT
    )
    normal = translation + centered @ symmetric_linear.mT
    tangent = field - normal
    mean_derivative = (normalized[:, None] * tangent).sum(dim=0)
    tangent_cross = torch.einsum("n,ni,nj->ij", normalized, centered, tangent)
    covariance_derivative = tangent_cross + tangent_cross.mT
    tangent_norm = weighted_inner(tangent, tangent, normalized).sqrt()
    normal_norm = weighted_inner(normal, normal, normalized).sqrt()
    if float(tangent_norm) < 1e-12 or float(normal_norm) < 1e-12:
        orthogonality_error = 0.0
    else:
        orthogonality_error = float(
            weighted_inner(tangent, normal, normalized).abs()
            / (tangent_norm * normal_norm)
        )
    return MomentTangentProjection(
        tangent=tangent,
        normal=normal,
        translation=translation,
        symmetric_linear=symmetric_linear,
        mean_derivative=mean_derivative,
        covariance_derivative=covariance_derivative,
        orthogonality_error=orthogonality_error,
    )


def gaussian_transport_field(
    source: Moments,
    target: Moments,
    states: torch.Tensor,
    *,
    eigenvalue_floor: float = 1e-10,
) -> torch.Tensor:
    """Negative W2 gradient of the Gaussian Fréchet objective."""

    source_root = symmetric_matrix_sqrt(
        source.covariance, eigenvalue_floor=eigenvalue_floor
    )
    source_inverse_root = symmetric_matrix_inverse_sqrt(
        source.covariance, eigenvalue_floor=eigenvalue_floor
    )
    middle = symmetric_matrix_sqrt(
        source_root @ target.covariance @ source_root,
        eigenvalue_floor=eigenvalue_floor,
    )
    transport = source_inverse_root @ middle @ source_inverse_root
    return target.mean + (states - source.mean) @ transport.mT - states


def score_correction(
    target: GaussianMixture,
    source: GaussianMixture,
    states: torch.Tensor,
) -> torch.Tensor:
    _, target_score = target.log_prob_and_score(states)
    _, source_score = source.log_prob_and_score(states)
    return target_score - source_score


def density_ratio(
    target: GaussianMixture,
    source: GaussianMixture,
    states: torch.Tensor,
) -> torch.Tensor:
    """Return ``q / p`` for target density ``p`` and source density ``q``."""

    target_log_probability, _ = target.log_prob_and_score(states)
    source_log_probability, _ = source.log_prob_and_score(states)
    return (source_log_probability - target_log_probability).exp()


def log_density_ratio(
    target: GaussianMixture,
    source: GaussianMixture,
    states: torch.Tensor,
) -> torch.Tensor:
    """Return ``log(q / p)`` for target density ``p`` and source density ``q``."""

    target_log_probability, _ = target.log_prob_and_score(states)
    source_log_probability, _ = source.log_prob_and_score(states)
    return source_log_probability - target_log_probability


def pearson_divergence(
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    quadrature_order: int = 20,
) -> float:
    """Evaluate Pearson ``chi^2(q || p)`` by quadrature under ``q``."""

    states, weights = source.quadrature(quadrature_order)
    normalized = weights / weights.sum()
    ratio = density_ratio(target, source, states)
    return float((normalized * ratio).sum() - 1.0)


def pearson_correction(
    target: GaussianMixture,
    source: GaussianMixture,
    states: torch.Tensor,
) -> torch.Tensor:
    """Return the Wasserstein descent field for Pearson ``chi^2(q || p)``.

    With ``r = q / p``, the field is ``-grad r = r (score_p-score_q)``.
    A scalar mean-only Fisher witness induces this field up to a positive
    constant when its function class contains the optimal density-ratio
    witness.
    """

    return density_ratio(target, source, states)[:, None] * score_correction(
        target, source, states
    )


def pooled_fisher_divergence(
    target: GaussianMixture,
    source: GaussianMixture,
    *,
    quadrature_order: int = 20,
) -> float:
    """Return the optimal pooled-Fisher mean discrepancy.

    With ``m=(p+q)/2``, the unrestricted Rayleigh quotient
    ``(E_q h-E_p h)^2 / Var_m(h)`` equals ``integral (q-p)^2/m``: twice
    triangular discrimination.  Its optimal witness is bounded by construction.
    """

    expectations = []
    for distribution in (target, source):
        states, weights = distribution.quadrature(quadrature_order)
        normalized = weights / weights.sum()
        witness = 2.0 * torch.tanh(
            0.5 * log_density_ratio(target, source, states)
        )
        expectations.append((normalized * witness.square()).sum())
    return float(0.5 * (expectations[0] + expectations[1]))


def pooled_fisher_correction(
    target: GaussianMixture,
    source: GaussianMixture,
    states: torch.Tensor,
) -> torch.Tensor:
    """Return W2 descent for the optimal pooled-Fisher discrepancy.

    For ``r=q/p``, the exact field is
    ``16 r/(1+r)^3 (score_p-score_q)``.  The logistic form avoids overflow
    when the source and target have little overlap.
    """

    source_probability = torch.sigmoid(
        log_density_ratio(target, source, states)
    )
    weight = 16.0 * source_probability * (1.0 - source_probability).square()
    return weight[:, None] * score_correction(target, source, states)


def tangent_field_from_projection(
    target: GaussianMixture,
    source: GaussianMixture,
    projection: MomentTangentProjection,
) -> TensorField:
    source_mean = source.moments().mean
    translation = projection.translation.detach()
    symmetric_linear = projection.symmetric_linear.detach()

    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        correction = score_correction(target, source, states)
        normal = translation + (states - source_mean.to(states)) @ symmetric_linear.mT
        return correction - normal

    return field


def score_field(target: GaussianMixture, source: GaussianMixture) -> TensorField:
    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        return score_correction(target, source, states)

    return field


def pearson_field(target: GaussianMixture, source: GaussianMixture) -> TensorField:
    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        return pearson_correction(target, source, states)

    return field


def pooled_fisher_field(
    target: GaussianMixture, source: GaussianMixture
) -> TensorField:
    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        return pooled_fisher_correction(target, source, states)

    return field


def static_field(target: GaussianMixture, source: GaussianMixture) -> TensorField:
    source_moments = source.moments()
    target_moments = target.moments()

    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        return gaussian_transport_field(source_moments, target_moments, states)

    return field


def sum_fields(*fields: TensorField) -> TensorField:
    def combined(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        return sum(
            (field(states, create_graph) for field in fields),
            torch.zeros_like(states),
        )

    return combined


def field_diagnostics(
    target: GaussianMixture,
    source: GaussianMixture,
    field: TensorField,
    *,
    quadrature_order: int = 20,
) -> dict[str, float]:
    states, weights = source.quadrature(quadrature_order)
    velocity = field(states, False)
    correction = score_correction(target, source, states)
    normalized = weights / weights.sum()
    source_moments = source.moments()
    centered = states - source_moments.mean
    mean_derivative = (normalized[:, None] * velocity).sum(dim=0)
    cross = torch.einsum("n,ni,nj->ij", normalized, centered, velocity)
    covariance_derivative = cross + cross.mT
    correction_norm = weighted_inner(correction, correction, normalized).sqrt()
    velocity_norm = weighted_inner(velocity, velocity, normalized).sqrt()
    alignment = weighted_inner(correction, velocity, normalized)
    return {
        "velocity_rms": float(
            (velocity_norm / math.sqrt(source.dimension)).detach()
        ),
        "score_rms": float(
            (correction_norm / math.sqrt(source.dimension)).detach()
        ),
        "score_cosine": float(
            (
                alignment
                / (correction_norm * velocity_norm).clamp_min(
                    torch.finfo(states.dtype).eps
                )
            ).detach()
        ),
        "reverse_kl_derivative": float((-alignment).detach()),
        "mean_derivative_norm": float(mean_derivative.norm().detach()),
        "covariance_derivative_norm": float(
            covariance_derivative.norm().detach()
        ),
    }


def finite_pushforward_kl(
    target: GaussianMixture,
    source: GaussianMixture,
    field: TensorField,
    *,
    step_size: float,
    quadrature_order: int = 20,
) -> dict[str, float]:
    """Evaluate KL after the differentiable map ``x -> x + h v(x)``."""

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    states, weights = source.quadrature(quadrature_order)
    states = states.detach().requires_grad_(True)
    velocity = field(states, True)
    jacobian_rows = []
    for coordinate in range(source.dimension):
        jacobian_rows.append(
            torch.autograd.grad(
                velocity[:, coordinate].sum(),
                states,
                retain_graph=True,
                create_graph=False,
            )[0]
        )
    jacobian = torch.stack(jacobian_rows, dim=1)
    identity = torch.eye(source.dimension, dtype=states.dtype, device=states.device)
    map_jacobian = identity[None, :, :] + step_size * jacobian
    determinant = torch.linalg.det(map_jacobian)
    transported = states + step_size * velocity
    source_log_prob, _ = source.log_prob_and_score(states)
    target_log_prob_before, _ = target.log_prob_and_score(states)
    target_log_prob_after, _ = target.log_prob_and_score(transported)
    normalized = weights / weights.sum()
    before = (normalized * (source_log_prob - target_log_prob_before)).sum()
    valid = determinant > 0
    if not bool(valid.all()):
        return {
            "kl_before": float(before.detach()),
            "kl_after": float("nan"),
            "kl_change": float("nan"),
            "positive_jacobian_fraction": float(valid.double().mean()),
            "minimum_jacobian_determinant": float(determinant.min()),
        }
    after = (
        normalized
        * (source_log_prob - determinant.log() - target_log_prob_after)
    ).sum()
    return {
        "kl_before": float(before.detach()),
        "kl_after": float(after.detach()),
        "kl_change": float((after - before).detach()),
        "positive_jacobian_fraction": 1.0,
        "minimum_jacobian_determinant": float(determinant.min()),
    }


def finite_pushforward_pearson(
    target: GaussianMixture,
    source: GaussianMixture,
    field: TensorField,
    *,
    step_size: float,
    quadrature_order: int = 20,
) -> dict[str, float]:
    """Evaluate Pearson ``chi^2`` after ``x -> x + h v(x)``.

    For an invertible map ``T``, the transported density obeys
    ``q_T(T(x)) = q(x) / det(J_T(x))``.  Evaluating the divergence under the
    original ``q`` quadrature avoids fitting a density to transported points.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    states, weights = source.quadrature(quadrature_order)
    states = states.detach().requires_grad_(True)
    velocity = field(states, True)
    jacobian_rows = []
    for coordinate in range(source.dimension):
        jacobian_rows.append(
            torch.autograd.grad(
                velocity[:, coordinate].sum(),
                states,
                retain_graph=True,
                create_graph=False,
            )[0]
        )
    jacobian = torch.stack(jacobian_rows, dim=1)
    identity = torch.eye(source.dimension, dtype=states.dtype, device=states.device)
    map_jacobian = identity[None, :, :] + step_size * jacobian
    determinant = torch.linalg.det(map_jacobian)
    valid = determinant > 0
    before = pearson_divergence(
        target, source, quadrature_order=quadrature_order
    )
    if not bool(valid.all()):
        return {
            "pearson_before": before,
            "pearson_after": float("nan"),
            "pearson_change": float("nan"),
            "positive_jacobian_fraction": float(valid.double().mean()),
            "minimum_jacobian_determinant": float(determinant.min()),
        }
    transported = states + step_size * velocity
    source_log_probability, _ = source.log_prob_and_score(states)
    target_log_probability, _ = target.log_prob_and_score(transported)
    transported_ratio = (
        source_log_probability - determinant.log() - target_log_probability
    ).exp()
    normalized = weights / weights.sum()
    after = (normalized * transported_ratio).sum() - 1.0
    return {
        "pearson_before": before,
        "pearson_after": float(after.detach()),
        "pearson_change": float(after.detach()) - before,
        "positive_jacobian_fraction": 1.0,
        "minimum_jacobian_determinant": float(determinant.min()),
    }


def build_toy_regimes(
    *, dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu"
) -> dict[str, tuple[GaussianMixture, GaussianMixture]]:
    device = torch.device(device)
    component_variance = 0.18**2
    covariance = component_variance * torch.eye(2, dtype=dtype, device=device)
    angles = torch.arange(8, dtype=dtype, device=device) * (2.0 * math.pi / 8.0)
    means = 2.0 * torch.stack((angles.cos(), angles.sin()), dim=1)
    target_weights = torch.full((8,), 1.0 / 8.0, dtype=dtype, device=device)
    alternating = torch.cos(4.0 * angles)
    shape_weights = target_weights * (1.0 + 0.65 * alternating)
    shape_weights = shape_weights / shape_weights.sum()
    target_shape = GaussianMixture(target_weights, means, covariance)
    source_shape = GaussianMixture(shape_weights, means, covariance)

    gaussian_target = GaussianMixture(
        torch.ones(1, dtype=dtype, device=device),
        torch.zeros(1, 2, dtype=dtype, device=device),
        torch.tensor([[1.0, 0.15], [0.15, 0.8]], dtype=dtype, device=device),
    )
    gaussian_source = GaussianMixture(
        torch.ones(1, dtype=dtype, device=device),
        torch.tensor([[0.45, -0.35]], dtype=dtype, device=device),
        torch.tensor([[1.35, -0.20], [-0.20, 0.62]], dtype=dtype, device=device),
    )
    matrix = torch.tensor([[1.18, 0.22], [-0.08, 0.82]], dtype=dtype, device=device)
    shift = torch.tensor([0.35, -0.25], dtype=dtype, device=device)
    combined_source = source_shape.affine_pushforward(matrix, shift)
    return {
        "gaussian_only": (gaussian_target, gaussian_source),
        "shape_only": (target_shape, source_shape),
        "combined": (target_shape, combined_source),
    }


def frechet_value(target: GaussianMixture, source: GaussianMixture) -> float:
    return float(frechet_from_moments(target.moments(), source.moments()).total)


__all__ = [
    "GaussianMixture",
    "MomentTangentProjection",
    "build_toy_regimes",
    "density_ratio",
    "field_diagnostics",
    "finite_pushforward_kl",
    "finite_pushforward_pearson",
    "frechet_value",
    "gaussian_transport_field",
    "log_density_ratio",
    "pearson_correction",
    "pearson_divergence",
    "pearson_field",
    "pooled_fisher_correction",
    "pooled_fisher_divergence",
    "pooled_fisher_field",
    "project_onto_fixed_moment_tangent",
    "score_correction",
    "score_field",
    "solve_symmetric_lyapunov",
    "static_field",
    "sum_fields",
    "tangent_field_from_projection",
    "weighted_inner",
    "weighted_moments",
]
