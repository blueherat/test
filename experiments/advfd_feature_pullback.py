"""Local feature-to-input pullbacks for an AdvFD Fréchet force."""

from __future__ import annotations

import torch
from torch.func import jacrev, vmap

from experiments.advfd_cleanroom.core import (
    AffineCalibration,
    calibrate_moments,
    frechet_from_moments,
    fit_calibration_from_moments,
    moments_from_mean_and_second,
)
from experiments.frechet_residual_score_toy import weighted_moments
from experiments.run_frechet_residual_score_toy import (
    calibrated_weighted_moments,
)


def build_feature_force_context(
    critic,
    target,
    source,
    *,
    order: int,
    whitening_epsilon: float,
    objective_mode: str = "official_regularized",
):
    """Freeze AdvFD moments and return its force in calibrated feature space."""

    valid_modes = {
        "official_regularized",
        "official_mean_only",
        "official_covariance_only",
        "pooled_full",
        "pooled_mean_only",
        "pooled_covariance_only",
    }
    if objective_mode not in valid_modes:
        raise ValueError(f"unknown objective mode: {objective_mode!r}")

    target_points, target_weights = target.quadrature(order)
    source_points, source_weights = source.quadrature(order)
    with torch.no_grad():
        target_features = critic(target_points)
        source_features = critic(source_points)
        target_raw = weighted_moments(target_features, target_weights)
        source_raw = weighted_moments(source_features, source_weights)
        pooled_mode = objective_mode.startswith("pooled_")
        calibration = fit_calibration_from_moments(
            target_raw,
            source_raw,
            mode="pooled" if pooled_mode else "real",
            epsilon=whitening_epsilon,
            detach_statistics=True,
        )
        source_moments = calibrated_weighted_moments(
            source_features, source_weights, calibration
        )

    mean = source_moments.mean.detach().requires_grad_(True)
    second = source_moments.second.detach().requires_grad_(True)
    variable_source = moments_from_mean_and_second(mean, second)
    if pooled_mode:
        target_moments = calibrate_moments(target_raw, calibration).detached()
        components = frechet_from_moments(target_moments, variable_source)
    else:
        dimension = mean.numel()
        identity = torch.eye(dimension, dtype=mean.dtype, device=mean.device)
        regularizer = (
            whitening_epsilon * calibration.transform.mT @ calibration.transform
        )
        regularized_source = moments_from_mean_and_second(
            mean,
            variable_source.covariance
            + regularizer
            + torch.outer(mean, mean),
        )
        target_identity = moments_from_mean_and_second(
            torch.zeros_like(mean), identity
        )
        components = frechet_from_moments(target_identity, regularized_source)
    if objective_mode in {"official_regularized", "pooled_full"}:
        distance = components.total
    elif objective_mode in {"official_mean_only", "pooled_mean_only"}:
        distance = components.mean
    else:
        distance = components.covariance
    mean_gradient, second_gradient = torch.autograd.grad(
        distance, (mean, second), allow_unused=True
    )
    if mean_gradient is None:
        mean_gradient = torch.zeros_like(mean)
    if second_gradient is None:
        second_gradient = torch.zeros_like(second)
    return {
        "calibration": AffineCalibration(
            calibration.center.detach(), calibration.transform.detach()
        ),
        "mean_gradient": mean_gradient.detach(),
        "second_gradient": second_gradient.detach(),
        "distance": float(distance.detach()),
        "objective_mode": objective_mode,
    }


def feature_potential(
    critic,
    context,
    states: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the fixed-critic functional derivative up to a constant."""

    features = context["calibration"].apply(critic(states))
    linear = features @ context["mean_gradient"]
    quadratic = torch.einsum(
        "ni,ij,nj->n", features, context["second_gradient"], features
    )
    return linear + quadratic


def feature_force_and_jacobian(
    critic,
    context,
    states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return desired feature descent and the calibrated critic Jacobian."""

    calibration = context["calibration"]

    def feature_single(state: torch.Tensor) -> torch.Tensor:
        raw = critic(state[None, :])
        return calibration.apply(raw)[0]

    features = vmap(feature_single)(states)
    jacobians = vmap(jacrev(feature_single))(states)
    symmetric_second = context["second_gradient"] + context["second_gradient"].mT
    feature_gradient = context["mean_gradient"] + features @ symmetric_second
    return -feature_gradient, jacobians


def learned_pullback_field(
    critic,
    context,
    *,
    mode: str,
    relative_damping: float = 0.0,
):
    """Build either the Euclidean transpose or damped least-squares pullback."""

    if mode not in {"transpose", "pseudoinverse"}:
        raise ValueError(f"unknown pullback mode: {mode}")
    if relative_damping < 0:
        raise ValueError("relative damping must be nonnegative")

    def field(states: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
        del create_graph
        feature_force, jacobians = feature_force_and_jacobian(
            critic, context, states
        )
        right_hand_side = torch.einsum("nfi,nf->ni", jacobians, feature_force)
        if mode == "transpose":
            return right_hand_side
        gram = torch.einsum("nfi,nfj->nij", jacobians, jacobians)
        input_dimension = gram.shape[-1]
        local_scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
        numerical_floor = torch.finfo(gram.dtype).eps**0.5
        damping = (relative_damping * local_scale + numerical_floor).view(-1, 1, 1)
        identity = torch.eye(
            input_dimension, dtype=gram.dtype, device=gram.device
        )[None, :, :]
        return torch.linalg.solve(
            gram + damping * identity, right_hand_side[:, :, None]
        )[:, :, 0]

    return field
