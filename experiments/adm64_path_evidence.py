#!/usr/bin/env python3
"""Numerically stable primitives for ADM same-covariance path evidence.

The functions in this module do not define the high-noise alternative.  They
only implement the part that is common to every admissible alternative:

    P_k = Normal(mu_k, diag(v_k))
    Q_k = Normal(mu_k + delta_k, diag(v_k)).

For an ADM score-difference alternative it is convenient to write
``delta_k = v_k * theta_k``.  The corresponding whitened mean shift is
``sqrt(v_k) * theta_k`` and its conditional KL is half its squared norm.

All reductions are performed per sample in float64.  Callers must construct
``delta_k`` before drawing the current transition noise.  The deterministic
final ADM step is outside this interface and must use Q=P.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class TemperedMeanShift:
    """A predictable score-direction mean shift and its KL diagnostics."""

    mean_shift: torch.Tensor
    raw_whitened_shift: torch.Tensor
    whitened_shift: torch.Tensor
    scale: torch.Tensor
    raw_kl: torch.Tensor
    applied_kl: torch.Tensor


@dataclass(frozen=True)
class LogLikelihoodIncrement:
    """Per-sample log(Q/P) increment and its two constituent terms."""

    value: torch.Tensor
    innovation_projection: torch.Tensor
    conditional_kl: torch.Tensor


@dataclass(frozen=True)
class AdditiveHeatShiftMapping:
    """Discrete original-timestep approximation to ``nu_plus = nu + delta``."""

    current_timestep: np.ndarray
    shifted_timestep: np.ndarray
    current_heat_variance: np.ndarray
    target_heat_variance: np.ndarray
    shifted_heat_variance: np.ndarray
    actual_heat_shift: np.ndarray
    absolute_mapping_error: np.ndarray


def _validate_matching_floating_tensors(*tensors: torch.Tensor) -> None:
    if not tensors:
        raise ValueError("at least one tensor is required")
    reference_shape = tensors[0].shape
    if len(reference_shape) < 2:
        raise ValueError("expected a leading batch dimension and at least one event dimension")
    for tensor in tensors:
        if tensor.shape != reference_shape:
            raise ValueError(
                f"all tensors must have the same shape: {reference_shape} != {tensor.shape}"
            )
        if not tensor.is_floating_point():
            raise TypeError(f"expected floating-point tensors, found {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError("all tensor values must be finite")


def _sum_event_float64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float64).reshape(tensor.shape[0], -1).sum(dim=1)


def _expand_batch_scalar(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if value.shape != (target.shape[0],):
        raise ValueError(f"expected one scalar per sample, found shape {tuple(value.shape)}")
    return value.reshape(value.shape[0], *([1] * (target.ndim - 1)))


def normalized_heat_variance(alpha_bar: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return the additive heat time ``nu=(1-alpha_bar)/alpha_bar``."""

    if isinstance(alpha_bar, torch.Tensor):
        if not alpha_bar.is_floating_point() or not torch.isfinite(alpha_bar).all():
            raise ValueError("alpha_bar must contain finite floating-point values")
        if torch.any(alpha_bar <= 0) or torch.any(alpha_bar >= 1):
            raise ValueError("alpha_bar must lie strictly between zero and one")
        return (1 - alpha_bar) / alpha_bar
    values = np.asarray(alpha_bar)
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("alpha_bar must contain finite floating-point values")
    if np.any(values <= 0) or np.any(values >= 1):
        raise ValueError("alpha_bar must lie strictly between zero and one")
    return (1 - values) / values


def nearest_additive_heat_shift(
    original_alpha_bar: np.ndarray,
    current_timesteps: Sequence[int] | np.ndarray,
    additive_heat_shift: float,
) -> AdditiveHeatShiftMapping:
    """Map a fixed additive heat shift to nearest valid discrete timesteps.

    The model is evaluated only at its original integer training timesteps.
    For each current timestep ``t``, this chooses ``t_plus >= t`` whose heat
    variance is closest to ``nu_t + additive_heat_shift``.  A tie is resolved
    toward the higher-noise timestep.  At the maximum training timestep the
    mapping is necessarily the identity, so callers must set Q=P there.
    """

    alpha_bar = np.asarray(original_alpha_bar, dtype=np.float64)
    if alpha_bar.ndim != 1 or alpha_bar.size < 2:
        raise ValueError("original_alpha_bar must be a one-dimensional schedule")
    heat_variance = np.asarray(normalized_heat_variance(alpha_bar), dtype=np.float64)
    if not np.all(np.diff(heat_variance) > 0):
        raise ValueError("the heat-variance schedule must be strictly increasing")
    if not math.isfinite(additive_heat_shift) or additive_heat_shift <= 0:
        raise ValueError("additive_heat_shift must be finite and strictly positive")

    current = np.asarray(current_timesteps, dtype=np.int64)
    if current.ndim != 1 or current.size == 0:
        raise ValueError("current_timesteps must be a non-empty one-dimensional sequence")
    if np.any(current < 0) or np.any(current >= alpha_bar.size):
        raise ValueError("current timestep is outside the original schedule")

    targets = heat_variance[current] + additive_heat_shift
    shifted = np.empty_like(current)
    for index, (timestep, target) in enumerate(zip(current.tolist(), targets.tolist())):
        insertion = int(np.searchsorted(heat_variance, target, side="left"))
        candidates = {int(timestep), min(int(alpha_bar.size - 1), insertion)}
        if insertion - 1 >= timestep:
            candidates.add(insertion - 1)
        # The secondary key picks the higher-noise timestep on an exact tie.
        shifted[index] = min(
            candidates,
            key=lambda candidate: (abs(float(heat_variance[candidate]) - target), -candidate),
        )

    shifted_nu = heat_variance[shifted]
    current_nu = heat_variance[current]
    return AdditiveHeatShiftMapping(
        current_timestep=current,
        shifted_timestep=shifted,
        current_heat_variance=current_nu,
        target_heat_variance=targets,
        shifted_heat_variance=shifted_nu,
        actual_heat_shift=shifted_nu - current_nu,
        absolute_mapping_error=np.abs(shifted_nu - targets),
    )


def normalized_heat_score_pullback_difference(
    epsilon_current: torch.Tensor,
    epsilon_shifted: torch.Tensor,
    alpha_bar_current: torch.Tensor,
    alpha_bar_shifted: torch.Tensor,
) -> torch.Tensor:
    """Compute the cross-scale score-ratio gradient in the current VP state.

    Let ``x=a*z`` at the current scale and ``x_plus=a_plus*z`` at the shifted
    scale, with ``rho=a_plus/a``.  ``epsilon_shifted`` must be the denoiser
    output evaluated at ``x_plus=rho*x``, not at the unchanged raw state.  If
    ``s_t(x)=-epsilon_t(x)/sqrt(1-alpha_bar_t)``, this function returns

        rho * s_plus(rho*x) - s_current(x),

    which is the gradient with respect to current ``x`` of the normalized-heat
    density ratio at the same ``z``.  Classifier gradients do not belong in the
    primary heat-semigroup quantity.
    """

    _validate_matching_floating_tensors(epsilon_current, epsilon_shifted)
    batch = epsilon_current.shape[0]
    for name, alpha in (
        ("alpha_bar_current", alpha_bar_current),
        ("alpha_bar_shifted", alpha_bar_shifted),
    ):
        if alpha.shape != (batch,) or not alpha.is_floating_point():
            raise ValueError(f"{name} must have shape [batch] and floating dtype")
        if not torch.isfinite(alpha).all() or torch.any(alpha <= 0) or torch.any(alpha >= 1):
            raise ValueError(f"{name} must lie strictly between zero and one")
    if torch.any(alpha_bar_shifted > alpha_bar_current):
        raise ValueError("the shifted scale must be at least as noisy as the current scale")

    current_alpha64 = alpha_bar_current.to(torch.float64)
    shifted_alpha64 = alpha_bar_shifted.to(torch.float64)
    rho = torch.sqrt(shifted_alpha64 / current_alpha64)
    current_noise_scale = torch.sqrt(1 - current_alpha64)
    shifted_noise_scale = torch.sqrt(1 - shifted_alpha64)
    expanded_rho = _expand_batch_scalar(rho, epsilon_current)
    expanded_current_noise = _expand_batch_scalar(current_noise_scale, epsilon_current)
    expanded_shifted_noise = _expand_batch_scalar(shifted_noise_scale, epsilon_current)
    current_score = -epsilon_current.to(torch.float64) / expanded_current_noise
    pulled_back_shifted_score = (
        -expanded_rho * epsilon_shifted.to(torch.float64) / expanded_shifted_noise
    )
    return pulled_back_shifted_score - current_score


def kl_tempered_score_mean_shift(
    score_difference: torch.Tensor,
    variance: torch.Tensor,
    max_conditional_kl: float,
) -> TemperedMeanShift:
    """Turn a score difference into ``delta=v*theta`` with a predictable KL cap.

    ``score_difference`` and ``variance`` have shape ``[batch, ...]``.  The
    untempered alternative has mean shift ``variance * score_difference``.
    Each sample is scaled by the smallest factor in ``(0, 1]`` needed to make

        0.5 * sum(delta**2 / variance) <= max_conditional_kl.

    This operation is admissible only when called before the current Gaussian
    innovation is observed.  A zero score difference remains zero and receives
    scale one.
    """

    _validate_matching_floating_tensors(score_difference, variance)
    if not math.isfinite(max_conditional_kl) or max_conditional_kl <= 0:
        raise ValueError("max_conditional_kl must be finite and strictly positive")
    if not torch.all(variance > 0):
        raise ValueError("all diagonal variances must be strictly positive")

    variance64 = variance.to(torch.float64)
    return kl_tempered_score_mean_shift_from_standard_deviation(
        score_difference,
        torch.sqrt(variance64),
        max_conditional_kl,
    )


def kl_tempered_score_mean_shift_from_standard_deviation(
    score_difference: torch.Tensor,
    standard_deviation: torch.Tensor,
    max_conditional_kl: float,
) -> TemperedMeanShift:
    """Construct the same tilt directly from the implemented noise multiplier.

    This is the preferred sampler-facing interface.  If a transition is
    implemented as ``mu + sigma * epsilon``, passing that exact stored
    ``sigma`` avoids a needless floating-point square followed by a square
    root.  The alternative is

        delta = sigma**2 * theta,
        Sigma = diag(sigma**2),

    and its raw whitened shift is exactly ``sigma * theta`` in the promoted
    float64 arithmetic used for all likelihood-ratio reductions.
    """

    _validate_matching_floating_tensors(score_difference, standard_deviation)
    if not math.isfinite(max_conditional_kl) or max_conditional_kl <= 0:
        raise ValueError("max_conditional_kl must be finite and strictly positive")
    if not torch.all(standard_deviation > 0):
        raise ValueError("all diagonal standard deviations must be strictly positive")

    standard_deviation64 = standard_deviation.to(torch.float64)
    score64 = score_difference.to(torch.float64)
    raw_whitened = standard_deviation64 * score64
    raw_kl = 0.5 * _sum_event_float64(raw_whitened.square())

    cap = torch.full_like(raw_kl, float(max_conditional_kl))
    scale = torch.ones_like(raw_kl)
    positive = raw_kl > 0
    scale[positive] = torch.minimum(
        scale[positive], torch.sqrt(cap[positive] / raw_kl[positive])
    )
    expanded_scale = _expand_batch_scalar(scale, score64)
    whitened_shift = raw_whitened * expanded_scale
    mean_shift = standard_deviation64 * whitened_shift
    applied_kl = 0.5 * _sum_event_float64(whitened_shift.square())
    return TemperedMeanShift(
        mean_shift=mean_shift,
        raw_whitened_shift=raw_whitened,
        whitened_shift=whitened_shift,
        scale=scale,
        raw_kl=raw_kl,
        applied_kl=applied_kl,
    )


def same_covariance_log_lr_from_noise(
    whitened_shift: torch.Tensor,
    sampled_noise: torch.Tensor,
) -> LogLikelihoodIncrement:
    """Compute log(Q/P) from the exact standard-normal driving innovation."""

    _validate_matching_floating_tensors(whitened_shift, sampled_noise)
    projection = _sum_event_float64(
        whitened_shift.to(torch.float64) * sampled_noise.to(torch.float64)
    )
    conditional_kl = 0.5 * _sum_event_float64(
        whitened_shift.to(torch.float64).square()
    )
    return LogLikelihoodIncrement(
        value=projection - conditional_kl,
        innovation_projection=projection,
        conditional_kl=conditional_kl,
    )


def same_covariance_log_lr_from_state(
    mean_shift: torch.Tensor,
    variance: torch.Tensor,
    next_state: torch.Tensor,
    p_mean: torch.Tensor,
) -> LogLikelihoodIncrement:
    """Compute log(Q/P) from an observed next state.

    This is algebraically identical to :func:`same_covariance_log_lr_from_noise`
    in exact arithmetic.  The noise form is preferred inside the sampler because
    subtracting ``next_state - p_mean`` can lose precision when the variance is
    small.
    """

    _validate_matching_floating_tensors(mean_shift, variance, next_state, p_mean)
    if not torch.all(variance > 0):
        raise ValueError("all diagonal variances must be strictly positive")
    delta64 = mean_shift.to(torch.float64)
    variance64 = variance.to(torch.float64)
    innovation64 = next_state.to(torch.float64) - p_mean.to(torch.float64)
    projection = _sum_event_float64(delta64 * innovation64 / variance64)
    conditional_kl = 0.5 * _sum_event_float64(delta64.square() / variance64)
    return LogLikelihoodIncrement(
        value=projection - conditional_kl,
        innovation_projection=projection,
        conditional_kl=conditional_kl,
    )


def log_e_mixture(
    component_log_e: torch.Tensor,
    weights: Sequence[float] | torch.Tensor,
    *,
    component_dim: int = -1,
) -> torch.Tensor:
    """Return ``log(sum_j weight_j * E_j)`` without exponentiating ``E_j``."""

    if not component_log_e.is_floating_point():
        raise TypeError("component_log_e must be floating point")
    if not torch.isfinite(component_log_e).all():
        raise ValueError("component_log_e must be finite")
    normalized_dim = component_dim % component_log_e.ndim
    component_count = component_log_e.shape[normalized_dim]
    weight_tensor = torch.as_tensor(
        weights, dtype=torch.float64, device=component_log_e.device
    )
    if weight_tensor.shape != (component_count,):
        raise ValueError(
            f"expected {component_count} mixture weights, found {tuple(weight_tensor.shape)}"
        )
    if not torch.isfinite(weight_tensor).all() or torch.any(weight_tensor < 0):
        raise ValueError("mixture weights must be finite and non-negative")
    total = weight_tensor.sum()
    if not torch.isclose(total, torch.tensor(1.0, dtype=torch.float64, device=total.device)):
        raise ValueError(f"mixture weights must sum to one, found {float(total):.17g}")
    if torch.any(weight_tensor == 0):
        log_weights = torch.where(
            weight_tensor > 0,
            torch.log(weight_tensor),
            torch.full_like(weight_tensor, -torch.inf),
        )
    else:
        log_weights = torch.log(weight_tensor)
    view_shape = [1] * component_log_e.ndim
    view_shape[normalized_dim] = component_count
    return torch.logsumexp(
        component_log_e.to(torch.float64) + log_weights.reshape(view_shape),
        dim=normalized_dim,
    )


def first_log_e_crossing(cumulative_log_e: torch.Tensor, alpha: float) -> torch.Tensor:
    """Return the first crossing index per path, or -1 when no crossing occurs."""

    if cumulative_log_e.ndim != 2:
        raise ValueError("cumulative_log_e must have shape [batch, checks]")
    if not cumulative_log_e.is_floating_point() or not torch.isfinite(cumulative_log_e).all():
        raise ValueError("cumulative_log_e must be finite floating-point values")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    crossed = cumulative_log_e.to(torch.float64) >= -math.log(alpha)
    first = crossed.to(torch.int64).argmax(dim=1)
    any_crossed = crossed.any(dim=1)
    return torch.where(any_crossed, first, torch.full_like(first, -1))


def run_self_test() -> dict[str, float | int | bool]:
    """Exercise exact-density, tempering, mixture, and martingale identities."""

    generator = torch.Generator(device="cpu").manual_seed(20260826)
    batch, dimensions = 7, 19
    p_mean = torch.randn(batch, dimensions, generator=generator, dtype=torch.float64)
    variance = torch.exp(
        0.4 * torch.randn(batch, dimensions, generator=generator, dtype=torch.float64)
    )
    score_difference = 0.2 * torch.randn(
        batch, dimensions, generator=generator, dtype=torch.float64
    )
    tempered = kl_tempered_score_mean_shift(score_difference, variance, 0.15)
    noise = torch.randn(batch, dimensions, generator=generator, dtype=torch.float64)
    next_state = p_mean + torch.sqrt(variance) * noise

    from_noise = same_covariance_log_lr_from_noise(tempered.whitened_shift, noise)
    from_state = same_covariance_log_lr_from_state(
        tempered.mean_shift, variance, next_state, p_mean
    )
    if not torch.allclose(from_noise.value, from_state.value, atol=2e-12, rtol=2e-12):
        raise AssertionError("noise- and state-based likelihood ratios disagree")

    q_mean = p_mean + tempered.mean_shift
    p_dist = torch.distributions.Normal(p_mean, torch.sqrt(variance))
    q_dist = torch.distributions.Normal(q_mean, torch.sqrt(variance))
    direct = (q_dist.log_prob(next_state) - p_dist.log_prob(next_state)).sum(dim=1)
    if not torch.allclose(from_noise.value, direct, atol=2e-12, rtol=2e-12):
        raise AssertionError("closed-form likelihood ratio disagrees with Normal.log_prob")
    if torch.any(tempered.applied_kl > 0.15 + 2e-13):
        raise AssertionError("KL tempering exceeded its cap")

    component_log_e = torch.tensor(
        [[-1.2, 0.4, 2.0], [900.0, 899.0, 898.0]], dtype=torch.float64
    )
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    stable_mixture = log_e_mixture(component_log_e, weights)
    expected_mixture = torch.logsumexp(component_log_e + torch.log(weights), dim=1)
    if not torch.equal(stable_mixture, expected_mixture):
        raise AssertionError("log-space mixture calculation is inconsistent")

    # A low-KL likelihood ratio has a well-behaved Monte Carlo mean.  This is
    # only a numerical smoke test; the martingale identity itself is analytic.
    draws = 300_000
    squared_norm = 0.2
    scalar_noise = torch.randn(draws, generator=generator, dtype=torch.float64)
    likelihood_ratio = torch.exp(math.sqrt(squared_norm) * scalar_noise - squared_norm / 2)
    monte_carlo_mean = float(likelihood_ratio.mean())
    if abs(monte_carlo_mean - 1.0) > 0.005:
        raise AssertionError("Monte Carlo likelihood-ratio mean is unexpectedly far from one")

    crossing_input = torch.tensor(
        [[0.0, 1.0, 3.1], [0.0, 1.0, 2.0]], dtype=torch.float64
    )
    crossings = first_log_e_crossing(crossing_input, alpha=0.05)
    if crossings.tolist() != [2, -1]:
        raise AssertionError("first-crossing calculation is incorrect")

    # Analytic VP-to-heat coordinate audit.  A Gaussian clean density remains
    # Gaussian under the normalized heat flow, so the ratio gradient is exact.
    alpha_current = torch.tensor([0.73, 0.61], dtype=torch.float64)
    alpha_shifted = torch.tensor([0.41, 0.29], dtype=torch.float64)
    clean_mean = torch.tensor([[0.3], [-0.2]], dtype=torch.float64)
    clean_variance = torch.tensor([[0.7], [1.1]], dtype=torch.float64)
    z = torch.tensor([[1.4], [-0.8]], dtype=torch.float64)
    a_current = torch.sqrt(alpha_current).reshape(-1, 1)
    a_shifted = torch.sqrt(alpha_shifted).reshape(-1, 1)
    b_current = torch.sqrt(1 - alpha_current).reshape(-1, 1)
    b_shifted = torch.sqrt(1 - alpha_shifted).reshape(-1, 1)
    nu_current = normalized_heat_variance(alpha_current).reshape(-1, 1)
    nu_shifted = normalized_heat_variance(alpha_shifted).reshape(-1, 1)
    x_current = a_current * z
    x_shifted = a_shifted * z

    score_z_current = -(z - clean_mean) / (clean_variance + nu_current)
    score_z_shifted = -(z - clean_mean) / (clean_variance + nu_shifted)
    score_x_current = score_z_current / a_current
    score_x_shifted = score_z_shifted / a_shifted
    epsilon_current = -b_current * score_x_current
    epsilon_shifted = -b_shifted * score_x_shifted
    coordinate_theta = normalized_heat_score_pullback_difference(
        epsilon_current,
        epsilon_shifted,
        alpha_current,
        alpha_shifted,
    )
    direct_ratio_gradient = (score_z_shifted - score_z_current) / a_current
    coordinate_error = float((coordinate_theta - direct_ratio_gradient).abs().max())
    if coordinate_error > 2e-14:
        raise AssertionError("normalized-heat score pullback failed the Gaussian identity")

    # Feeding the unchanged raw x to the higher-noise model compares a
    # different z-position.  This deliberately verifies that the naive formula
    # is not accidentally equivalent to the required pullback.
    naive_shifted_score = -(
        x_current - a_shifted * clean_mean
    ) / (a_shifted.square() * (clean_variance + nu_shifted))
    naive_theta = naive_shifted_score - score_x_current
    naive_error = float((naive_theta - direct_ratio_gradient).abs().max())
    if naive_error < 1e-3:
        raise AssertionError("the naive same-raw-x score unexpectedly passed the coordinate audit")

    # Repeat the identity on a genuinely multimodal clean density.  This rules
    # out accidentally passing only because a single Gaussian has a linear
    # score field.
    mixture_z = torch.linspace(-3.0, 3.0, 17, dtype=torch.float64).reshape(-1, 1)
    mixture_means = torch.tensor([-1.2, 0.4, 1.8], dtype=torch.float64).reshape(1, -1)
    mixture_variances = torch.tensor([0.18, 0.45, 0.25], dtype=torch.float64).reshape(1, -1)
    mixture_log_weights = torch.log(
        torch.tensor([0.25, 0.50, 0.25], dtype=torch.float64)
    ).reshape(1, -1)

    def mixture_heat_score(points: torch.Tensor, heat_time: float) -> torch.Tensor:
        component_variance = mixture_variances + heat_time
        displacement = points - mixture_means
        component_log_density = (
            mixture_log_weights
            - 0.5 * torch.log(2 * math.pi * component_variance)
            - 0.5 * displacement.square() / component_variance
        )
        responsibilities = torch.softmax(component_log_density, dim=1)
        return (
            responsibilities * (-displacement / component_variance)
        ).sum(dim=1, keepdim=True)

    mixture_alpha_current = 0.68
    mixture_alpha_shifted = 0.37
    mixture_a_current = math.sqrt(mixture_alpha_current)
    mixture_a_shifted = math.sqrt(mixture_alpha_shifted)
    mixture_b_current = math.sqrt(1 - mixture_alpha_current)
    mixture_b_shifted = math.sqrt(1 - mixture_alpha_shifted)
    mixture_nu_current = (1 - mixture_alpha_current) / mixture_alpha_current
    mixture_nu_shifted = (1 - mixture_alpha_shifted) / mixture_alpha_shifted
    mixture_score_current_z = mixture_heat_score(mixture_z, mixture_nu_current)
    mixture_score_shifted_z = mixture_heat_score(mixture_z, mixture_nu_shifted)
    mixture_score_current_x = mixture_score_current_z / mixture_a_current
    mixture_score_shifted_x = mixture_score_shifted_z / mixture_a_shifted
    mixture_epsilon_current = -mixture_b_current * mixture_score_current_x
    mixture_epsilon_shifted = -mixture_b_shifted * mixture_score_shifted_x
    mixture_theta = normalized_heat_score_pullback_difference(
        mixture_epsilon_current,
        mixture_epsilon_shifted,
        torch.full((len(mixture_z),), mixture_alpha_current, dtype=torch.float64),
        torch.full((len(mixture_z),), mixture_alpha_shifted, dtype=torch.float64),
    )
    mixture_direct = (
        mixture_score_shifted_z - mixture_score_current_z
    ) / mixture_a_current
    mixture_coordinate_error = float((mixture_theta - mixture_direct).abs().max())
    if mixture_coordinate_error > 2e-14:
        raise AssertionError("normalized-heat pullback failed the Gaussian-mixture identity")

    toy_alpha_schedule = np.array([0.95, 0.80, 0.50, 0.20, 0.05], dtype=np.float64)
    mapping = nearest_additive_heat_shift(toy_alpha_schedule, [0, 2, 4], 0.4)
    if np.any(mapping.shifted_timestep < mapping.current_timestep):
        raise AssertionError("additive heat mapping moved toward a cleaner timestep")
    if int(mapping.shifted_timestep[-1]) != 4 or float(mapping.actual_heat_shift[-1]) != 0:
        raise AssertionError("maximum-noise boundary must map to Q=P")

    return {
        "passed": True,
        "batch": batch,
        "dimensions": dimensions,
        "maximum_applied_kl": float(tempered.applied_kl.max()),
        "maximum_density_formula_error": float((from_noise.value - direct).abs().max()),
        "monte_carlo_likelihood_ratio_mean": monte_carlo_mean,
        "normalized_heat_coordinate_error": coordinate_error,
        "gaussian_mixture_coordinate_error": mixture_coordinate_error,
        "naive_same_raw_x_coordinate_error": naive_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run deterministic numerical checks")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("no action selected; pass --self-test")
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
