"""Path-level foresight operators for Internal Guidance.

For a state ``z`` at time ``t``, let ``Phi_s(z)`` be the endpoint produced by
the final (strong) SiT field over a chosen interval and let ``Phi_ig(z)`` be
the endpoint produced by the ordinary Internal-Guidance field over the same
interval.  The path endpoint extrapolation is

    E_rho(z) = Phi_s(z) + rho * (Phi_ig(z) - Phi_s(z)).

The two exact anchors are important: ``rho=0`` is the strong path and
``rho=1`` is ordinary IG.  Values above one extrapolate the *integrated* IG
effect rather than a pointwise strong-minus-internal-head vector.

The module also contains the exact one-Euler-step decomposition used by the
successful foresight material-derivative condition.  For guided velocity
``G`` and weak field ``W``, the forward-guided/weak-inverse displacement is

    H * (G(z, t) - W(z + H*G(z, t), t + H)).

Adding and subtracting ``W(z, t)`` separates the already-tuned local IG term
from the genuinely future-dependent weak-field change.  The experiment keeps
only the latter to avoid silently rescaling ordinary IG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class PathEndpointPair:
    """Endpoints of paired strong and ordinary-IG trajectories."""

    strong: Tensor
    guided: Tensor

    def validate(self) -> None:
        if self.strong.shape != self.guided.shape:
            raise ValueError("strong and guided endpoints must have identical shapes")
        if self.strong.device != self.guided.device:
            raise ValueError("strong and guided endpoints must use the same device")
        if self.strong.dtype != self.guided.dtype:
            raise ValueError("strong and guided endpoints must use the same dtype")

    @property
    def displacement(self) -> Tensor:
        self.validate()
        return self.guided - self.strong


def extrapolate_path_endpoints(pair: PathEndpointPair, *, rho: float) -> Tensor:
    """Return the affine path endpoint at coefficient ``rho``.

    Endpoint branches deliberately avoid arithmetic so the two controls are
    bitwise exact.  Non-negative finite coefficients are accepted; values in
    ``[0, 1]`` interpolate and values above one extrapolate beyond ordinary IG.
    """

    pair.validate()
    rho = float(rho)
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError("rho must be finite and non-negative")
    if rho == 0.0:
        return pair.strong
    if rho == 1.0:
        return pair.guided
    return pair.strong + rho * (pair.guided - pair.strong)


def sample_rms(value: Tensor) -> Tensor:
    """Per-sample root-mean-square over all non-batch dimensions."""

    if value.ndim < 2:
        raise ValueError("sample_rms expects a batch dimension and feature dimensions")
    return value.float().flatten(1).square().mean(dim=1).sqrt()


def mix_characteristic_velocity(
    weak: Tensor,
    guided: Tensor,
    *,
    rho: float,
) -> Tensor:
    """Interpolate the characteristic used by a finite weak-field query.

    ``rho=0`` follows the weak field and ``rho=1`` follows the current guided
    field.  For Internal Guidance ``G=S+gamma*(S-W)``, the strong field lies
    at ``rho=1/(1+gamma)`` on this affine line.  Exact endpoint branches make
    the historical guided-characteristic condition a bitwise anchor.
    """

    if weak.shape != guided.shape:
        raise ValueError("weak and guided fields must have identical shapes")
    if weak.device != guided.device:
        raise ValueError("weak and guided fields must use the same device")
    if weak.dtype != guided.dtype:
        raise ValueError("weak and guided fields must use the same dtype")
    rho = float(rho)
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError("rho must be finite and non-negative")
    if rho == 0.0:
        return weak
    if rho == 1.0:
        return guided
    return weak + rho * (guided - weak)


def split_internal_guidance(
    strong: Tensor,
    weak: Tensor,
    *,
    gamma: float,
) -> tuple[Tensor, Tensor]:
    """Split Internal Guidance into base transport and calibration.

    The identity

    ``S + gamma*(S-W) = W + (1+gamma)*(S-W)``

    is exact.  The first term is the weak/base transport and the second term
    is the displacement contributed by strong-versus-weak calibration.
    """

    if strong.shape != weak.shape:
        raise ValueError("strong and weak fields must have identical shapes")
    if strong.device != weak.device:
        raise ValueError("strong and weak fields must use the same device")
    if strong.dtype != weak.dtype:
        raise ValueError("strong and weak fields must use the same dtype")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    calibration = (1.0 + gamma) * (strong - weak)
    return weak, calibration


def calibration_split_foresight_velocity(
    strong: Tensor,
    weak_now: Tensor,
    weak_query: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """Use a future weak reference with no independently tuned strength.

    Internal Guidance is first written as ``W + beta*(S-W)``, where
    ``beta=1+gamma``.  Replacing only the reference inside this calibration
    contrast by a future weak query gives

    ``W_now + beta*(S-W_query)``.

    Equivalently, this is ordinary IG plus
    ``beta*(W_now-W_query)``.  Hence the correction coefficient is fixed by
    the original guidance scale rather than introduced as a new hyperparameter.
    """

    shapes = {strong.shape, weak_now.shape, weak_query.shape}
    if len(shapes) != 1:
        raise ValueError("strong and weak fields must have identical shapes")
    devices = {strong.device, weak_now.device, weak_query.device}
    if len(devices) != 1:
        raise ValueError("strong and weak fields must use the same device")
    dtypes = {strong.dtype, weak_now.dtype, weak_query.dtype}
    if len(dtypes) != 1:
        raise ValueError("strong and weak fields must use the same dtype")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    beta = 1.0 + gamma
    return weak_now + beta * (strong - weak_query)


def affine_counterfactual_ratio_velocity(
    strong: Tensor,
    weak_base: Tensor,
    weak_references: tuple[Tensor, ...],
    reference_weights: tuple[float, ...],
    *,
    gamma: float,
) -> Tensor:
    """Guide against a geometric ensemble of counterfactual weak references.

    At one fixed linear-flow time, every velocity has the affine form

    ``v_i(z,t) = z/t + (1-t)/t * score_i(z,t)``.

    Consequently, a velocity combination represents the same affine
    combination of scores exactly when its coefficients sum to one.  This
    helper enforces that closure by requiring non-negative reference weights
    that sum to one and returns

    ``W + (1+gamma) * (S - sum_j pi_j Q_j)``.

    If the fields are exact scores of densities, the implied current-time
    density is proportional to

    ``p_W * (p_S / prod_j p_Qj**pi_j)**(1+gamma)``.

    A single unit-weight reference deliberately delegates to the historical
    implementation so the existing PFR condition remains an exact anchor.
    """

    if not weak_references:
        raise ValueError("at least one weak reference is required")
    if len(weak_references) != len(reference_weights):
        raise ValueError("weak references and weights must have equal length")
    tensors = (strong, weak_base, *weak_references)
    if len({value.shape for value in tensors}) != 1:
        raise ValueError("strong, base, and references must have identical shapes")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("strong, base, and references must share a device")
    if len({value.dtype for value in tensors}) != 1:
        raise ValueError("strong, base, and references must share a dtype")
    weights = tuple(float(weight) for weight in reference_weights)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("reference weights must be finite and non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("reference weights must sum to one")
    if len(weak_references) == 1 and weights[0] == 1.0:
        return calibration_split_foresight_velocity(
            strong,
            weak_base,
            weak_references[0],
            gamma=gamma,
        )
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    reference = weak_references[0] * weights[0]
    for value, weight in zip(weak_references[1:], weights[1:], strict=True):
        reference = reference + value * weight
    return weak_base + (1.0 + gamma) * (strong - reference)


def counterfactual_telescoping_velocity(
    current_levels: tuple[Tensor, ...],
    counterfactual_predecessors: tuple[Tensor, ...],
    *,
    gamma: float,
) -> Tensor:
    """Residualize every computational refinement against its predecessor.

    Let ``current_levels=(V_0, ..., V_L)`` be progressively deeper readouts at
    one state and let ``counterfactual_predecessors=(Q_0, ..., Q_{L-1})`` be
    predecessor readouts at one shared counterfactual query. The result is

    ``V_0 + (1+gamma) * sum_k (V_k - Q_{k-1})``.

    If the query collapses to the current state, ``Q_k=V_k`` and the sum
    telescopes exactly to ordinary Internal Guidance from ``V_0`` to ``V_L``.
    The affine coefficients always sum to one. With two levels this is exactly
    the existing single-reference counterfactual calibration rule.
    """

    if len(current_levels) < 2:
        raise ValueError("at least two current hierarchy levels are required")
    if len(counterfactual_predecessors) != len(current_levels) - 1:
        raise ValueError("one counterfactual predecessor is required per transition")
    tensors = (*current_levels, *counterfactual_predecessors)
    if len({value.shape for value in tensors}) != 1:
        raise ValueError("all hierarchy tensors must have identical shapes")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("all hierarchy tensors must share a device")
    if len({value.dtype for value in tensors}) != 1:
        raise ValueError("all hierarchy tensors must share a dtype")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    beta = 1.0 + gamma
    refinements = [
        current_levels[index] - counterfactual_predecessors[index - 1]
        for index in range(1, len(current_levels))
    ]
    return current_levels[0] + beta * torch.stack(refinements).sum(dim=0)


def _broadcast_path_time(time: Tensor | float, value: Tensor, *, name: str) -> Tensor:
    """Return a scalar or per-sample path time broadcast over ``value``."""

    result = torch.as_tensor(time, dtype=value.dtype, device=value.device)
    if result.ndim == 0:
        result = result.reshape(*([1] * value.ndim))
    elif result.ndim == 1 and result.shape[0] == value.shape[0]:
        result = result.reshape(value.shape[0], *([1] * (value.ndim - 1)))
    else:
        raise ValueError(f"{name} must be scalar or have one value per sample")
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def linear_velocity_to_endpoint_score(
    velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Lift a linear-path velocity to the endpoint-normalized Gaussian score.

    For ``Z_t=(1-t)E+tX`` and ``Y=Z_t/t=X+((1-t)/t)E``, the score of the
    density of ``Y`` is

    ``r(Y,t) = t * (t*v(Z_t,t)-Z_t) / (1-t)``.

    This is an affine change of parameterization, not an approximation.
    """

    if velocity.shape != state.shape:
        raise ValueError("velocity and state must have identical shapes")
    if velocity.device != state.device or velocity.dtype != state.dtype:
        raise ValueError("velocity and state must share device and dtype")
    path_time = _broadcast_path_time(time, state, name="time")
    if torch.any(path_time <= 0.0) or torch.any(path_time >= 1.0):
        raise ValueError("endpoint-score conversion requires time in (0, 1)")
    return path_time * (path_time * velocity - state) / (1.0 - path_time)


def endpoint_score_to_linear_velocity(
    score: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Invert :func:`linear_velocity_to_endpoint_score` exactly."""

    if score.shape != state.shape:
        raise ValueError("score and state must have identical shapes")
    if score.device != state.device or score.dtype != state.dtype:
        raise ValueError("score and state must share device and dtype")
    path_time = _broadcast_path_time(time, state, name="time")
    if torch.any(path_time <= 0.0) or torch.any(path_time >= 1.0):
        raise ValueError("endpoint-score conversion requires time in (0, 1)")
    return state / path_time + (1.0 - path_time) * score / path_time.square()


def linear_velocity_to_marginal_score(
    velocity: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Convert velocity to the score of the unscaled path marginal ``p_t(z)``."""

    if velocity.shape != state.shape:
        raise ValueError("velocity and state must have identical shapes")
    if velocity.device != state.device or velocity.dtype != state.dtype:
        raise ValueError("velocity and state must share device and dtype")
    path_time = _broadcast_path_time(time, state, name="time")
    if torch.any(path_time < 0.0) or torch.any(path_time >= 1.0):
        raise ValueError("marginal-score conversion requires time in [0, 1)")
    return (path_time * velocity - state) / (1.0 - path_time)


def marginal_score_to_linear_velocity(
    score: Tensor,
    state: Tensor,
    time: Tensor | float,
) -> Tensor:
    """Convert a path-marginal score back to current-time linear velocity."""

    if score.shape != state.shape:
        raise ValueError("score and state must have identical shapes")
    if score.device != state.device or score.dtype != state.dtype:
        raise ValueError("score and state must share device and dtype")
    path_time = _broadcast_path_time(time, state, name="time")
    if torch.any(path_time <= 0.0) or torch.any(path_time >= 1.0):
        raise ValueError("marginal-score inversion requires time in (0, 1)")
    return (state + (1.0 - path_time) * score) / path_time


def factorized_scale_space_guidance_velocity(
    strong_now: Tensor,
    weak_now: Tensor,
    temporal_now: Tensor,
    temporal_reference: Tensor,
    state_now: Tensor,
    state_reference: Tensor,
    time_now: Tensor | float,
    time_reference: Tensor | float,
    *,
    gamma: float,
    temporal_weight: Tensor | float,
) -> Tensor:
    """Compose depth and temporal density ratios in marginal-score space.

    The ordinary Internal-Guidance score is augmented by

    ``gamma * temporal_weight * (score_temporal_now-score_temporal_ref)``.

    When the temporal branch is the weak head and ``temporal_weight=1``, the
    intermediate weak/current density cancels exactly, leaving a telescoping
    strong/current versus weak/reference density ratio.
    """

    tensors = (
        strong_now,
        weak_now,
        temporal_now,
        temporal_reference,
        state_now,
        state_reference,
    )
    if len({value.shape for value in tensors}) != 1:
        raise ValueError("all velocities and states must have identical shapes")
    if len({value.device for value in tensors}) != 1 or len(
        {value.dtype for value in tensors}
    ) != 1:
        raise ValueError("all velocities and states must share device and dtype")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    weight = _broadcast_path_time(
        temporal_weight, state_now, name="temporal_weight"
    )
    if torch.any(weight < 0.0) or torch.any(weight > 1.0):
        raise ValueError("temporal_weight must lie in [0, 1]")
    if gamma == 0.0:
        return strong_now
    ordinary = strong_now + gamma * (strong_now - weak_now)
    if torch.count_nonzero(weight).item() == 0:
        return ordinary
    strong_score = linear_velocity_to_marginal_score(
        strong_now, state_now, time_now
    )
    weak_score = linear_velocity_to_marginal_score(
        weak_now, state_now, time_now
    )
    temporal_score = linear_velocity_to_marginal_score(
        temporal_now, state_now, time_now
    )
    reference_score = linear_velocity_to_marginal_score(
        temporal_reference, state_reference, time_reference
    )
    ordinary_score = strong_score + gamma * (strong_score - weak_score)
    guided_score = ordinary_score + gamma * weight * (
        temporal_score - reference_score
    )
    return marginal_score_to_linear_velocity(guided_score, state_now, time_now)


@dataclass(frozen=True)
class CrossTimeVelocityChange:
    """Exact split of a velocity difference across two path times."""

    parameterization_transport: Tensor
    score_evolution: Tensor

    @property
    def total(self) -> Tensor:
        return self.parameterization_transport + self.score_evolution


def decompose_cross_time_velocity_change(
    velocity_now: Tensor,
    velocity_reference: Tensor,
    state: Tensor,
    time_now: Tensor | float,
    time_reference: Tensor | float,
) -> CrossTimeVelocityChange:
    """Split ``v_now-v_ref`` into representation and score changes.

    Let ``V_t(s)=z/t+(1-t)s/t`` map a marginal score to linear-path
    velocity.  Inserting ``V_ref(s_now)`` gives the exact identity

    ``v_now-v_ref = [v_now-V_ref(s_now)] + [V_ref(s_now)-v_ref]``.

    The first bracket changes only the velocity parameterization while
    holding the current score fixed.  The second changes only the score while
    holding the reference-time velocity parameterization fixed.
    """

    if velocity_now.shape != velocity_reference.shape or velocity_now.shape != state.shape:
        raise ValueError("velocities and state must have identical shapes")
    if len({velocity_now.device, velocity_reference.device, state.device}) != 1:
        raise ValueError("velocities and state must share a device")
    if len({velocity_now.dtype, velocity_reference.dtype, state.dtype}) != 1:
        raise ValueError("velocities and state must share a dtype")
    score_now = linear_velocity_to_marginal_score(
        velocity_now, state, time_now
    )
    transported_now = marginal_score_to_linear_velocity(
        score_now, state, time_reference
    )
    return CrossTimeVelocityChange(
        parameterization_transport=velocity_now - transported_now,
        score_evolution=transported_now - velocity_reference,
    )


def align_linear_path_state_to_endpoint_coordinate(
    state: Tensor,
    time: Tensor | float,
    reference_time: Tensor | float,
) -> Tensor:
    """Move a state across path times while holding ``Y=Z_t/t`` fixed."""

    current = _broadcast_path_time(time, state, name="time")
    reference = _broadcast_path_time(
        reference_time, state, name="reference_time"
    )
    if torch.any(current <= 0.0) or torch.any(current >= 1.0):
        raise ValueError("state alignment requires time in (0, 1)")
    if torch.any(reference <= 0.0) or torch.any(reference >= 1.0):
        raise ValueError("state alignment requires reference_time in (0, 1)")
    return state * (reference / current)


def telescoping_scale_space_guidance_velocity(
    strong_now: Tensor,
    weak_reference: Tensor,
    state_now: Tensor,
    state_reference: Tensor,
    time_now: Tensor | float,
    time_reference: Tensor | float,
    *,
    gamma: float,
) -> Tensor:
    """Apply a strong-current versus weak-reference score-density ratio.

    Both velocities are first represented as scores of the normalized
    Gaussian channel ``Y=Z_t/t``.  The guided score is

    ``r_S + gamma * (r_S-r_W_ref)``,

    which is the score of ``q_S * (q_S/q_W_ref)**gamma`` whenever the two
    fields are population scores.  The result is converted back to a velocity
    at the current state and time.
    """

    shapes = {strong_now.shape, weak_reference.shape, state_now.shape,
              state_reference.shape}
    if len(shapes) != 1:
        raise ValueError("all velocities and states must have identical shapes")
    devices = {strong_now.device, weak_reference.device, state_now.device,
               state_reference.device}
    dtypes = {strong_now.dtype, weak_reference.dtype, state_now.dtype,
              state_reference.dtype}
    if len(devices) != 1 or len(dtypes) != 1:
        raise ValueError("all velocities and states must share device and dtype")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if gamma == 0.0:
        return strong_now
    strong_score = linear_velocity_to_endpoint_score(
        strong_now, state_now, time_now
    )
    weak_score = linear_velocity_to_endpoint_score(
        weak_reference, state_reference, time_reference
    )
    guided_score = strong_score + gamma * (strong_score - weak_score)
    return endpoint_score_to_linear_velocity(guided_score, state_now, time_now)


def transported_internal_gap_velocity(
    strong_now: Tensor,
    weak_now: Tensor,
    strong_query: Tensor,
    weak_query: Tensor,
    *,
    gamma: float,
    anchor: str = "weak",
    interaction_sign: float = 1.0,
) -> Tensor:
    """Transport the aligned strong-minus-weak gap across a query.

    The four predictions form a depth-by-query factorial table.  Its unique
    common-mode-invariant interaction is

    ``omega = (strong_query-weak_query) - (strong_now-weak_now)``.

    With the hierarchical weak anchor, adding ``beta*omega`` to ordinary IG
    is exactly ``weak_now + beta*(strong_query-weak_query)``.  The strong
    anchor uses the original ``gamma`` coefficient instead.  A negative
    ``interaction_sign`` is retained solely as a causal sign control.
    """

    shapes = {
        strong_now.shape,
        weak_now.shape,
        strong_query.shape,
        weak_query.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all strong and weak fields must have identical shapes")
    devices = {
        strong_now.device,
        weak_now.device,
        strong_query.device,
        weak_query.device,
    }
    if len(devices) != 1:
        raise ValueError("all strong and weak fields must use the same device")
    dtypes = {
        strong_now.dtype,
        weak_now.dtype,
        strong_query.dtype,
        weak_query.dtype,
    }
    if len(dtypes) != 1:
        raise ValueError("all strong and weak fields must use the same dtype")
    gamma = float(gamma)
    interaction_sign = float(interaction_sign)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if not math.isfinite(interaction_sign):
        raise ValueError("interaction_sign must be finite")
    if anchor not in {"weak", "strong"}:
        raise ValueError("anchor must be weak or strong")

    gap_now = strong_now - weak_now
    gap_query = strong_query - weak_query
    interaction = gap_query - gap_now
    if anchor == "weak":
        guided = weak_now + (1.0 + gamma) * gap_now
        return guided + interaction_sign * (1.0 + gamma) * interaction
    guided = strong_now + gamma * gap_now
    return guided + interaction_sign * gamma * interaction


def forecast_weak_reference(
    weak_now: Tensor,
    weak_query: Tensor,
    *,
    factor: float,
) -> Tensor:
    """Forecast a weak reference along a finite query displacement.

    ``factor=0`` keeps the current weak head, ``factor=1`` replaces it with
    the queried weak head, and ``factor=2`` linearly extrapolates one equal
    displacement beyond the query. Endpoint branches are bitwise exact.
    """

    if weak_now.shape != weak_query.shape:
        raise ValueError("weak references must have identical shapes")
    if weak_now.device != weak_query.device:
        raise ValueError("weak references must use the same device")
    if weak_now.dtype != weak_query.dtype:
        raise ValueError("weak references must use the same dtype")
    factor = float(factor)
    if not math.isfinite(factor) or factor < 0.0:
        raise ValueError("factor must be finite and non-negative")
    if factor == 0.0:
        return weak_now
    if factor == 1.0:
        return weak_query
    return weak_now + factor * (weak_query - weak_now)


@dataclass(frozen=True)
class SampleProjection:
    """Per-sample Euclidean projection onto a reference tensor."""

    parallel: Tensor
    orthogonal: Tensor
    coefficient: Tensor


@dataclass(frozen=True)
class EulerForesightDecomposition:
    """Local and genuinely future-dependent parts of one Euler round trip."""

    local_displacement: Tensor
    future_displacement: Tensor
    roundtrip_displacement: Tensor


@dataclass(frozen=True)
class FutureDriftDecomposition:
    """Exact split of the successful future weak-field correction.

    Let ``g = S - W``.  The correction used by FMD-IG obeys

    ``W_now - W_future = (g_future - g_now) + (S_now - S_future)``.

    The first term is the evolution of the Internal-Guidance discrepancy.  The
    second is the negative finite change of the strong field.  Keeping both
    terms explicit prevents an empirical gain from being attributed to the
    weak field when it could instead come from strong-flow curvature.
    """

    gap_now: Tensor
    gap_future: Tensor
    gap_change: Tensor
    strong_curvature_correction: Tensor
    weak_drift_correction: Tensor


@dataclass(frozen=True)
class MaterialChangeDecomposition:
    """Exact time/state split of one finite weak-field change.

    For ``W_0=W(z,t)``, ``W_t=W(z,t+h)`` and
    ``W_+=W(z+hG,t+h)``,

    ``W_0-W_+ = (W_0-W_t) + (W_t-W_+)``.

    The two terms approach ``-h*partial_t W`` and ``-h*J_W G`` as the
    horizon vanishes.  The finite identity is exact and does not rely on that
    local expansion.
    """

    temporal: Tensor
    advective: Tensor
    combined: Tensor


@dataclass(frozen=True)
class EndpointPosteriorChange:
    """Exact endpoint-posterior split of a finite velocity change.

    For the linear bridge ``z_t=t*x+(1-t)*eps``, any velocity estimate ``w``
    implies endpoint estimates ``x_hat=z+(1-t)w`` and ``eps_hat=z-tw``.
    Between any two queried space-time points, their changes obey

    ``delta_x - delta_eps = w_now - w_query``.

    The subtraction uniquely cancels the common coordinate displacement of
    the two endpoint estimates, up to an overall scalar multiplier.
    """

    clean: Tensor
    noise: Tensor
    negative_noise: Tensor
    velocity: Tensor


def decompose_endpoint_posterior_change(
    state_now: Tensor,
    time_now: float | Tensor,
    velocity_now: Tensor,
    state_query: Tensor,
    time_query: float | Tensor,
    velocity_query: Tensor,
) -> EndpointPosteriorChange:
    """Split a finite velocity change into clean/noise posterior changes."""

    shapes = {
        state_now.shape,
        velocity_now.shape,
        state_query.shape,
        velocity_query.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all states and velocities must have identical shapes")
    devices = {
        state_now.device,
        velocity_now.device,
        state_query.device,
        velocity_query.device,
    }
    if len(devices) != 1:
        raise ValueError("all states and velocities must use the same device")
    dtypes = {
        state_now.dtype,
        velocity_now.dtype,
        state_query.dtype,
        velocity_query.dtype,
    }
    if len(dtypes) != 1:
        raise ValueError("all states and velocities must use the same dtype")

    clean_now = state_now + (1.0 - time_now) * velocity_now
    clean_query = state_query + (1.0 - time_query) * velocity_query
    noise_now = state_now - time_now * velocity_now
    noise_query = state_query - time_query * velocity_query
    clean = clean_now - clean_query
    noise = noise_now - noise_query
    negative_noise = -noise
    velocity = velocity_now - velocity_query
    return EndpointPosteriorChange(
        clean=clean,
        noise=noise,
        negative_noise=negative_noise,
        velocity=velocity,
    )


def decompose_future_weak_drift(
    strong_now: Tensor,
    weak_now: Tensor,
    strong_future: Tensor,
    weak_future: Tensor,
) -> FutureDriftDecomposition:
    """Return the exact gap/strong split of ``weak_now - weak_future``."""

    shapes = {
        strong_now.shape,
        weak_now.shape,
        strong_future.shape,
        weak_future.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all current/future fields must have identical shapes")
    devices = {
        strong_now.device,
        weak_now.device,
        strong_future.device,
        weak_future.device,
    }
    if len(devices) != 1:
        raise ValueError("all current/future fields must use the same device")
    dtypes = {
        strong_now.dtype,
        weak_now.dtype,
        strong_future.dtype,
        weak_future.dtype,
    }
    if len(dtypes) != 1:
        raise ValueError("all current/future fields must use the same dtype")

    gap_now = strong_now - weak_now
    gap_future = strong_future - weak_future
    gap_change = gap_future - gap_now
    strong_curvature_correction = strong_now - strong_future
    weak_drift_correction = weak_now - weak_future
    return FutureDriftDecomposition(
        gap_now=gap_now,
        gap_future=gap_future,
        gap_change=gap_change,
        strong_curvature_correction=strong_curvature_correction,
        weak_drift_correction=weak_drift_correction,
    )


def decompose_material_change(
    weak_now: Tensor,
    weak_future_same_state: Tensor,
    weak_future_along_path: Tensor,
) -> MaterialChangeDecomposition:
    """Split ``W(z,t)-W(z+hG,t+h)`` into time and state changes exactly."""

    shapes = {
        weak_now.shape,
        weak_future_same_state.shape,
        weak_future_along_path.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all weak fields must have identical shapes")
    devices = {
        weak_now.device,
        weak_future_same_state.device,
        weak_future_along_path.device,
    }
    if len(devices) != 1:
        raise ValueError("all weak fields must use the same device")
    dtypes = {
        weak_now.dtype,
        weak_future_same_state.dtype,
        weak_future_along_path.dtype,
    }
    if len(dtypes) != 1:
        raise ValueError("all weak fields must use the same dtype")

    temporal = weak_now - weak_future_same_state
    advective = weak_future_same_state - weak_future_along_path
    return MaterialChangeDecomposition(
        temporal=temporal,
        advective=advective,
        combined=weak_now - weak_future_along_path,
    )


def finite_lie_bracket_change(
    weak_now: Tensor,
    weak_future_along_guided: Tensor,
    guided_now: Tensor,
    guided_future_along_weak: Tensor,
) -> Tensor:
    """Return a forward finite-difference estimate of ``-h*[G,W]``.

    Time is treated as an augmented coordinate.  With

    ``D_G W = partial_t W + J_W G`` and
    ``D_W G = partial_t G + J_G W``, the returned quantity is

    ``(W-W_G^+) - (G-G_W^+) = -h*(D_G W-D_W G) + O(h^2)``.

    Unlike a material derivative, this quantity vanishes exactly whenever
    the two queried fields coincide.
    """

    shapes = {
        weak_now.shape,
        weak_future_along_guided.shape,
        guided_now.shape,
        guided_future_along_weak.shape,
    }
    if len(shapes) != 1:
        raise ValueError("all bracket fields must have identical shapes")
    devices = {
        weak_now.device,
        weak_future_along_guided.device,
        guided_now.device,
        guided_future_along_weak.device,
    }
    if len(devices) != 1:
        raise ValueError("all bracket fields must use the same device")
    dtypes = {
        weak_now.dtype,
        weak_future_along_guided.dtype,
        guided_now.dtype,
        guided_future_along_weak.dtype,
    }
    if len(dtypes) != 1:
        raise ValueError("all bracket fields must use the same dtype")
    return (weak_now - weak_future_along_guided) - (
        guided_now - guided_future_along_weak
    )


def foresight_weak_guidance(
    strong_now: Tensor,
    weak_now: Tensor,
    weak_future: Tensor,
    *,
    gamma: float,
    alpha: float,
) -> Tensor:
    """Use a current/future mixture as the weak reference in IG.

    ``alpha=0`` is ordinary Internal Guidance and ``alpha=1`` replaces the
    instantaneous weak field by its value at a predicted future state:

    ``S + gamma * (S - ((1-alpha)*W_now + alpha*W_future))``.
    """

    if strong_now.shape != weak_now.shape or weak_now.shape != weak_future.shape:
        raise ValueError("strong/current/future weak fields must have identical shapes")
    gamma = float(gamma)
    alpha = float(alpha)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and non-negative")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if gamma == 0.0:
        return strong_now
    if alpha == 0.0:
        return strong_now + gamma * (strong_now - weak_now)
    if alpha == 1.0:
        return strong_now + gamma * (strong_now - weak_future)
    weak_reference = (1.0 - alpha) * weak_now + alpha * weak_future
    return strong_now + gamma * (strong_now - weak_reference)


def relax_future_weak_reference(
    weak_reference: Tensor,
    weak_future: Tensor,
    *,
    gamma: float,
    eta: float,
) -> Tensor:
    """Perform one relaxed Picard update of the IG weak reference.

    If ``V = S + gamma * (S - weak_reference)`` and
    ``T(V) = S + gamma * (S - weak_future)``, then

    ``(1-rho) * V + rho * T(V)``

    is represented by the returned weak reference with ``rho = eta/gamma``.
    Keeping ``eta`` as the public coefficient exactly matches the historical
    FMD update on the first iteration.
    """

    if weak_reference.shape != weak_future.shape:
        raise ValueError("current and future weak references must have identical shapes")
    gamma = float(gamma)
    eta = float(eta)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    if not math.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    if eta == 0.0:
        return weak_reference
    if eta == gamma:
        return weak_future
    alpha = eta / gamma
    return weak_reference + alpha * (weak_future - weak_reference)


def richardson_forward_change(
    weak_now: Tensor,
    weak_half: Tensor,
    weak_future: Tensor,
) -> Tensor:
    """Estimate minus one horizon of material change with second-order accuracy.

    Along a smooth predicted characteristic, let ``W_s`` denote the weak field
    at offsets ``s=0, h/2, h``.  Then

    ``4 * (W_0 - W_{h/2}) - (W_0 - W_h) = -h D W + O(h^3)``.

    The historical FMD difference ``W_0-W_h`` has an ``O(h^2)`` truncation
    term.  This Richardson combination removes that term without requiring a
    backward-time query.
    """

    if weak_now.shape != weak_half.shape or weak_half.shape != weak_future.shape:
        raise ValueError("all Richardson weak fields must have identical shapes")
    if weak_now.device != weak_half.device or weak_half.device != weak_future.device:
        raise ValueError("all Richardson weak fields must use the same device")
    if weak_now.dtype != weak_half.dtype or weak_half.dtype != weak_future.dtype:
        raise ValueError("all Richardson weak fields must use the same dtype")
    return 4.0 * (weak_now - weak_half) - (weak_now - weak_future)


def mix_material_curvature(
    first_material_change: Tensor,
    finite_change: Tensor,
    *,
    curvature_weight: float,
) -> Tensor:
    """Mix the first material term with finite-horizon curvature.

    ``first_material_change`` is the Richardson estimate and ``finite_change``
    is the historical ``W_0 - W_h`` correction. Their difference starts at
    second order in the horizon. ``curvature_weight=0`` keeps only the first
    material term, while ``curvature_weight=1`` exactly recovers historical
    FMD.
    """

    if first_material_change.shape != finite_change.shape:
        raise ValueError("material changes must have identical shapes")
    if first_material_change.device != finite_change.device:
        raise ValueError("material changes must use the same device")
    if first_material_change.dtype != finite_change.dtype:
        raise ValueError("material changes must use the same dtype")
    curvature_weight = float(curvature_weight)
    if not math.isfinite(curvature_weight) or curvature_weight < 0.0:
        raise ValueError("curvature_weight must be finite and non-negative")
    if curvature_weight == 0.0:
        return first_material_change
    if curvature_weight == 1.0:
        return finite_change
    return first_material_change + curvature_weight * (
        finite_change - first_material_change
    )


def decompose_euler_foresight_roundtrip(
    guided_now: Tensor,
    weak_now: Tensor,
    weak_future: Tensor,
    *,
    horizon: float,
) -> EulerForesightDecomposition:
    """Decompose ``Euler_W^-1(Euler_G(z)) - z`` exactly.

    The Euler round-trip displacement is

    ``H * (G(z,t) - W(z + H*G(z,t), t+H))``.

    Adding and subtracting ``W(z,t)`` separates the ordinary local guidance
    contrast from the only term that queries the predicted future state.
    """

    if guided_now.shape != weak_now.shape or weak_now.shape != weak_future.shape:
        raise ValueError("all foresight velocities must have identical shapes")
    horizon = float(horizon)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and positive")
    local = horizon * (guided_now - weak_now)
    future = horizon * (weak_now - weak_future)
    return EulerForesightDecomposition(
        local_displacement=local,
        future_displacement=future,
        roundtrip_displacement=local + future,
    )


def project_per_sample(value: Tensor, reference: Tensor) -> SampleProjection:
    """Split ``value`` into parts parallel and orthogonal to ``reference``."""

    if value.shape != reference.shape:
        raise ValueError("value and reference must have identical shapes")
    if value.ndim < 2:
        raise ValueError("projection expects a batch and feature dimensions")
    value_flat = value.float().flatten(1)
    reference_flat = reference.float().flatten(1)
    numerator = (value_flat * reference_flat).sum(dim=1)
    denominator = reference_flat.square().sum(dim=1)
    tiny = torch.finfo(value_flat.dtype).tiny
    coefficient_flat = torch.where(
        denominator > tiny,
        numerator / denominator.clamp_min(tiny),
        torch.zeros_like(numerator),
    )
    coefficient = coefficient_flat.reshape(
        len(value), *([1] * (value.ndim - 1))
    ).to(dtype=value.dtype)
    parallel = coefficient * reference
    return SampleProjection(
        parallel=parallel,
        orthogonal=value - parallel,
        coefficient=coefficient_flat,
    )


def project_to_forward_ray(value: Tensor, direction: Tensor) -> SampleProjection:
    """Project each sample onto the non-negative ray spanned by ``direction``.

    This is the closed-form solution of

    ``argmin_{a >= 0} ||a*direction-value||^2``.

    Unlike an unconstrained line projection, the result cannot move a future
    query backward along the current characteristic.
    """

    line_projection = project_per_sample(value, direction)
    coefficient_flat = line_projection.coefficient.clamp_min(0.0)
    coefficient = coefficient_flat.reshape(
        len(value), *([1] * (value.ndim - 1))
    ).to(dtype=value.dtype)
    parallel = coefficient * direction
    return SampleProjection(
        parallel=parallel,
        orthogonal=value - parallel,
        coefficient=coefficient_flat,
    )


def match_sample_rms(value: Tensor, reference: Tensor) -> Tensor:
    """Rescale each sample in ``value`` to the RMS of ``reference``."""

    if value.shape != reference.shape:
        raise ValueError("value and reference must have identical shapes")
    value_rms = sample_rms(value)
    reference_rms = sample_rms(reference)
    tiny = torch.finfo(value_rms.dtype).tiny
    scale = torch.where(
        value_rms > tiny,
        reference_rms / value_rms.clamp_min(tiny),
        torch.zeros_like(value_rms),
    )
    return value * scale.reshape(len(value), *([1] * (value.ndim - 1))).to(
        dtype=value.dtype
    )
