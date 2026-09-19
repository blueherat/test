"""Reference estimators for AG error cancellation, without model or sampler calls.

These are algebraic research tools, not a validated image-quality method.
Inputs must be predictions for the SAME state, noise level, condition, and
prediction parameterization. ``gamma=0`` returns the strong prediction;
the NVlabs EDM2 ``guidance`` parameter is ``1 + gamma``.
"""

from __future__ import annotations

import torch


def _predictions(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if not values or values[0].ndim < 2:
        raise ValueError("Expected tensors with batch and feature dimensions")
    first = values[0]
    if any(v.shape != first.shape or v.device != first.device for v in values):
        raise ValueError("Prediction shapes and devices must agree")
    if any(not v.is_floating_point() for v in values):
        raise ValueError("Predictions must be floating point")
    return tuple(v.to(torch.float64).flatten(1) for v in values)


@torch.no_grad()
def calibrate_risk_gamma(
    clean: torch.Tensor, strong: torch.Tensor, weak: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fit one coefficient to a calibration batch at one fixed noise level.

    All moments are means over samples AND features. Clean/noisy pairs must
    come from a declared calibration distribution. This function cannot infer
    the true target from generated states alone. It returns both unconstrained
    and nonnegative coefficients; no image-quality optimum is implied.
    """
    x, ds, dw = _predictions(clean, strong, weak)
    d = ds - dw
    residual = x - ds
    a = (residual * d).mean()
    b = d.square().mean()
    identified = b > 0
    denominator = torch.where(identified, b, torch.ones_like(b))
    gamma = torch.where(identified, a / denominator, torch.zeros_like(a))
    positive = gamma.clamp_min(0)
    ls = residual.square().mean()
    lw = (x - dw).square().mean()
    return {
        "gamma": gamma,
        "gamma_nonnegative": positive,
        "identified": identified,
        "A": a,
        "B": b,
        "strong_mse": ls,
        "weak_mse": lw,
        "fitted_mse": (residual - gamma * d).square().mean(),
        "fitted_nonnegative_mse": (residual - positive * d).square().mean(),
        "orthogonality": ((residual - gamma * d) * d).mean(),
    }


@torch.no_grad()
def hierarchy_gamma(
    strong: torch.Tensor,
    weak: torch.Tensor,
    weaker: torch.Tensor,
    *,
    ridge_relative: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Solve D0 + gamma*(D0-D1) ~= D1 + gamma*(D1-D2), per sample.

    The returned coefficient is unconstrained. ``identified`` only detects
    a numerically usable denominator; it is NOT evidence that the estimated
    shared limit is correct. ``residual_relative`` measures fit to the equality,
    not quality. A positive ridge minimizes ||d0-gamma*h||^2 +
    ridge_relative*||d0||^2*gamma^2. It changes the unregularized estimator.
    """
    if ridge_relative < 0:
        raise ValueError("ridge_relative must be nonnegative")
    d0_model, d1_model, d2_model = _predictions(strong, weak, weaker)
    d0 = d0_model - d1_model
    d1 = d1_model - d2_model
    h = d1 - d0
    norm0 = d0.square().sum(1)
    norm1 = d1.square().sum(1)
    normh = h.square().sum(1)
    dot01 = (d0 * d1).sum(1)
    scale = torch.maximum(norm0, norm1)
    tiny = torch.finfo(torch.float64).tiny
    threshold = 64 * torch.finfo(torch.float64).eps * scale
    identified = (normh > threshold) & (norm0 > 0)
    denominator = normh + ridge_relative * norm0
    safe_denominator = torch.where(identified, denominator, torch.ones_like(normh))
    gamma = torch.where(
        identified, (d0 * h).sum(1) / safe_denominator, torch.zeros_like(normh),
    )
    relative = (d0 - gamma[:, None] * h).norm(dim=1) / norm0.sqrt().clamp_min(tiny)
    return {
        "gamma": gamma,
        "identified": identified,
        "residual_relative": relative,
        "kappa_ls": dot01 / norm0.clamp_min(tiny),
        "cosine": dot01 / (norm0.sqrt() * norm1.sqrt()).clamp_min(tiny),
        "gap_rms": (norm0 / d0.shape[1]).sqrt(),
        "second_difference_rms": (normh / d0.shape[1]).sqrt(),
    }


def apply_ag(
    strong: torch.Tensor, weak: torch.Tensor, gamma: torch.Tensor | float,
) -> torch.Tensor:
    """Apply a supplied scalar or per-sample coefficient in prediction units."""
    if strong.shape != weak.shape or strong.device != weak.device:
        raise ValueError("Prediction shapes and devices must agree")
    coefficient = torch.as_tensor(gamma, device=strong.device, dtype=strong.dtype)
    if coefficient.ndim == 1:
        if coefficient.shape[0] != strong.shape[0]:
            raise ValueError("Per-sample gamma must have batch length")
        coefficient = coefficient.reshape((-1,) + (1,) * (strong.ndim - 1))
    elif coefficient.ndim != 0:
        raise ValueError("gamma must be scalar or have shape (batch,)")
    return strong + coefficient * (strong - weak)
