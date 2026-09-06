"""Bounded, fully reorthogonalized Krylov diagnostics for a symmetric operator.

The caller supplies the symmetric *linear* matvec, e.g. (J v + J.T v)/2.
Symmetrizing the small Ritz matrix does not symmetrize a nonsymmetric matvec.
The largest Ritz value describes the explored subspace. In particular, a
negative value, even with a small residual, does not certify a negative
semidefinite operator: the start may miss a positive eigenspace entirely.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor


def _ritz(basis: Tensor, abasis: Tensor) -> tuple[Tensor, dict, Tensor]:
    """Use cached operator images only; basis vectors are stored as rows."""
    raw = basis @ abasis.T
    projected = (raw + raw.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(projected)
    coefficients = eigenvectors[:, -1]
    vector = coefficients @ basis
    avector = coefficients @ abasis
    length = torch.linalg.vector_norm(vector)
    vector, avector = vector / length, avector / length
    rayleigh = torch.dot(vector, avector)
    residual = torch.linalg.vector_norm(avector - rayleigh * vector)
    avector_norm = torch.linalg.vector_norm(avector)
    denominator = torch.maximum(avector_norm, rayleigh.abs()).clamp_min(torch.finfo(torch.float64).tiny)
    orthogonality = basis @ basis.T - torch.eye(len(basis), device=basis.device, dtype=torch.float64)
    statistics = {
        "iterations": len(basis),
        "ritz_value": float(eigenvalues[-1]),
        "rayleigh": float(rayleigh),
        "residual_norm": float(residual),
        "relative_residual": float(residual / denominator),
        "projected_asymmetry_norm": float(torch.linalg.matrix_norm(raw - raw.T)),
        "orthogonality_error": float(torch.linalg.matrix_norm(orthogonality)),
    }
    if not all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) for value in statistics.values()):
        raise FloatingPointError("nonfinite Ritz diagnostic")
    return vector, statistics, projected


def symmetric_krylov(
    matvec: Callable[[Tensor], Tensor],
    start: Tensor,
    max_iterations: int = 24,
    checkpoints: Iterable[int] = (8, 16, 24),
) -> dict:
    """Return the top algebraic Ritz pair after at most ``max_iterations`` calls.

    This is symmetric Arnoldi/Lanczos with two full orthogonalization passes,
    retaining every basis vector and its actual matvec image. All basis,
    orthogonalization, and small-matrix operations use FP64 on start.device.
    The callback receives FP64 vectors with start.shape, may cast internally
    to its model precision, and must return a real tensor on the same device
    with the same number of elements. Its output is detached immediately;
    matvec itself is not wrapped in no_grad, so it may compute JVPs/VJPs.

    ``vector`` has start.shape and FP64 dtype. ``basis`` and ``abasis`` are
    [iterations, start.numel()] row matrices. ``history`` contains the requested
    reachable checkpoints and the final iteration, without extra matvec calls.
    Rayleigh and residual use the cached combination A_basis.T @ coefficients;
    an independent matvec of the final vector is the caller's optional check,
    especially when the callback uses finite-precision model arithmetic.

    No residual-based adaptive stopping is used. Early termination occurs only
    at ambient dimension or numerical invariant-subspace breakdown. The latter
    uses eps64 * dimension * ||Aq||, a roundoff scale rather than a spectral or
    quality threshold. A missed eigenspace is never replaced with random noise.
    """
    if not callable(matvec):
        raise TypeError("matvec must be callable")
    if not isinstance(start, Tensor) or not start.is_floating_point():
        raise TypeError("start must be a real floating-point tensor")
    if start.numel() == 0 or not bool(torch.isfinite(start).all()):
        raise ValueError("start must be nonempty and finite")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    requested = tuple(checkpoints)
    if any(type(k) is not int or k < 1 for k in requested):
        raise ValueError("checkpoints must contain positive integers")
    requested = set(requested)
    initial = start.detach().reshape(-1).to(dtype=torch.float64)
    initial_norm = torch.linalg.vector_norm(initial)
    if not bool(torch.isfinite(initial_norm)) or initial_norm == 0:
        raise ValueError("start must have a finite, nonzero FP64 norm")
    dimension = initial.numel()
    capacity = min(max_iterations, dimension)
    basis = torch.empty((capacity, dimension), device=start.device, dtype=torch.float64)
    abasis = torch.empty_like(basis)
    basis[0] = initial / initial_norm
    history = []
    breakdown_norm = breakdown_threshold = None
    reason = "max_iterations"
    vector = statistics = projected = None
    for index in range(capacity):
        # A callback must not mutate a retained basis vector in place.
        image = matvec(basis[index].reshape(start.shape).clone())
        if not isinstance(image, Tensor) or not image.is_floating_point():
            raise TypeError("matvec must return a real floating-point tensor")
        if image.numel() != dimension or image.device != start.device:
            raise ValueError("matvec output must preserve dimension and device")
        image = image.detach().reshape(-1).to(dtype=torch.float64)
        if not bool(torch.isfinite(image).all()):
            raise FloatingPointError("matvec produced nonfinite values")
        abasis[index] = image
        count = index + 1
        current_basis, current_abasis = basis[:count], abasis[:count]
        final = count == capacity
        if final:
            reason = "dimension_exhausted" if count == dimension else "max_iterations"
        else:
            remainder = image.clone()
            for _ in range(2):
                remainder -= (current_basis @ remainder) @ current_basis
            remainder_norm = torch.linalg.vector_norm(remainder)
            image_norm = torch.linalg.vector_norm(image)
            if not bool(torch.isfinite(remainder_norm) & torch.isfinite(image_norm)):
                raise FloatingPointError("nonfinite Krylov norm")
            threshold = torch.finfo(torch.float64).eps * dimension * image_norm
            if remainder_norm <= threshold:
                final, reason = True, "numerical_breakdown"
                breakdown_norm, breakdown_threshold = float(remainder_norm), float(threshold)
            else:
                basis[count] = remainder / remainder_norm
        if count in requested or final:
            vector, statistics, projected = _ritz(current_basis, current_abasis)
            if history:
                statistics["rayleigh_change_from_previous"] = statistics["rayleigh"] - history[-1]["rayleigh"]
            else:
                statistics["rayleigh_change_from_previous"] = None
            history.append(statistics.copy())
        if final:
            break
    return {
        **statistics,
        "vector": vector.reshape(start.shape),
        "basis": basis[:count].clone(),
        "abasis": abasis[:count].clone(),
        "projected_matrix": projected,
        "history": history,
        "matvec_calls": count,
        "termination": reason,
        "breakdown_norm": breakdown_norm,
        "breakdown_threshold": breakdown_threshold,
        "residual_source": "linear combination of cached operator images; no extra matvec",
        "spectral_scope": "explored Krylov subspace only; a negative top Ritz value does not certify a globally nonpositive spectrum",
    }
