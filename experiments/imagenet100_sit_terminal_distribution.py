"""Shared definitions for the SiT terminal-distribution control audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AuditCondition:
    """One endpoint-control condition evaluated against a shared baseline."""

    name: str
    mode: str
    gamma: float
    response_scale: float

    def __post_init__(self) -> None:
        if not self.name or any(character in self.name for character in ":,/"):
            raise ValueError("condition names must be non-empty filesystem-safe tokens")
        if self.mode not in {"factorized", "closed"}:
            raise ValueError(f"unsupported audit condition mode: {self.mode}")
        if not np.isfinite(self.gamma) or not np.isfinite(self.response_scale):
            raise ValueError("condition coefficients must be finite")
        if self.mode == "closed" and self.response_scale != 1.0:
            raise ValueError("closed conditions must use response_scale=1")

    @property
    def formula(self) -> str:
        if self.mode == "closed":
            return "z'=S(z)+gamma*[S(z)-W(z)]"
        return "z'=S(b)+rho*[S(z)-S(b)]+gamma*[S(b)-W(b)]"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "gamma": float(self.gamma),
            "response_scale": float(self.response_scale),
            "formula": self.formula,
        }


DEFAULT_AUDIT_CONDITIONS = (
    AuditCondition("factorized_g1_r1p5", "factorized", 1.0, 1.5),
    AuditCondition("factorized_g1p5_r1p35", "factorized", 1.5, 1.35),
    AuditCondition("factorized_g2_r1p35", "factorized", 2.0, 1.35),
    AuditCondition("factorized_g2p5_r1p35", "factorized", 2.5, 1.35),
    AuditCondition("factorized_g3_r1", "factorized", 3.0, 1.0),
    AuditCondition("closed_g3", "closed", 3.0, 1.0),
)


def parse_condition(value: str) -> AuditCondition:
    """Parse ``name:mode:gamma:response_scale`` from the command line."""

    fields = value.split(":")
    if len(fields) != 4:
        raise ValueError(
            "condition must use name:mode:gamma:response_scale, "
            f"received {value!r}"
        )
    name, mode, gamma, response_scale = fields
    return AuditCondition(name, mode, float(gamma), float(response_scale))


def validate_conditions(conditions: tuple[AuditCondition, ...]) -> None:
    if not conditions:
        raise ValueError("at least one audit condition is required")
    names = [condition.name for condition in conditions]
    if len(names) != len(set(names)):
        raise ValueError("audit condition names must be unique")


def factorized_terms(
    anchor_baseline: torch.Tensor,
    anchor_current: torch.Tensor,
    other_baseline: torch.Tensor,
    *,
    gamma: float,
    response_scale: float,
) -> dict[str, torch.Tensor]:
    """Return the exact factorized drift and its control decomposition."""

    nominal_gap = anchor_baseline - other_baseline
    state_response = anchor_current - anchor_baseline
    forcing = float(gamma) * nominal_gap
    response_control = (float(response_scale) - 1.0) * state_response
    control = forcing + response_control
    drift = anchor_current + control
    direct_drift = (
        anchor_baseline
        + float(response_scale) * state_response
        + float(gamma) * nominal_gap
    )
    return {
        "drift": drift,
        "direct_drift": direct_drift,
        "nominal_gap": nominal_gap,
        "state_response": state_response,
        "forcing": forcing,
        "response_control": response_control,
        "control": control,
    }


def closed_terms(
    anchor_current: torch.Tensor,
    other_current: torch.Tensor,
    *,
    gamma: float,
) -> dict[str, torch.Tensor]:
    """Return a closed AutoGuidance drift in the same control notation."""

    current_gap = anchor_current - other_current
    forcing = float(gamma) * current_gap
    zeros = torch.zeros_like(forcing)
    return {
        "drift": anchor_current + forcing,
        "direct_drift": anchor_current + forcing,
        "nominal_gap": current_gap,
        "state_response": zeros,
        "forcing": forcing,
        "response_control": zeros,
        "control": forcing,
    }


def sample_mean_square(value: torch.Tensor) -> torch.Tensor:
    """Mean square over all non-batch dimensions."""

    return value.float().flatten(1).square().mean(dim=1)


def sample_mean_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Mean inner product per coordinate over all non-batch dimensions."""

    if left.shape != right.shape:
        raise ValueError("sample product requires tensors with matching shapes")
    return (left.float() * right.float()).flatten(1).mean(dim=1)


def gaussian_frechet_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    covariance_eps: float = 1e-8,
) -> float:
    """Fréchet distance between Gaussian fits using a symmetric PSD formula."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("feature arrays must be two-dimensional with equal width")
    if len(first) < 2 or len(second) < 2:
        raise ValueError("at least two samples are required")
    mean_first = first.mean(axis=0)
    mean_second = second.mean(axis=0)
    covariance_first = np.cov(first, rowvar=False)
    covariance_second = np.cov(second, rowvar=False)
    dimension = first.shape[1]
    covariance_first = covariance_first + covariance_eps * np.eye(dimension)
    covariance_second = covariance_second + covariance_eps * np.eye(dimension)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_first)
    square_root = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    middle = square_root @ covariance_second @ square_root
    middle = 0.5 * (middle + middle.T)
    trace_square_root = np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None)).sum()
    mean_term = np.square(mean_first - mean_second).sum()
    covariance_term = (
        np.trace(covariance_first) + np.trace(covariance_second) - 2.0 * trace_square_root
    )
    return float(max(mean_term + covariance_term, 0.0))


def sliced_wasserstein_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Root-mean-square 1D Wasserstein distance across feature columns."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("sliced Wasserstein inputs must have equal [N, K] shapes")
    difference = np.sort(first, axis=0) - np.sort(second, axis=0)
    return float(np.sqrt(np.mean(np.square(difference))))


def linear_rbf_mmd2(
    first: np.ndarray,
    second: np.ndarray,
    *,
    bandwidth: float,
) -> float:
    """Linear-time unbiased RBF MMD estimate for equal-sized sample sets."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("linear MMD inputs must have equal [N, K] shapes")
    usable = len(first) - len(first) % 2
    if usable < 2 or not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("linear MMD requires paired samples and positive bandwidth")
    first = first[:usable]
    second = second[:usable]
    scale = 2.0 * float(bandwidth) ** 2

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.exp(-np.square(left - right).sum(axis=1) / scale)

    first_a, first_b = first[0::2], first[1::2]
    second_a, second_b = second[0::2], second[1::2]
    estimate = (
        kernel(first_a, first_b)
        + kernel(second_a, second_b)
        - kernel(first_a, second_b)
        - kernel(first_b, second_a)
    )
    return float(estimate.mean())
