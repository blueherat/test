"""Independent mathematical core for FD-Loss and AdvFD experiments.

The implementation follows the equations in the papers, not the official AdvFD
repository. Samples are rows and feature dimensions are columns throughout.
Population (biased) covariances are used because the losses estimate population
moments rather than an unbiased finite-sample statistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Moments:
    mean: torch.Tensor
    second: torch.Tensor
    covariance: torch.Tensor

    def detached(self) -> "Moments":
        return Moments(
            mean=self.mean.detach(),
            second=self.second.detach(),
            covariance=self.covariance.detach(),
        )


@dataclass(frozen=True)
class FrechetComponents:
    mean: torch.Tensor
    covariance: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.mean + self.covariance


@dataclass(frozen=True)
class AffineCalibration:
    center: torch.Tensor
    transform: torch.Tensor

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        _require_feature_matrix(features)
        return (features - self.center) @ self.transform


def _require_feature_matrix(features: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError(
            f"Expected a [samples, features] matrix, got shape {tuple(features.shape)}"
        )
    if features.shape[0] < 1:
        raise ValueError("At least one sample is required")


def _symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.mT)


def moments_from_mean_and_second(
    mean: torch.Tensor, second: torch.Tensor
) -> Moments:
    covariance = _symmetrize(second - torch.outer(mean, mean))
    return Moments(mean=mean, second=second, covariance=covariance)


def moments_from_mean_and_covariance(
    mean: torch.Tensor, covariance: torch.Tensor
) -> Moments:
    covariance = _symmetrize(covariance)
    second = covariance + torch.outer(mean, mean)
    return Moments(mean=mean, second=second, covariance=covariance)


def batch_moments(features: torch.Tensor) -> Moments:
    """Compute population first and second moments for a feature batch."""

    _require_feature_matrix(features)
    mean = features.mean(dim=0)
    second = features.mT @ features / features.shape[0]
    return moments_from_mean_and_second(mean, second)


def mixture_moments(first: Moments, second: Moments) -> Moments:
    """Moments of the equally weighted mixture of two distributions."""

    mean = 0.5 * (first.mean + second.mean)
    raw_second = 0.5 * (first.second + second.second)
    return moments_from_mean_and_second(mean, raw_second)


def project_moments(moments: Moments, projection: torch.Tensor) -> Moments:
    """Push moments through the row-vector map ``x -> x @ projection``."""

    if projection.ndim != 2 or projection.shape[0] != moments.mean.numel():
        raise ValueError(
            "Projection must have shape [input_dim, output_dim], got "
            f"{tuple(projection.shape)} for input dim {moments.mean.numel()}"
        )
    mean = moments.mean @ projection
    covariance = projection.mT @ moments.covariance @ projection
    return moments_from_mean_and_covariance(mean, covariance)


def symmetric_matrix_sqrt(
    matrix: torch.Tensor, *, eigenvalue_floor: float = 0.0
) -> torch.Tensor:
    """Return the symmetric PSD square root using an eigendecomposition."""

    matrix = _symmetrize(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eigenvalue_floor).sqrt()
    return _symmetrize((eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.mT)


def symmetric_matrix_inverse_sqrt(
    matrix: torch.Tensor, *, eigenvalue_floor: float = 1e-12
) -> torch.Tensor:
    """Return a regularized symmetric inverse square root."""

    if eigenvalue_floor <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")
    matrix = _symmetrize(matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    inverse_roots = eigenvalues.clamp_min(eigenvalue_floor).rsqrt()
    return _symmetrize(
        (eigenvectors * inverse_roots.unsqueeze(0)) @ eigenvectors.mT
    )


def frechet_from_moments(
    first: Moments,
    second: Moments,
    *,
    covariance_jitter: float = 0.0,
) -> FrechetComponents:
    """Squared Gaussian 2-Wasserstein distance split into mean/covariance terms."""

    if first.mean.shape != second.mean.shape:
        raise ValueError("Moment dimensions must match")
    dimension = first.mean.numel()
    identity = torch.eye(
        dimension, dtype=first.covariance.dtype, device=first.covariance.device
    )
    covariance_first = _symmetrize(first.covariance)
    covariance_second = _symmetrize(second.covariance)
    if covariance_jitter:
        covariance_first = covariance_first + covariance_jitter * identity
        covariance_second = covariance_second + covariance_jitter * identity

    root_first = symmetric_matrix_sqrt(covariance_first)
    covariance_product = _symmetrize(
        root_first @ covariance_second @ root_first
    )
    cross_trace = (
        torch.linalg.eigvalsh(covariance_product).clamp_min(0.0).sqrt().sum()
    )
    mean_term = (first.mean - second.mean).square().sum()
    covariance_term = (
        torch.trace(covariance_first)
        + torch.trace(covariance_second)
        - 2.0 * cross_trace
    )
    # Tiny negative values can arise from eigendecomposition roundoff. Keeping a
    # differentiable clamp avoids reporting an impossible negative FD.
    covariance_term = covariance_term.clamp_min(0.0)
    return FrechetComponents(mean=mean_term, covariance=covariance_term)


def frechet_from_features(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    covariance_jitter: float = 0.0,
) -> FrechetComponents:
    return frechet_from_moments(
        batch_moments(first),
        batch_moments(second),
        covariance_jitter=covariance_jitter,
    )


CalibrationMode = Literal["none", "real", "pooled"]


def fit_calibration_from_moments(
    real: Moments,
    generated: Moments | None = None,
    *,
    mode: CalibrationMode,
    epsilon: float = 1e-3,
    detach_statistics: bool = True,
) -> AffineCalibration:
    """Fit the common affine map directly from population moments."""

    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    dimension = real.mean.numel()
    if mode == "none":
        return AffineCalibration(
            center=torch.zeros_like(real.mean),
            transform=torch.eye(
                dimension,
                dtype=real.covariance.dtype,
                device=real.covariance.device,
            ),
        )
    if mode == "pooled":
        if generated is None:
            raise ValueError("Pooled calibration requires generated moments")
        if generated.mean.shape != real.mean.shape:
            raise ValueError("Real and generated moment dimensions must match")
        reference = mixture_moments(real, generated)
    elif mode == "real":
        reference = real
    else:
        raise ValueError(f"Unknown calibration mode: {mode!r}")
    if detach_statistics:
        reference = reference.detached()

    identity = torch.eye(
        dimension,
        dtype=reference.covariance.dtype,
        device=reference.covariance.device,
    )
    whitener = symmetric_matrix_inverse_sqrt(
        reference.covariance + epsilon * identity,
        eigenvalue_floor=max(epsilon, torch.finfo(reference.covariance.dtype).eps),
    )
    return AffineCalibration(center=reference.mean, transform=whitener)


def calibrate_moments(
    moments: Moments, calibration: AffineCalibration
) -> Moments:
    """Push moments through a common affine calibration."""

    if moments.mean.shape != calibration.center.shape:
        raise ValueError("Moment and calibration dimensions must match")
    mean = (moments.mean - calibration.center) @ calibration.transform
    covariance = (
        calibration.transform.mT
        @ moments.covariance
        @ calibration.transform
    )
    return moments_from_mean_and_covariance(mean, covariance)


def fit_calibration(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    *,
    mode: CalibrationMode,
    epsilon: float = 1e-3,
    detach_statistics: bool = True,
) -> AffineCalibration:
    """Fit the common affine map used by a calibrated feature discrepancy."""

    _require_feature_matrix(real_features)
    _require_feature_matrix(generated_features)
    if real_features.shape[1] != generated_features.shape[1]:
        raise ValueError("Real and generated feature dimensions must match")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")

    real = batch_moments(real_features)
    generated = batch_moments(generated_features)
    return fit_calibration_from_moments(
        real,
        generated,
        mode=mode,
        epsilon=epsilon,
        detach_statistics=detach_statistics,
    )


def calibrate_features(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    *,
    mode: CalibrationMode,
    epsilon: float = 1e-3,
    detach_statistics: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a common affine calibration before evaluating feature FD.

    ``real`` follows AdvFD's real-reference whitening. ``pooled`` is a
    Fisher-style control based on the equally weighted real/generated mixture.
    The latter is included as a mechanism control, not claimed as novel.
    """

    calibration = fit_calibration(
        real_features,
        generated_features,
        mode=mode,
        epsilon=epsilon,
        detach_statistics=detach_statistics,
    )
    return calibration.apply(real_features), calibration.apply(generated_features)


def normalized_frechet_loss(
    components: FrechetComponents, *, constant: float = 0.01
) -> torch.Tensor:
    """FD-Loss normalization with a stop-gradient denominator."""

    if constant <= 0.0:
        raise ValueError("constant must be positive")
    distance = components.total
    return distance / (distance.detach() + constant)


class EMAMomentTracker:
    """Track detached history while retaining current-batch gradients.

    ``update`` returns the effective moments used by the current loss and commits
    a detached copy for the next step. The state can be warm-started from a large
    base-model sample using ``initialize``.
    """

    def __init__(self, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must lie in [0, 1)")
        self.decay = decay
        self._mean: torch.Tensor | None = None
        self._second: torch.Tensor | None = None

    @property
    def initialized(self) -> bool:
        return self._mean is not None

    def initialize(self, features: torch.Tensor) -> Moments:
        moments = batch_moments(features).detached()
        return self.initialize_from_moments(moments)

    def initialize_from_moments(self, moments: Moments) -> Moments:
        moments = moments.detached()
        self._mean = moments.mean
        self._second = moments.second
        return moments

    def preview(self, features: torch.Tensor) -> Moments:
        """Return the next EMA moments without advancing persistent state."""

        return self.preview_moments(batch_moments(features))

    def preview_moments(self, current: Moments) -> Moments:
        if self._mean is None or self._second is None:
            return current
        previous_mean = self._mean.detach().to(
            device=current.mean.device, dtype=current.mean.dtype
        )
        previous_second = self._second.detach().to(
            device=current.second.device, dtype=current.second.dtype
        )
        return moments_from_mean_and_second(
            self.decay * previous_mean + (1.0 - self.decay) * current.mean,
            self.decay * previous_second + (1.0 - self.decay) * current.second,
        )

    def commit(self, moments: Moments) -> Moments:
        """Advance persistent state with a detached effective-moment snapshot."""

        self._mean = moments.mean.detach()
        self._second = moments.second.detach()
        return moments

    def update(self, features: torch.Tensor) -> Moments:
        return self.commit(self.preview(features))

    def state_dict(self) -> dict[str, torch.Tensor | float | None]:
        return {
            "decay": self.decay,
            "mean": self._mean,
            "second": self._second,
        }

    def load_state_dict(
        self, state: dict[str, torch.Tensor | float | None]
    ) -> None:
        decay = float(state["decay"])
        if decay != self.decay:
            raise ValueError(
                f"EMA decay mismatch: tracker={self.decay}, checkpoint={decay}"
            )
        mean = state["mean"]
        second = state["second"]
        self._mean = None if mean is None else torch.as_tensor(mean).detach()
        self._second = None if second is None else torch.as_tensor(second).detach()


class StreamingMomentAccumulator:
    """Accumulate population moments without retaining feature batches."""

    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        self.dtype = dtype
        self.count = 0
        self._sum: torch.Tensor | None = None
        self._second_sum: torch.Tensor | None = None

    def update(self, features: torch.Tensor) -> None:
        _require_feature_matrix(features)
        values = features.detach().to(dtype=self.dtype)
        batch_sum = values.sum(dim=0)
        batch_second = values.mT @ values
        if self._sum is None:
            self._sum = batch_sum
            self._second_sum = batch_second
        else:
            if self._sum.shape != batch_sum.shape:
                raise ValueError("Feature dimension changed while accumulating moments")
            self._sum = self._sum + batch_sum
            assert self._second_sum is not None
            self._second_sum = self._second_sum + batch_second
        self.count += int(features.shape[0])

    def moments(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Moments:
        if self.count == 0 or self._sum is None or self._second_sum is None:
            raise RuntimeError("No features have been accumulated")
        mean = self._sum / self.count
        second = self._second_sum / self.count
        if device is not None:
            mean = mean.to(device=device)
            second = second.to(device=device)
        mean = mean.to(dtype=dtype)
        second = second.to(dtype=dtype)
        return moments_from_mean_and_second(mean, second)
