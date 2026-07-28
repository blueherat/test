"""Core operators for time-dependent layerwise RAE probability paths.

The middle layer is used only to identify a spatially predictable channel
subspace inside the final RAE latent.  Splitting the final latent inside that
subspace keeps the endpoint, decoder, and stage-2 tensor shape unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


PathMode = Literal["static", "annealed", "reverse"]
PathFamily = Literal["power", "rational"]


def _token_rows(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got {tuple(latent.shape)}")
    return latent.permute(0, 2, 3, 1).reshape(-1, latent.shape[1])


def spatial_center(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image token mean and spatial residual."""

    if latent.ndim != 4:
        raise ValueError(f"expected BCHW latent, got {tuple(latent.shape)}")
    mean = latent.mean(dim=(-2, -1), keepdim=True)
    return mean, latent - mean


@dataclass
class MiddleFinalCovariance:
    """Streaming moments for reduced-rank middle-to-final regression."""

    middle_gram: torch.Tensor
    middle_final: torch.Tensor
    final_gram: torch.Tensor
    token_count: int = 0

    @classmethod
    def zeros(
        cls,
        channels: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float64,
    ) -> "MiddleFinalCovariance":
        shape = (int(channels), int(channels))
        return cls(
            middle_gram=torch.zeros(shape, device=device, dtype=dtype),
            middle_final=torch.zeros(shape, device=device, dtype=dtype),
            final_gram=torch.zeros(shape, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def update(self, middle: torch.Tensor, final: torch.Tensor) -> None:
        if middle.shape != final.shape:
            raise ValueError(
                "middle and final must share BCHW shape, got "
                f"{tuple(middle.shape)} and {tuple(final.shape)}"
            )
        _, middle_residual = spatial_center(middle)
        _, final_residual = spatial_center(final)
        middle_rows = _token_rows(middle_residual).to(
            device=self.middle_gram.device,
            dtype=self.middle_gram.dtype,
        )
        final_rows = _token_rows(final_residual).to(
            device=self.middle_gram.device,
            dtype=self.middle_gram.dtype,
        )
        self.middle_gram.add_(middle_rows.transpose(0, 1) @ middle_rows)
        self.middle_final.add_(middle_rows.transpose(0, 1) @ final_rows)
        self.final_gram.add_(final_rows.transpose(0, 1) @ final_rows)
        self.token_count += int(middle_rows.shape[0])

    def normalized(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.token_count <= 0:
            raise RuntimeError("no samples were accumulated")
        denominator = float(self.token_count)
        return (
            self.middle_gram / denominator,
            self.middle_final / denominator,
            self.final_gram / denominator,
        )


@dataclass(frozen=True)
class DetailSubspace:
    """Orthogonal channel basis identifying middle-predictable final detail."""

    basis: torch.Tensor
    explained_predictable_fraction: float
    explained_final_fraction: float
    ridge_scale: float
    token_count: int

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def channels(self) -> int:
        return int(self.basis.shape[0])

    def to(self, value: torch.Tensor) -> torch.Tensor:
        return self.basis.to(device=value.device, dtype=value.dtype)


@dataclass(frozen=True)
class PredictabilitySubspace:
    """Final-channel basis ranked by variance-normalized predictability."""

    basis: torch.Tensor
    scores: torch.Tensor
    middle_ridge_scale: float
    final_ridge_scale: float
    token_count: int

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    @property
    def channels(self) -> int:
        return int(self.basis.shape[0])


def _symmetric(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (value + value.transpose(0, 1))


@torch.no_grad()
def fit_middle_to_final_regression(
    moments: MiddleFinalCovariance,
    *,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, float]:
    """Fit the shared channel map ``middle @ W ~= final``."""

    middle_gram, middle_final, _ = moments.normalized()
    if float(ridge) < 0:
        raise ValueError("ridge must be non-negative")
    channels = int(middle_gram.shape[0])
    average_variance = middle_gram.diagonal().mean().clamp_min(1e-12)
    ridge_scale = float(ridge) * average_variance
    regularized = middle_gram + ridge_scale * torch.eye(
        channels,
        device=middle_gram.device,
        dtype=middle_gram.dtype,
    )
    return torch.linalg.solve(regularized, middle_final), float(ridge_scale)


@torch.no_grad()
def fit_fractional_predictability_subspace(
    moments: MiddleFinalCovariance,
    rank: int,
    *,
    ridge: float = 1e-3,
    final_ridge: float = 1e-3,
) -> PredictabilitySubspace:
    """Fit final directions by predictable fraction instead of absolute energy.

    The original SPC basis diagonalizes the covariance of the ridge-regression
    prediction.  That criterion naturally favors high-variance final channels.
    Here the predicted covariance is whitened by final covariance before the
    eigendecomposition, which is the regularized generalized eigenproblem

    ``Cov(predicted final) v = score * Cov(final) v``.

    The returned basis is Euclidean-orthonormal so it remains a valid projector
    for the existing latent split operators.
    """

    middle_gram, _, final_gram = moments.normalized()
    channels = int(middle_gram.shape[0])
    if not 0 < int(rank) <= channels:
        raise ValueError(f"rank must be in [1,{channels}], got {rank}")
    if float(final_ridge) < 0:
        raise ValueError("final_ridge must be non-negative")

    regression, middle_ridge_scale = fit_middle_to_final_regression(
        moments, ridge=ridge
    )
    predicted = _symmetric(regression.transpose(0, 1) @ middle_gram @ regression)

    average_final_variance = final_gram.diagonal().mean().clamp_min(1e-12)
    final_ridge_scale = float(final_ridge) * average_final_variance
    regularized_final = _symmetric(final_gram) + final_ridge_scale * torch.eye(
        channels,
        device=final_gram.device,
        dtype=final_gram.dtype,
    )
    final_values, final_vectors = torch.linalg.eigh(regularized_final)
    floor = torch.finfo(final_values.dtype).eps * final_values.max().clamp_min(1.0)
    inverse_sqrt = (
        final_vectors
        * final_values.clamp_min(floor).rsqrt().unsqueeze(0)
    ) @ final_vectors.transpose(0, 1)
    whitened = _symmetric(inverse_sqrt @ predicted @ inverse_sqrt)
    scores, vectors = torch.linalg.eigh(whitened)
    order = torch.argsort(scores, descending=True)
    selected = order[: int(rank)]
    generalized_vectors = inverse_sqrt @ vectors[:, selected]
    basis, _ = torch.linalg.qr(generalized_vectors, mode="reduced")
    return PredictabilitySubspace(
        basis=basis.float().cpu().contiguous(),
        scores=scores[selected].float().cpu().contiguous(),
        middle_ridge_scale=float(middle_ridge_scale),
        final_ridge_scale=float(final_ridge_scale),
        token_count=int(moments.token_count),
    )


@torch.no_grad()
def fit_final_pca_subspace(
    moments: MiddleFinalCovariance,
    rank: int,
) -> torch.Tensor:
    """Return the top Euclidean-orthonormal final covariance directions."""

    _, _, final_gram = moments.normalized()
    channels = int(final_gram.shape[0])
    if not 0 < int(rank) <= channels:
        raise ValueError(f"rank must be in [1,{channels}], got {rank}")
    values, vectors = torch.linalg.eigh(_symmetric(final_gram))
    selected = torch.argsort(values, descending=True)[: int(rank)]
    return vectors[:, selected].float().cpu().contiguous()


@torch.no_grad()
def subspace_regression_metrics(
    fit_moments: MiddleFinalCovariance,
    evaluation_moments: MiddleFinalCovariance,
    basis: torch.Tensor,
    *,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Evaluate a train-fitted middle-to-final map in one final subspace."""

    regression, _ = fit_middle_to_final_regression(fit_moments, ridge=ridge)
    middle_gram, middle_final, final_gram = evaluation_moments.normalized()
    channels = int(final_gram.shape[0])
    if basis.ndim != 2 or int(basis.shape[0]) != channels:
        raise ValueError(f"basis must have shape [{channels},K]")
    basis = basis.to(device=final_gram.device, dtype=final_gram.dtype)
    basis, _ = torch.linalg.qr(basis, mode="reduced")

    predicted = _symmetric(regression.transpose(0, 1) @ middle_gram @ regression)
    target_prediction = middle_final.transpose(0, 1) @ regression
    residual = _symmetric(
        final_gram
        - target_prediction
        - target_prediction.transpose(0, 1)
        + predicted
    )

    def projected_trace(value: torch.Tensor) -> torch.Tensor:
        return torch.trace(basis.transpose(0, 1) @ value @ basis)

    target_energy = projected_trace(final_gram).clamp_min(1e-20)
    prediction_energy = projected_trace(predicted).clamp_min(0.0)
    cross_energy = projected_trace(target_prediction)
    residual_energy = projected_trace(residual).clamp_min(0.0)
    total_final_energy = torch.trace(final_gram).clamp_min(1e-20)
    correlation = cross_energy / torch.sqrt(
        target_energy * prediction_energy.clamp_min(1e-20)
    )
    return {
        "rank": float(basis.shape[1]),
        "final_energy_fraction": float(target_energy / total_final_energy),
        "final_variance_per_dimension": float(target_energy / basis.shape[1]),
        "prediction_energy_fraction": float(prediction_energy / target_energy),
        "residual_fraction": float(residual_energy / target_energy),
        "r2": float(1.0 - residual_energy / target_energy),
        "aggregate_correlation": float(correlation),
    }


@torch.no_grad()
def fit_detail_subspace(
    moments: MiddleFinalCovariance,
    rank: int,
    *,
    ridge: float = 1e-3,
) -> DetailSubspace:
    """Fit the top final-channel directions predictable from middle tokens.

    Ridge regression gives ``middle @ W ~= final``.  The eigenspace of
    ``W.T @ Cov(middle) @ W`` identifies directions in final-latent channel
    space whose spatial variation is most predictable from the middle layer.
    """

    middle_gram, middle_final, final_gram = moments.normalized()
    channels = int(middle_gram.shape[0])
    if middle_gram.shape != (channels, channels):
        raise ValueError("invalid covariance shape")
    if not 0 < int(rank) <= channels:
        raise ValueError(f"rank must be in [1,{channels}], got {rank}")
    if float(ridge) < 0:
        raise ValueError("ridge must be non-negative")

    average_variance = middle_gram.diagonal().mean().clamp_min(1e-12)
    ridge_scale = float(ridge) * average_variance
    regularized = middle_gram + ridge_scale * torch.eye(
        channels,
        device=middle_gram.device,
        dtype=middle_gram.dtype,
    )
    regression = torch.linalg.solve(regularized, middle_final)
    predictable = regression.transpose(0, 1) @ middle_gram @ regression
    predictable = 0.5 * (predictable + predictable.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(predictable)
    order = torch.argsort(eigenvalues, descending=True)
    selected = order[: int(rank)]
    basis = eigenvectors[:, selected]
    basis, _ = torch.linalg.qr(basis, mode="reduced")

    predictable_values = eigenvalues.clamp_min(0.0)
    selected_predictable = predictable_values[selected].sum()
    total_predictable = predictable_values.sum().clamp_min(1e-12)
    projected_final = torch.trace(basis.transpose(0, 1) @ final_gram @ basis)
    total_final = torch.trace(final_gram).clamp_min(1e-12)
    return DetailSubspace(
        basis=basis.float().cpu().contiguous(),
        explained_predictable_fraction=float(selected_predictable / total_predictable),
        explained_final_fraction=float(projected_final / total_final),
        ridge_scale=float(ridge_scale),
        token_count=int(moments.token_count),
    )


def split_semantic_detail(
    final: torch.Tensor,
    subspace: DetailSubspace | torch.Tensor,
    *,
    detail_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a final latent while preserving its global token mean exactly."""

    mean, residual = spatial_center(final)
    basis = subspace.basis if isinstance(subspace, DetailSubspace) else subspace
    if basis.ndim != 2 or basis.shape[0] != final.shape[1]:
        raise ValueError(
            f"basis must have shape [{final.shape[1]},K], got {tuple(basis.shape)}"
        )
    basis = basis.to(device=final.device, dtype=final.dtype)
    if float(detail_scale) <= 0:
        raise ValueError("detail_scale must be positive")
    rows = _token_rows(residual)
    detail_rows = float(detail_scale) * (rows @ basis) @ basis.transpose(0, 1)
    detail = detail_rows.reshape(
        final.shape[0], final.shape[2], final.shape[3], final.shape[1]
    ).permute(0, 3, 1, 2).contiguous()
    semantic = mean + (residual - detail)
    return semantic, detail


def random_detail_basis(
    channels: int,
    rank: int,
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not 0 < int(rank) <= int(channels):
        raise ValueError("rank must be positive and no larger than channels")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((int(channels), int(rank)), generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.to(dtype=dtype).contiguous()


def path_coefficients(
    time: torch.Tensor,
    mode: PathMode,
    *,
    power: float = 1.0,
    family: PathFamily = "power",
    floor: float = 0.0,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return semantic/detail coefficients and their time derivatives."""

    if time.ndim != 1:
        raise ValueError(f"time must have shape [B], got {tuple(time.shape)}")
    if not 0.0 <= float(floor) < 1.0:
        raise ValueError("floor must lie in [0, 1)")
    one = torch.ones_like(time)
    zero = torch.zeros_like(time)
    remaining = (1.0 - time).clamp_min(0.0)
    if family == "power":
        if float(power) < 1.0:
            raise ValueError("power must be at least 1 to keep endpoint derivatives finite")
        fading = float(floor) + (1.0 - float(floor)) * remaining.pow(float(power))
        derivative = (
            -(1.0 - float(floor))
            * float(power)
            * remaining.pow(float(power) - 1.0)
        )
    elif family == "rational":
        if float(alpha) <= 0.0:
            raise ValueError("alpha must be positive")
        denominator = 1.0 + float(alpha) * time
        fading = float(floor) + (1.0 - float(floor)) * remaining / denominator
        derivative = (
            -(1.0 - float(floor))
            * (1.0 + float(alpha))
            / denominator.square()
        )
    else:
        raise ValueError(f"unknown path family: {family}")
    if mode == "static":
        return one, one, zero, zero
    if mode == "annealed":
        return one, fading, zero, derivative
    if mode == "reverse":
        return fading, one, derivative, zero
    raise ValueError(f"unknown path mode: {mode}")


@dataclass(frozen=True)
class LayerwisePathPlan:
    state: torch.Tensor
    target: torch.Tensor
    data_state: torch.Tensor
    data_derivative: torch.Tensor
    semantic: torch.Tensor
    detail: torch.Tensor


def plan_layerwise_path(
    clean: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    subspace: DetailSubspace | torch.Tensor,
    *,
    mode: PathMode,
    power: float = 1.0,
    family: PathFamily = "power",
    floor: float = 0.0,
    alpha: float = 1.0,
    detail_scale: float = 1.0,
) -> LayerwisePathPlan:
    """Construct state and exact velocity target for a layerwise RAE path."""

    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have the same shape")
    if len(clean) != len(time):
        raise ValueError("time batch size must match latent batch size")
    semantic, detail = split_semantic_detail(
        clean, subspace, detail_scale=float(detail_scale)
    )
    sem_c, detail_c, sem_d, detail_d = path_coefficients(
        time,
        mode,
        power=power,
        family=family,
        floor=floor,
        alpha=alpha,
    )
    expand = (-1,) + (1,) * (clean.ndim - 1)
    sem_c = sem_c.reshape(expand)
    detail_c = detail_c.reshape(expand)
    sem_d = sem_d.reshape(expand)
    detail_d = detail_d.reshape(expand)
    time_expanded = time.reshape(expand)
    data_state = sem_c * semantic + detail_c * detail
    data_derivative = sem_d * semantic + detail_d * detail
    state = (1.0 - time_expanded) * data_state + time_expanded * noise
    target = noise - data_state + (1.0 - time_expanded) * data_derivative
    return LayerwisePathPlan(
        state=state,
        target=target,
        data_state=data_state,
        data_derivative=data_derivative,
        semantic=semantic,
        detail=detail,
    )
