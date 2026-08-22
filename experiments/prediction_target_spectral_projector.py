"""Low-rank spectral operators for geometry-aware prediction targets.

The toy path convention is

    z_t = (1 - t) x + t epsilon,    v = epsilon - x.

For a symmetric operator K, the generalized target is

    u_K = K x - (I - K) epsilon.

The exact conversion back to velocity is

    v = D_K(t)^-1 ((2 K - I) z_t - u_K),
    D_K(t) = (1 - t) I + (2 t - 1) K.

This module represents a soft projector as a low-rank eigendecomposition and
never materializes a dense D x D matrix during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectralProjector:
    """Symmetric soft projector P = U diag(w) U^T.

    ``basis`` has orthonormal columns and ``weights`` lie in [0, 1].  Empty
    tensors represent the zero projector.  The object is intentionally frozen:
    projector estimation is separated from predictor optimization to prevent
    the two from colluding through the common velocity loss.
    """

    basis: torch.Tensor
    weights: torch.Tensor
    source: str

    def __post_init__(self) -> None:
        if self.basis.ndim != 2:
            raise ValueError("basis must be [D,r]")
        if self.weights.ndim != 1 or self.weights.shape[0] != self.basis.shape[1]:
            raise ValueError("weights must have one value per basis column")
        if self.basis.device != self.weights.device:
            raise ValueError("basis and weights must be on the same device")
        if self.basis.dtype != self.weights.dtype:
            raise ValueError("basis and weights must have the same dtype")
        if torch.any(self.weights < 0) or torch.any(self.weights > 1):
            raise ValueError("projector weights must lie in [0,1]")
        if self.basis.shape[1]:
            gram = self.basis.T.float() @ self.basis.float()
            identity = torch.eye(len(self.weights), device=gram.device)
            if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
                raise ValueError("basis columns must be orthonormal")

    @property
    def ambient_dim(self) -> int:
        return int(self.basis.shape[0])

    @property
    def rank(self) -> int:
        return int(self.basis.shape[1])

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.ambient_dim:
            raise ValueError("value has the wrong ambient dimension")
        if self.rank == 0:
            return torch.zeros_like(value)
        coefficients = value @ self.basis
        return (coefficients * self.weights) @ self.basis.T


@dataclass(frozen=True)
class SpectralTarget:
    """Operator target K = k_normal I + (k_tangent-k_normal) P."""

    name: str
    projector: SpectralProjector
    tangent_k: float
    normal_k: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.tangent_k <= 1.0):
            raise ValueError("tangent_k must lie in [0,1]")
        if not (0.0 <= self.normal_k <= 1.0):
            raise ValueError("normal_k must lie in [0,1]")

    def _basis_k(self) -> torch.Tensor:
        return self.normal_k + (self.tangent_k - self.normal_k) * self.projector.weights

    def apply_k(self, value: torch.Tensor) -> torch.Tensor:
        result = self.normal_k * value
        if self.projector.rank:
            coefficients = value @ self.projector.basis
            correction = self._basis_k() - self.normal_k
            result = result + (coefficients * correction) @ self.projector.basis.T
        return result

    def target(self, clean: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        # Kx - (I-K)epsilon = K(x+epsilon) - epsilon.
        return self.apply_k(clean + epsilon) - epsilon

    def velocity(
        self,
        output: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
        conversion_clip: float,
    ) -> torch.Tensor:
        """Convert a predicted generalized target to velocity exactly."""
        rhs = 2.0 * self.apply_k(state) - state - output
        normal_denominator = (
            (1.0 - time) + (2.0 * time - 1.0) * self.normal_k
        ).clamp_min(conversion_clip)
        result = rhs / normal_denominator[:, None]
        if self.projector.rank:
            basis_k = self._basis_k()
            basis_denominator = (
                (1.0 - time[:, None])
                + (2.0 * time[:, None] - 1.0) * basis_k[None]
            ).clamp_min(conversion_clip)
            coefficients = rhs @ self.projector.basis
            inverse_correction = (
                basis_denominator.reciprocal()
                - normal_denominator[:, None].reciprocal()
            )
            result = result + (
                coefficients * inverse_correction
            ) @ self.projector.basis.T
        return result

    def clean(
        self,
        output: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
        conversion_clip: float,
    ) -> torch.Tensor:
        return state - time[:, None] * self.velocity(
            output, state, time, conversion_clip
        )


def zero_projector(
    ambient_dim: int, *, device: torch.device, dtype: torch.dtype, source: str
) -> SpectralProjector:
    return SpectralProjector(
        basis=torch.empty(ambient_dim, 0, device=device, dtype=dtype),
        weights=torch.empty(0, device=device, dtype=dtype),
        source=source,
    )


def hard_projector(basis: torch.Tensor, *, source: str) -> SpectralProjector:
    return SpectralProjector(
        basis=basis,
        weights=torch.ones(basis.shape[1], device=basis.device, dtype=basis.dtype),
        source=source,
    )


def estimate_pca_projector(
    samples: torch.Tensor,
    *,
    rank: int,
    source: str,
) -> tuple[SpectralProjector, torch.Tensor]:
    """Estimate a hard rank-r projector from a detached sample bank."""
    if samples.ndim != 2:
        raise ValueError("samples must be [N,D]")
    if not (1 <= rank <= min(samples.shape)):
        raise ValueError("rank must be in [1,min(N,D)]")
    centered = samples.detach() - samples.detach().mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    basis = vh[:rank].T.to(device=samples.device, dtype=samples.dtype)
    eigenvalues = singular_values.square() / max(len(samples) - 1, 1)
    return hard_projector(basis, source=source), eigenvalues.to(samples.dtype)


def estimate_soft_spectral_projector(
    samples: torch.Tensor,
    *,
    tau_ratio: float,
    max_rank: int,
    min_weight: float,
    source: str,
) -> tuple[SpectralProjector, torch.Tensor, float]:
    """Estimate P_tau = Sigma (Sigma + tau I)^-1 from clean samples.

    The shrinkage weights are continuous, basis invariant within degenerate
    eigenspaces, and avoid choosing a hard intrinsic rank.  Small-weight modes
    are omitted only as a computational approximation.
    """
    if samples.ndim != 2:
        raise ValueError("samples must be [N,D]")
    if tau_ratio <= 0:
        raise ValueError("tau_ratio must be positive")
    if not (1 <= max_rank <= min(samples.shape)):
        raise ValueError("max_rank must be in [1,min(N,D)]")
    if not (0.0 <= min_weight < 1.0):
        raise ValueError("min_weight must lie in [0,1)")
    centered = samples.detach() - samples.detach().mean(dim=0, keepdim=True)
    _u, singular_values, vh = torch.linalg.svd(centered.float(), full_matrices=False)
    eigenvalues = singular_values.square() / max(len(samples) - 1, 1)
    tau = float(tau_ratio * eigenvalues[0].item())
    weights = eigenvalues / (eigenvalues + tau)
    keep = min(max_rank, int((weights >= min_weight).sum().item()))
    keep = max(1, keep)
    basis = vh[:keep].T.to(device=samples.device, dtype=samples.dtype)
    retained_weights = weights[:keep].to(device=samples.device, dtype=samples.dtype)
    projector = SpectralProjector(
        basis=basis,
        weights=retained_weights,
        source=source,
    )
    return projector, eigenvalues.to(samples.dtype), tau


def projector_alignment(
    estimated: SpectralProjector,
    true_basis: torch.Tensor,
) -> dict[str, float]:
    """Report hard-subspace alignment without using it in estimation."""
    if true_basis.ndim != 2 or true_basis.shape[0] != estimated.ambient_dim:
        raise ValueError("true_basis has incompatible shape")
    if estimated.rank == 0:
        return {
            "principal_cosine_min": 0.0,
            "principal_cosine_mean": 0.0,
            "projector_frobenius_error": 1.0,
        }
    overlap = true_basis.T.float() @ estimated.basis.float()
    singular_values = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
    true_projector = true_basis.float() @ true_basis.float().T
    estimated_projector = (
        estimated.basis.float()
        * estimated.weights.float()[None]
    ) @ estimated.basis.float().T
    denominator = torch.linalg.norm(true_projector).clamp_min(1e-12)
    return {
        "principal_cosine_min": float(singular_values.min().cpu()),
        "principal_cosine_mean": float(singular_values.mean().cpu()),
        "projector_frobenius_error": float(
            (torch.linalg.norm(estimated_projector - true_projector) / denominator).cpu()
        ),
    }
