"""Basis-invariant metrics for a RAE encoder-decoder cross-layer atlas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


def spatial_center_rows(state: torch.Tensor) -> torch.Tensor:
    if state.ndim != 3:
        raise ValueError(f"expected [B,N,C], got {tuple(state.shape)}")
    centered = state.float() - state.float().mean(dim=1, keepdim=True)
    return centered.reshape(-1, centered.shape[-1])


def token_gram_vectors(states: Sequence[torch.Tensor], eps: float = 1e-12) -> torch.Tensor:
    """Return normalized centered token-Gram vectors as ``[B, L, N*N]``.

    Each state has shape ``[B, N, C]``.  Comparing token Gram matrices makes
    the metric insensitive to channel width and orthogonal channel gauges.
    """

    if not states:
        raise ValueError("states must be non-empty")
    batch = int(states[0].shape[0])
    tokens = int(states[0].shape[1])
    vectors = []
    for state in states:
        if state.ndim != 3:
            raise ValueError(f"expected [B,N,C], got {tuple(state.shape)}")
        if state.shape[:2] != (batch, tokens):
            raise ValueError("all states must share batch and token dimensions")
        centered = state.float() - state.float().mean(dim=1, keepdim=True)
        gram = torch.bmm(centered, centered.transpose(1, 2)) / float(state.shape[-1])
        vector = gram.flatten(1)
        vector = vector / vector.norm(dim=1, keepdim=True).clamp_min(float(eps))
        vectors.append(vector)
    return torch.stack(vectors, dim=1)


def cross_layer_cka(
    encoder_states: Sequence[torch.Tensor],
    decoder_states: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Compute per-sample spatial linear CKA for every encoder-decoder pair."""

    encoder = token_gram_vectors(encoder_states)
    decoder = token_gram_vectors(decoder_states)
    if encoder.shape[0] != decoder.shape[0] or encoder.shape[-1] != decoder.shape[-1]:
        raise ValueError("encoder and decoder states must share batch and token count")
    return torch.einsum("bek,bdk->bed", encoder, decoder).clamp(min=-1.0, max=1.0)


def paired_and_mismatched_cka(
    encoder_states: Sequence[torch.Tensor],
    decoder_states: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return paired CKA and a rolled-image control for shared position bias."""

    encoder = token_gram_vectors(encoder_states)
    decoder = token_gram_vectors(decoder_states)
    if encoder.shape[0] != decoder.shape[0] or encoder.shape[-1] != decoder.shape[-1]:
        raise ValueError("encoder and decoder states must share batch and token count")
    paired = torch.einsum("bek,bdk->bed", encoder, decoder).clamp(min=-1.0, max=1.0)
    if encoder.shape[0] < 2:
        return paired, None
    mismatched = torch.einsum(
        "bek,bdk->bed",
        encoder,
        torch.roll(decoder, shifts=1, dims=0),
    ).clamp(min=-1.0, max=1.0)
    return paired, mismatched


@dataclass
class CKAMoments:
    paired_sum: torch.Tensor
    paired_square_sum: torch.Tensor
    paired_count: torch.Tensor
    mismatched_sum: torch.Tensor
    mismatched_square_sum: torch.Tensor
    mismatched_count: torch.Tensor

    @classmethod
    def zeros(
        cls,
        encoder_layers: int,
        decoder_layers: int,
        *,
        device: torch.device | str,
    ) -> "CKAMoments":
        shape = (int(encoder_layers), int(decoder_layers))
        matrix = lambda: torch.zeros(shape, dtype=torch.float64, device=device)
        scalar = lambda: torch.zeros((), dtype=torch.float64, device=device)
        return cls(matrix(), matrix(), scalar(), matrix(), matrix(), scalar())

    @torch.no_grad()
    def update(self, paired: torch.Tensor, mismatched: torch.Tensor | None) -> None:
        if paired.ndim != 3 or paired.shape[1:] != self.paired_sum.shape:
            raise ValueError("paired CKA has incompatible shape")
        values = paired.detach().to(device=self.paired_sum.device, dtype=torch.float64)
        self.paired_sum.add_(values.sum(dim=0))
        self.paired_square_sum.add_(values.square().sum(dim=0))
        self.paired_count.add_(len(values))
        if mismatched is not None:
            if mismatched.shape != paired.shape:
                raise ValueError("mismatched CKA must match paired CKA")
            control = mismatched.detach().to(
                device=self.mismatched_sum.device,
                dtype=torch.float64,
            )
            self.mismatched_sum.add_(control.sum(dim=0))
            self.mismatched_square_sum.add_(control.square().sum(dim=0))
            self.mismatched_count.add_(len(control))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.paired_sum,
            self.paired_square_sum,
            self.paired_count,
            self.mismatched_sum,
            self.mismatched_square_sum,
            self.mismatched_count,
        )

    def summary(self) -> dict[str, torch.Tensor | int]:
        paired_count = self.paired_count.clamp_min(1.0)
        paired_mean = self.paired_sum / paired_count
        paired_var = self.paired_square_sum / paired_count - paired_mean.square()
        mismatch_count = self.mismatched_count.clamp_min(1.0)
        mismatch_mean = self.mismatched_sum / mismatch_count
        mismatch_var = self.mismatched_square_sum / mismatch_count - mismatch_mean.square()
        return {
            "paired_mean": paired_mean,
            "paired_std": paired_var.clamp_min(0.0).sqrt(),
            "mismatched_mean": mismatch_mean,
            "mismatched_std": mismatch_var.clamp_min(0.0).sqrt(),
            "excess_mean": paired_mean - mismatch_mean,
            "paired_count": int(self.paired_count.item()),
            "mismatched_count": int(self.mismatched_count.item()),
        }


@dataclass
class RidgeMoments:
    decoder_gram: torch.Tensor
    decoder_encoder: torch.Tensor
    target_energy: torch.Tensor
    token_count: torch.Tensor

    @classmethod
    def zeros(
        cls,
        decoder_channels: int,
        encoder_channels: int,
        *,
        device: torch.device | str,
    ) -> "RidgeMoments":
        return cls(
            decoder_gram=torch.zeros(
                (decoder_channels, decoder_channels), dtype=torch.float64, device=device
            ),
            decoder_encoder=torch.zeros(
                (decoder_channels, encoder_channels), dtype=torch.float64, device=device
            ),
            target_energy=torch.zeros((), dtype=torch.float64, device=device),
            token_count=torch.zeros((), dtype=torch.float64, device=device),
        )

    @torch.no_grad()
    def update(self, decoder: torch.Tensor, encoder: torch.Tensor) -> None:
        if decoder.shape[:2] != encoder.shape[:2]:
            raise ValueError("decoder and encoder states must share batch and token dimensions")
        decoder_rows = spatial_center_rows(decoder).to(
            device=self.decoder_gram.device,
            dtype=torch.float64,
        )
        encoder_rows = spatial_center_rows(encoder).to(
            device=self.decoder_gram.device,
            dtype=torch.float64,
        )
        if decoder_rows.shape[1] != self.decoder_gram.shape[0]:
            raise ValueError("decoder channel dimension disagrees with moments")
        if encoder_rows.shape[1] != self.decoder_encoder.shape[1]:
            raise ValueError("encoder channel dimension disagrees with moments")
        self.decoder_gram.add_(decoder_rows.transpose(0, 1) @ decoder_rows)
        self.decoder_encoder.add_(decoder_rows.transpose(0, 1) @ encoder_rows)
        self.target_energy.add_(encoder_rows.square().sum())
        self.token_count.add_(len(decoder_rows))

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.decoder_gram,
            self.decoder_encoder,
            self.target_energy,
            self.token_count,
        )


@torch.no_grad()
def fit_ridge_map(moments: RidgeMoments, ridge: float = 1e-3) -> tuple[torch.Tensor, float]:
    if float(ridge) < 0:
        raise ValueError("ridge must be non-negative")
    if float(moments.token_count) <= 0:
        raise RuntimeError("cannot fit ridge map without observations")
    channels = moments.decoder_gram.shape[0]
    average_energy = moments.decoder_gram.diagonal().mean().clamp_min(1e-12)
    ridge_scale = float(ridge) * average_energy
    regularized = moments.decoder_gram + ridge_scale * torch.eye(
        channels,
        device=moments.decoder_gram.device,
        dtype=moments.decoder_gram.dtype,
    )
    mapping = torch.linalg.solve(regularized, moments.decoder_encoder)
    return mapping.float(), float(ridge_scale)


@torch.no_grad()
def linear_probe_scores(
    decoder: torch.Tensor,
    encoder: torch.Tensor,
    mapping: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if decoder.shape[:2] != encoder.shape[:2]:
        raise ValueError("decoder and encoder states must share batch and token dimensions")
    decoder_centered = decoder.float() - decoder.float().mean(dim=1, keepdim=True)
    encoder_centered = encoder.float() - encoder.float().mean(dim=1, keepdim=True)
    if mapping.shape != (decoder.shape[-1], encoder.shape[-1]):
        raise ValueError("mapping shape disagrees with decoder and encoder channels")
    prediction = decoder_centered @ mapping.to(decoder_centered)
    error = (prediction - encoder_centered).square().flatten(1).sum(dim=1)
    target_energy = encoder_centered.square().flatten(1).sum(dim=1).clamp_min(float(eps))
    relative_error = torch.sqrt(error / target_energy)
    cosine = torch.nn.functional.cosine_similarity(
        prediction.flatten(1),
        encoder_centered.flatten(1),
        dim=1,
        eps=float(eps),
    )
    return relative_error, cosine


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.double().flatten()
    _, inverse, counts = torch.unique(
        values,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    starts = counts.cumsum(dim=0) - counts
    average_ranks = starts.double() + (counts.double() - 1.0) / 2.0
    return average_ranks[inverse]


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().flatten()
    right = right.double().flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float((left @ right / denominator).item())


def spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel() or left.numel() < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    return _correlation(_rankdata(left.flatten()), _rankdata(right.flatten()))


def summarize_atlas(score: torch.Tensor) -> dict[str, object]:
    """Summarize whether decoder depth follows a reverse encoder hierarchy."""

    if score.ndim != 2 or min(score.shape) < 2:
        raise ValueError("atlas must be a two-dimensional matrix")
    finite = torch.nan_to_num(score.detach().double().cpu())
    best_encoder = finite.argmax(dim=0)
    decoder_depth = torch.arange(finite.shape[1], dtype=torch.float64)
    reverse_spearman = -spearman_correlation(decoder_depth, best_encoder.double())
    monotone_pairs = (best_encoder[1:] <= best_encoder[:-1]).double().mean()
    positive = finite.clamp_min(0.0)
    denominator = positive.sum(dim=0).clamp_min(1e-12)
    encoder_depth = torch.arange(finite.shape[0], dtype=torch.float64).view(-1, 1)
    soft_depth = (positive * encoder_depth).sum(dim=0) / denominator
    soft_reverse_spearman = -spearman_correlation(decoder_depth, soft_depth)
    return {
        "best_encoder_indices": [int(value) for value in best_encoder],
        "reverse_spearman_argmax": reverse_spearman,
        "reverse_spearman_soft": soft_reverse_spearman,
        "nonincreasing_pair_fraction": float(monotone_pairs),
        "mean_peak_score": float(finite.max(dim=0).values.mean()),
        "mean_score": float(finite.mean()),
    }


def compare_atlases(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("atlases must have equal shape")
    reference = torch.nan_to_num(reference.detach().double().cpu())
    candidate = torch.nan_to_num(candidate.detach().double().cpu())
    reference_best = reference.argmax(dim=0)
    candidate_best = candidate.argmax(dim=0)
    return {
        "pearson": _correlation(reference, candidate),
        "rms_distance": float((reference - candidate).square().mean().sqrt()),
        "exact_mapping_rate": float((reference_best == candidate_best).double().mean()),
        "within_one_mapping_rate": float(
            ((reference_best - candidate_best).abs() <= 1).double().mean()
        ),
    }


__all__ = [
    "CKAMoments",
    "RidgeMoments",
    "compare_atlases",
    "cross_layer_cka",
    "fit_ridge_map",
    "linear_probe_scores",
    "paired_and_mismatched_cka",
    "spatial_center_rows",
    "spearman_correlation",
    "summarize_atlas",
    "token_gram_vectors",
]
