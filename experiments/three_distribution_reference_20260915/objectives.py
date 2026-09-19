"""A fixed-real-component mixture objective and its endpoint EM update.

P: real endpoint law; G: frozen strong sampler endpoint law; Q: CURRENT weak
sampler endpoint law. M = (1-kappa) P + kappa Q is fitted to G. The mixture
projection is known from AdaGAN (2017); the three-source adversarial objective
is related to Rumi-GAN (2020). Our research question is its use as an AG reference.

The EM interpretation requires endpoint density posteriors. Applying these
functions to feature/noisy-input classifiers does not establish endpoint EM.
No discriminator or weights are evaluated during guided sampling.
"""

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


SOURCE_ORDER = ("real", "strong", "weak_snapshot")


def reference_fraction(guidance_anchor: float) -> float:
    """First-order calibration kappa/(1-kappa) = guidance_anchor.

    This is not an exact identity for finite-error constant-coefficient AG.
    """
    if not math.isfinite(guidance_anchor) or guidance_anchor <= 0:
        raise ValueError("guidance_anchor must be finite and positive")
    return guidance_anchor / (1.0 + guidance_anchor)


def _check_fraction(kappa: float) -> None:
    if not math.isfinite(kappa) or not 0 < kappa < 1:
        raise ValueError("kappa must be strictly between zero and one")


def source_log_densities(
    logits: torch.Tensor,
    source_priors: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> torch.Tensor:
    """Log densities up to a common x-dependent term, in P/G/Q order.

    Source priors are the effective class-conditional sampling priors used by
    the classifier. Labels must be conditioned on, not confused with sources.
    """
    if logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("expected an N by 3 tensor in real/strong/weak order")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    value = logits if logits.dtype == torch.float64 else logits.float()
    priors = torch.as_tensor(source_priors, dtype=value.dtype, device=value.device)
    if priors.shape != (3,) or not torch.isfinite(priors).all() or (priors <= 0).any():
        raise ValueError("three positive, finite source priors are required")
    if not torch.isclose(priors.sum(), priors.new_tensor(1.0), atol=1e-6, rtol=0):
        raise ValueError("source priors must sum to one")
    return value - priors.log()


@torch.no_grad()
def endpoint_responsibility(
    logits: torch.Tensor,
    kappa: float,
    source_priors: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> torch.Tensor:
    """tau(x) = kappa q(x) / ((1-kappa) p(x) + kappa q(x)).

    Evaluate on STRONG endpoints. Raw weights are in [0, 1]; full-bank
    normalized weights need not be bounded by one. Freeze the weak snapshot
    and classifier throughout each fitting round. Never recompute tau from a
    different noisy input at every FM step and still call this endpoint EM.
    """
    _check_fraction(kappa)
    log_density = source_log_densities(logits, source_priors)
    odds = log_density[:, 2] - log_density[:, 0]
    return torch.sigmoid(odds + math.log(kappa) - math.log1p(-kappa))


def mixture_discriminator_logit(
    logits: torch.Tensor,
    kappa: float,
    source_priors: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> torch.Tensor:
    """log(g / M), so sigmoid(output) is the G-vs-M GAN discriminator.

    Unlike tau, this diagnostic uses the strong source logit too. It is a
    density/value diagnostic, not a prediction of guidance quality.
    """
    _check_fraction(kappa)
    density = source_log_densities(logits, source_priors)
    mixture = torch.logaddexp(
        math.log1p(-kappa) + density[:, 0], math.log(kappa) + density[:, 2]
    )
    return density[:, 1] - mixture


@torch.no_grad()
def noisy_observation_log_weights(
    logits: torch.Tensor,
    kappa: float,
    source_priors: Sequence[float] = (1 / 3, 1 / 3, 1 / 3),
) -> torch.Tensor:
    """Log Monte Carlo weights for the common-noising observation EM variant.

    logits has shape [endpoints, channel_draws, 3]. Each endpoint must come
    from Q_snapshot; each draw samples t~nu and Z~K_t(.|endpoint). Return
    log mean_draws[g_t(Z)/((1-kappa)p_t(Z)+kappa*q_t(Z))]. These weights
    reweight WEAK endpoints. This is not endpoint tau evaluated at noisy Z.
    The ratio has no general upper bound. Work in log space, without clipping.
    """
    if logits.ndim != 3 or logits.shape[2] != 3 or logits.shape[1] == 0:
        raise ValueError("expected endpoint by channel_draw by 3 source logits")
    n, draws, _ = logits.shape
    ratios = mixture_discriminator_logit(logits.reshape(-1, 3), kappa, source_priors)
    return torch.logsumexp(ratios.reshape(n, draws), dim=1) - math.log(draws)


def normalize_endpoint_log_weights(log_weights: np.ndarray, labels: np.ndarray):
    """Full-bank class normalization for the unbounded noisy-observation weights."""
    values, labels = np.asarray(log_weights, dtype=np.float64), np.asarray(labels)
    if values.ndim != 1 or values.shape != labels.shape or not len(values):
        raise ValueError("nonempty log weights and labels must have equal vector shapes")
    if not np.isfinite(values).all() or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("finite log weights and integer labels are required")
    result, normalizers = np.empty_like(values), {}
    for label in np.unique(labels):
        mask = labels == label
        selected = values[mask]
        maximum = float(selected.max())
        log_mean = maximum + float(np.log(np.exp(selected - maximum).mean()))
        normalizers[int(label)] = log_mean
        result[mask] = np.exp(selected - log_mean)
    return result, normalizers


def mixture_gan_discriminator_loss(
    strong_logits: torch.Tensor,
    real_logits: torch.Tensor,
    weak_logits: torch.Tensor,
    kappa: float,
) -> torch.Tensor:
    """Binary discriminator logits, with G on one side and the P/Q mixture on the other.

    Independent source means preserve the specified mixture even if batch
    sizes differ. This is a discriminator loss, not a head FM objective.
    """
    _check_fraction(kappa)
    for value in (strong_logits, real_logits, weak_logits):
        if value.numel() == 0 or not torch.isfinite(value).all():
            raise ValueError("each source needs nonempty finite logits")
    return (
        F.softplus(-strong_logits).mean()
        + (1 - kappa) * F.softplus(real_logits).mean()
        + kappa * F.softplus(weak_logits).mean()
    )


def normalize_endpoint_weights(raw: np.ndarray, labels: np.ndarray):
    """Normalize on the COMPLETE training endpoint bank separately per label.

    Returns weights and per-class normalizers. Validation or minibatch
    statistics must not enter these normalizers.
    """
    raw = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(labels)
    if raw.ndim != 1 or raw.shape != labels.shape or len(raw) == 0:
        raise ValueError("raw weights and labels must be nonempty vectors of equal length")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be integers")
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        raise ValueError("raw responsibilities must be finite and in [0, 1]")
    result = np.empty_like(raw)
    normalizers = {}
    for label in np.unique(labels):
        mask = labels == label
        mean = float(raw[mask].mean())
        if mean <= 0:
            raise ValueError(f"zero responsibility mass for label {label}; cannot form target")
        normalizers[int(label)] = mean
        result[mask] = raw[mask] / mean
    return result, normalizers


def weighted_endpoint_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """A detached positive-weight FM/denoising loss on strong endpoints.

    Apply native model/time scaling before calling (JiT uses the existing
    clean-prediction denominator). The weights come from endpoint IDs, while
    time and noise are resampled every training step.
    """
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have matching batch/event dimensions")
    if weights.shape != (prediction.shape[0],):
        raise ValueError("one full-bank-normalized weight is required per endpoint")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    error = (prediction.float() - target.detach().float()).square().flatten(1).mean(1)
    return (weights.detach().to(error) * error).mean()
