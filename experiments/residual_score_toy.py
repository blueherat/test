"""Analytic near-equilibrium score-difference estimation utilities.

The real and generated distributions share Gaussian components and differ only
in their mixture weights.  Gaussian noising therefore preserves a closed-form
mixture density and score at every noise scale.  This makes it possible to
audit score-difference estimators against ground truth instead of relying on a
generation metric as a proxy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Domain = Literal["real", "fake"]


def _as_batch_sigma(
    sigma: float | torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype)
    if value.ndim == 0:
        value = value.expand(len(reference))
    if value.shape != (len(reference),):
        raise ValueError("sigma must be scalar or have shape [batch]")
    if torch.any(value < 0):
        raise ValueError("sigma must be non-negative")
    return value


@dataclass(frozen=True)
class NearEquilibriumGaussianMixture:
    """A shared-component pair ``p`` and ``q_epsilon`` with analytic scores."""

    means: torch.Tensor
    component_std: float
    perturbation: torch.Tensor

    def __post_init__(self) -> None:
        if self.means.ndim != 2:
            raise ValueError("means must have shape [components, dimension]")
        if len(self.means) < 2:
            raise ValueError("at least two mixture components are required")
        if self.component_std <= 0:
            raise ValueError("component_std must be positive")
        if self.perturbation.shape != (len(self.means),):
            raise ValueError("perturbation must have one entry per component")
        if abs(float(self.perturbation.mean())) > 1e-7:
            raise ValueError("perturbation must have zero mean")

    @classmethod
    def ring(
        cls,
        *,
        components: int = 12,
        radius: float = 2.5,
        component_std: float = 0.22,
        perturbation_amplitude: float = 0.85,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> "NearEquilibriumGaussianMixture":
        if components < 4:
            raise ValueError("components must be at least four")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if not 0 < perturbation_amplitude < 1:
            raise ValueError("perturbation_amplitude must lie in (0,1)")
        angles = torch.arange(components, dtype=dtype, device=device)
        angles = angles * (2.0 * math.pi / components)
        means = radius * torch.stack((angles.cos(), angles.sin()), dim=1)
        # Combining two harmonics avoids a trivial even/odd classification
        # problem while retaining an exactly zero-sum weight perturbation.
        perturbation = (
            0.7 * torch.sin(angles)
            + 0.3 * torch.cos(2.0 * angles + 0.37)
        )
        perturbation = perturbation - perturbation.mean()
        perturbation = perturbation_amplitude * perturbation / perturbation.abs().max()
        return cls(means, component_std, perturbation)

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def weights(self, epsilon: float, domain: Domain) -> torch.Tensor:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must lie in [0,1]")
        base = torch.full_like(self.perturbation, 1.0 / len(self.perturbation))
        if domain == "real":
            return base
        if domain != "fake":
            raise ValueError(f"unknown domain: {domain}")
        weights = base * (1.0 + float(epsilon) * self.perturbation)
        if torch.any(weights <= 0):
            raise ValueError("epsilon produced non-positive mixture weights")
        return weights / weights.sum()

    def sample_clean(
        self,
        count: int,
        *,
        epsilon: float,
        domain: Domain,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive")
        weights = self.weights(epsilon, domain).cpu()
        labels = torch.multinomial(
            weights,
            count,
            replacement=True,
            generator=generator,
        ).to(self.means.device)
        noise = torch.randn(
            count,
            self.dimension,
            generator=generator,
            dtype=self.means.dtype,
            device="cpu",
        ).to(self.means.device)
        return self.means[labels] + float(self.component_std) * noise

    def log_prob_and_score(
        self,
        states: torch.Tensor,
        *,
        sigma: float | torch.Tensor,
        epsilon: float,
        domain: Domain,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim != 2 or states.shape[1] != self.dimension:
            raise ValueError("states must have shape [batch, dimension]")
        sigma_batch = _as_batch_sigma(sigma, states)
        variance = self.component_std**2 + sigma_batch.square()
        residual = states[:, None, :] - self.means.to(states)[None, :, :]
        weights = self.weights(epsilon, domain).to(states)
        log_components = (
            weights.log()[None, :]
            - 0.5 * self.dimension * torch.log(2.0 * math.pi * variance)[:, None]
            - 0.5 * residual.square().sum(dim=-1) / variance[:, None]
        )
        log_probability = torch.logsumexp(log_components, dim=1)
        responsibilities = torch.softmax(log_components, dim=1)
        component_scores = -residual / variance[:, None, None]
        score = (responsibilities[..., None] * component_scores).sum(dim=1)
        return log_probability, score

    def residual_score(
        self,
        states: torch.Tensor,
        *,
        sigma: float | torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """Return ``score_real - score_fake`` at a common noised state."""

        _, score_real = self.log_prob_and_score(
            states, sigma=sigma, epsilon=epsilon, domain="real"
        )
        _, score_fake = self.log_prob_and_score(
            states, sigma=sigma, epsilon=epsilon, domain="fake"
        )
        return score_real - score_fake

    def bayes_real_logit(
        self,
        states: torch.Tensor,
        *,
        sigma: float | torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        log_real, _ = self.log_prob_and_score(
            states, sigma=sigma, epsilon=epsilon, domain="real"
        )
        log_fake, _ = self.log_prob_and_score(
            states, sigma=sigma, epsilon=epsilon, domain="fake"
        )
        return log_real - log_fake


def sample_log_uniform_sigma(
    count: int,
    *,
    sigma_min: float,
    sigma_max: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0 or sigma_min <= 0 or sigma_max < sigma_min:
        raise ValueError("invalid log-uniform sigma configuration")
    uniform = torch.rand(count, generator=generator, dtype=dtype, device="cpu")
    log_min = math.log(sigma_min)
    log_max = math.log(sigma_max)
    return (log_min + uniform * (log_max - log_min)).exp().to(device)


class NoiseEmbedding(nn.Module):
    def __init__(self, frequencies: int = 6) -> None:
        super().__init__()
        if frequencies <= 0:
            raise ValueError("frequencies must be positive")
        values = 2.0 ** torch.arange(frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", values, persistent=False)

    @property
    def output_dim(self) -> int:
        return 2 + 2 * len(self.frequencies)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim != 1:
            raise ValueError("sigma must have shape [batch]")
        coordinate = torch.log1p(sigma)
        phase = coordinate[:, None] * self.frequencies[None, :].to(sigma)
        return torch.cat(
            (
                sigma[:, None],
                coordinate[:, None],
                phase.sin(),
                phase.cos(),
            ),
            dim=1,
        )


def _mlp(input_dim: int, output_dim: int, hidden_dim: int, depth: int) -> nn.Sequential:
    if hidden_dim <= 0 or depth <= 0:
        raise ValueError("hidden_dim and depth must be positive")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend((nn.Linear(current, hidden_dim), nn.SiLU()))
        current = hidden_dim
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class RatioEstimator(nn.Module):
    """Time-conditioned real-vs-fake log-density-ratio estimator."""

    def __init__(
        self,
        dimension: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        frequencies: int = 6,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.embedding = NoiseEmbedding(frequencies)
        self.network = _mlp(
            dimension + self.embedding.output_dim,
            1,
            hidden_dim,
            depth,
        )

    def forward(self, states: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if states.ndim != 2 or states.shape[1] != self.dimension:
            raise ValueError("states have the wrong shape")
        sigma = _as_batch_sigma(sigma, states)
        features = torch.cat((states, self.embedding(sigma)), dim=1)
        return self.network(features)[:, 0]


class NoiseEstimator(nn.Module):
    """Noise predictor used for one distribution in a DSM baseline."""

    def __init__(
        self,
        dimension: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        frequencies: int = 6,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.embedding = NoiseEmbedding(frequencies)
        self.network = _mlp(
            dimension + self.embedding.output_dim,
            dimension,
            hidden_dim,
            depth,
        )

    def forward(self, states: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = _as_batch_sigma(sigma, states)
        return self.network(torch.cat((states, self.embedding(sigma)), dim=1))


class SharedDomainNoiseEstimator(nn.Module):
    """One denoiser shared by real and fake domains.

    ``domain=1`` denotes real data and ``domain=0`` denotes generated data.
    """

    def __init__(
        self,
        dimension: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        frequencies: int = 6,
        domain_dim: int = 16,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.embedding = NoiseEmbedding(frequencies)
        self.domain_embedding = nn.Embedding(2, domain_dim)
        self.network = _mlp(
            dimension + self.embedding.output_dim + domain_dim,
            dimension,
            hidden_dim,
            depth,
        )

    def forward(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        domain: torch.Tensor,
    ) -> torch.Tensor:
        sigma = _as_batch_sigma(sigma, states)
        if domain.shape != (len(states),):
            raise ValueError("domain must have shape [batch]")
        features = torch.cat(
            (
                states,
                self.embedding(sigma),
                self.domain_embedding(domain.long()),
            ),
            dim=1,
        )
        return self.network(features)


class FactorizedDomainNoiseEstimator(nn.Module):
    """Domain denoiser with an explicit common/residual decomposition.

    For real-domain sign ``+1`` and fake-domain sign ``-1``, the prediction is
    ``common + sign * residual``.  At the population MSE optimum these outputs
    equal the average and half-difference of the two conditional noise means.
    Consequently the residual score depends only on the residual head; common
    denoising error cancels algebraically instead of only statistically.
    """

    def __init__(
        self,
        dimension: int,
        *,
        hidden_dim: int = 128,
        depth: int = 3,
        frequencies: int = 6,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.embedding = NoiseEmbedding(frequencies)
        self.network = _mlp(
            dimension + self.embedding.output_dim,
            2 * dimension,
            hidden_dim,
            depth,
        )

    def components(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = _as_batch_sigma(sigma, states)
        output = self.network(torch.cat((states, self.embedding(sigma)), dim=1))
        return output.chunk(2, dim=1)

    def forward(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        domain: torch.Tensor,
    ) -> torch.Tensor:
        if domain.shape != (len(states),):
            raise ValueError("domain must have shape [batch]")
        common, residual = self.components(states, sigma)
        sign = domain.to(states).mul(2.0).sub(1.0)
        return common + sign[:, None] * residual


def ratio_score_difference(
    model: RatioEstimator,
    states: torch.Tensor,
    sigma: torch.Tensor,
    *,
    create_graph: bool = False,
) -> torch.Tensor:
    """Differentiate the real-class logit to estimate ``score_p-score_q``."""

    if not states.requires_grad:
        states = states.detach().requires_grad_(True)
    logits = model(states, sigma)
    return torch.autograd.grad(
        logits.sum(),
        states,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]


def separate_dsm_score_difference(
    real_model: NoiseEstimator,
    fake_model: NoiseEstimator,
    states: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    sigma = _as_batch_sigma(sigma, states)
    if torch.any(sigma <= 0):
        raise ValueError("DSM score conversion requires positive sigma")
    predicted_real_noise = real_model(states, sigma)
    predicted_fake_noise = fake_model(states, sigma)
    return (predicted_fake_noise - predicted_real_noise) / sigma[:, None]


def shared_dsm_score_difference(
    model: SharedDomainNoiseEstimator,
    states: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    sigma = _as_batch_sigma(sigma, states)
    if torch.any(sigma <= 0):
        raise ValueError("DSM score conversion requires positive sigma")
    real_domain = torch.ones(len(states), dtype=torch.long, device=states.device)
    fake_domain = torch.zeros_like(real_domain)
    predicted_real_noise = model(states, sigma, real_domain)
    predicted_fake_noise = model(states, sigma, fake_domain)
    return (predicted_fake_noise - predicted_real_noise) / sigma[:, None]


def factorized_dsm_score_difference(
    model: FactorizedDomainNoiseEstimator,
    states: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    sigma = _as_batch_sigma(sigma, states)
    if torch.any(sigma <= 0):
        raise ValueError("DSM score conversion requires positive sigma")
    _, residual_noise = model.components(states, sigma)
    return -2.0 * residual_noise / sigma[:, None]


def field_metrics(estimate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimate and target must have equal matrix shapes")
    tiny = torch.finfo(estimate.dtype).tiny
    error_energy = (estimate - target).square().sum(dim=1).mean()
    estimate_energy = estimate.square().sum(dim=1).mean()
    target_energy = target.square().sum(dim=1).mean()
    inner = (estimate * target).sum(dim=1).mean()
    cosine = inner / (estimate_energy * target_energy).clamp_min(tiny).sqrt()
    positive_fraction = ((estimate * target).sum(dim=1) > 0).float().mean()
    return {
        "relative_l2": float(
            (error_energy / target_energy.clamp_min(tiny)).sqrt().detach()
        ),
        "global_cosine": float(cosine.detach()),
        "norm_ratio": float(
            (estimate_energy / target_energy.clamp_min(tiny)).sqrt().detach()
        ),
        "positive_alignment_fraction": float(positive_fraction.detach()),
        "estimate_rms": float(estimate_energy.sqrt().detach()),
        "target_rms": float(target_energy.sqrt().detach()),
        "inner_product": float(inner.detach()),
    }


def pairwise_field_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must have equal matrix shapes")
    tiny = torch.finfo(left.dtype).tiny
    left_energy = left.square().sum(dim=1).mean()
    right_energy = right.square().sum(dim=1).mean()
    inner = (left * right).sum(dim=1).mean()
    return {
        "pairwise_cosine": float(
            (
                inner / (left_energy * right_energy).clamp_min(tiny).sqrt()
            ).detach()
        ),
        "pairwise_relative_l2": float(
            (
                (left - right).square().sum(dim=1).mean()
                / (0.5 * (left_energy + right_energy)).clamp_min(tiny)
            )
            .sqrt()
            .detach()
        ),
        "left_rms": float(left_energy.sqrt().detach()),
        "right_rms": float(right_energy.sqrt().detach()),
    }


def classifier_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have equal shapes")
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    accuracy = ((logits >= 0) == (labels >= 0.5)).float().mean()
    real = logits[labels >= 0.5]
    fake = logits[labels < 0.5]
    # Pairwise Mann-Whitney interpretation of AUC; toy heldout sets are small
    # enough for an exact computation without an sklearn dependency.
    auc = (real[:, None] > fake[None, :]).float().mean()
    auc = auc + 0.5 * (real[:, None] == fake[None, :]).float().mean()
    return {
        "classifier_bce": float(loss.detach()),
        "classifier_accuracy": float(accuracy.detach()),
        "classifier_auc": float(auc.detach()),
        "real_logit_mean": float(real.mean().detach()),
        "fake_logit_mean": float(fake.mean().detach()),
    }


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def field_jacobian(field: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Return per-sample Jacobians for a sample-independent batched field."""

    if field.shape != states.shape or not states.requires_grad:
        raise ValueError("field/states must match and states must require gradients")
    rows = []
    for output_index in range(field.shape[1]):
        rows.append(
            torch.autograd.grad(
                field[:, output_index].sum(),
                states,
                retain_graph=output_index + 1 < field.shape[1],
                create_graph=False,
            )[0]
        )
    return torch.stack(rows, dim=1)


def pushforward_kl(
    mixture: NearEquilibriumGaussianMixture,
    states: torch.Tensor,
    field: torch.Tensor,
    *,
    sigma: float,
    epsilon: float,
    step_size: float,
) -> dict[str, float]:
    """Estimate KL after the finite map ``y=x+step_size*field(x)``.

    For a locally orientation-preserving map, change of variables gives
    ``log q'(T(x)) = log q(x) - log det(I + eta J_field(x))``.  In two
    dimensions the Jacobian is computed exactly with autograd.
    """

    if step_size <= 0:
        raise ValueError("step_size must be positive")
    jacobian = field_jacobian(field, states)
    identity = torch.eye(
        states.shape[1], device=states.device, dtype=states.dtype
    )[None]
    determinant = torch.linalg.det(identity + float(step_size) * jacobian)
    valid = determinant > 0
    transported = states + float(step_size) * field.detach()
    log_q, _ = mixture.log_prob_and_score(
        states.detach(), sigma=sigma, epsilon=epsilon, domain="fake"
    )
    log_p_before, _ = mixture.log_prob_and_score(
        states.detach(), sigma=sigma, epsilon=epsilon, domain="real"
    )
    log_p_after, _ = mixture.log_prob_and_score(
        transported, sigma=sigma, epsilon=epsilon, domain="real"
    )
    baseline = log_q - log_p_before
    if valid.any():
        after = log_q[valid] - determinant[valid].log() - log_p_after[valid]
        kl_after = float(after.mean().detach())
    else:
        kl_after = float("nan")
    kl_before = float(baseline.mean().detach())
    return {
        "kl_before": kl_before,
        "kl_after": kl_after,
        "kl_change": kl_after - kl_before,
        "positive_jacobian_fraction": float(valid.float().mean()),
    }


__all__ = [
    "NearEquilibriumGaussianMixture",
    "FactorizedDomainNoiseEstimator",
    "NoiseEstimator",
    "RatioEstimator",
    "SharedDomainNoiseEstimator",
    "classifier_metrics",
    "field_jacobian",
    "field_metrics",
    "factorized_dsm_score_difference",
    "pairwise_field_metrics",
    "parameter_count",
    "pushforward_kl",
    "ratio_score_difference",
    "sample_log_uniform_sigma",
    "separate_dsm_score_difference",
    "shared_dsm_score_difference",
]
