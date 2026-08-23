#!/usr/bin/env python3
"""Compare terminal-distribution residual estimators against analytic truth.

The experiment deliberately separates two questions that image-space FID
cannot separate:

1. Does an estimator recover the residual score ``score_real-score_fake``?
2. Does its finite update actually move the generated distribution toward the
   real distribution?

All learned estimators see the same finite clean sample banks.  Fresh Gaussian
noise is generated during training, so noisy classifiers and denoisers receive
diffusive augmentation without receiving analytic density information.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from experiments.residual_score_toy import (
    NearEquilibriumGaussianMixture,
    FactorizedDomainNoiseEstimator,
    NoiseEstimator,
    RatioEstimator,
    SharedDomainNoiseEstimator,
    classifier_metrics,
    factorized_dsm_score_difference,
    field_metrics,
    field_jacobian,
    pairwise_field_metrics,
    parameter_count,
    ratio_score_difference,
    sample_log_uniform_sigma,
    separate_dsm_score_difference,
    shared_dsm_score_difference,
)


RATIO_METHODS = {"zero_ratio", "ratio", "sobolev_ratio"}
DSM_METHODS = {
    "separate_dsm",
    "shared_dsm",
    "factorized_dsm",
    "factorized_dsm_coupled",
}
ALL_METHODS = RATIO_METHODS | DSM_METHODS


@dataclass(frozen=True)
class ExperimentConfig:
    epsilons: tuple[float, ...]
    train_samples: tuple[int, ...]
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    sobolev_lambdas: tuple[float, ...]
    train_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    depth: int
    frequencies: int
    domain_dim: int
    train_sigma_min: float
    train_sigma_max: float
    eval_sigmas: tuple[float, ...]
    eval_samples: int
    pushforward_step_size: float
    quadrature_grid_size: int
    quadrature_batch_size: int
    log_every: int
    components: int
    radius: float
    component_std: float
    perturbation_amplitude: float
    data_seed: int
    eval_seed: int
    device: str


@dataclass
class EstimatorBundle:
    label: str
    kind: str
    modules: dict[str, torch.nn.Module]
    history: list[dict[str, float | int]]
    elapsed_seconds: float
    parameter_count: int
    sobolev_lambda: float | None = None

    def train(self, mode: bool = True) -> None:
        for module in self.modules.values():
            module.train(mode)

    def state_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "kind": self.kind,
            "sobolev_lambda": self.sobolev_lambda,
            "modules": {
                name: module.state_dict() for name, module in self.modules.items()
            },
        }


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def sample_indices(
    size: int,
    count: int,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return torch.randint(size, (count,), generator=generator).to(device)


def sample_noise_like(
    reference: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        generator=generator,
        dtype=reference.dtype,
        device="cpu",
    ).to(reference.device)


def build_clean_banks(
    mixture: NearEquilibriumGaussianMixture,
    *,
    count: int,
    epsilon: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = cpu_generator(seed)
    real = mixture.sample_clean(
        count,
        epsilon=epsilon,
        domain="real",
        generator=generator,
    )
    fake = mixture.sample_clean(
        count,
        epsilon=epsilon,
        domain="fake",
        generator=generator,
    )
    return real, fake


def draw_sigmas(
    count: int,
    *,
    kind: str,
    config: ExperimentConfig,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kind == "zero_ratio":
        return torch.zeros(count, device=device, dtype=dtype)
    return sample_log_uniform_sigma(
        count,
        sigma_min=config.train_sigma_min,
        sigma_max=config.train_sigma_max,
        generator=generator,
        device=device,
        dtype=dtype,
    )


def train_ratio_estimator(
    *,
    kind: str,
    sobolev_lambda: float,
    real_bank: torch.Tensor,
    fake_bank: torch.Tensor,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> EstimatorBundle:
    if kind not in RATIO_METHODS:
        raise ValueError(f"not a ratio method: {kind}")
    seed_everything(seed)
    model = RatioEstimator(
        real_bank.shape[1],
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        frequencies=config.frequencies,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = cpu_generator(seed + 17)
    per_domain = config.batch_size // 2
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    model.train()
    for step in range(1, config.train_steps + 1):
        real = real_bank[
            sample_indices(
                len(real_bank), per_domain, generator=generator, device=device
            )
        ]
        fake = fake_bank[
            sample_indices(
                len(fake_bank), per_domain, generator=generator, device=device
            )
        ]
        sigma_half = draw_sigmas(
            per_domain,
            kind=kind,
            config=config,
            generator=generator,
            device=device,
            dtype=real.dtype,
        )
        real_noisy = real + sigma_half[:, None] * sample_noise_like(
            real, generator=generator
        )
        fake_noisy = fake + sigma_half[:, None] * sample_noise_like(
            fake, generator=generator
        )
        states = torch.cat((real_noisy, fake_noisy), dim=0)
        sigmas = torch.cat((sigma_half, sigma_half), dim=0)
        labels = torch.cat(
            (
                torch.ones(per_domain, device=device, dtype=states.dtype),
                torch.zeros(per_domain, device=device, dtype=states.dtype),
            )
        )
        use_sobolev = kind == "sobolev_ratio" and sobolev_lambda > 0
        if use_sobolev:
            states.requires_grad_(True)
        logits = model(states, sigmas)
        bce = F.binary_cross_entropy_with_logits(logits, labels)
        if use_sobolev:
            input_gradient = torch.autograd.grad(
                logits.sum(), states, create_graph=True
            )[0]
            sobolev = input_gradient.square().sum(dim=1).mean()
        else:
            sobolev = torch.zeros((), device=device, dtype=states.dtype)
        loss = bce + float(sobolev_lambda) * sobolev
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "bce": float(bce.detach()),
                    "sobolev_penalty": float(sobolev.detach()),
                }
            )
    elapsed = time.perf_counter() - start
    label = kind
    if kind == "sobolev_ratio":
        label = f"sobolev_ratio_lam{sobolev_lambda:g}"
    return EstimatorBundle(
        label=label,
        kind=kind,
        modules={"ratio": model},
        history=history,
        elapsed_seconds=elapsed,
        parameter_count=parameter_count(model),
        sobolev_lambda=sobolev_lambda if kind == "sobolev_ratio" else None,
    )


def train_dsm_estimator(
    *,
    kind: str,
    real_bank: torch.Tensor,
    fake_bank: torch.Tensor,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> EstimatorBundle:
    if kind not in DSM_METHODS:
        raise ValueError(f"not a DSM method: {kind}")
    seed_everything(seed)
    dimension = real_bank.shape[1]
    if kind == "separate_dsm":
        modules: dict[str, torch.nn.Module] = {
            "real": NoiseEstimator(
                dimension,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                frequencies=config.frequencies,
            ).to(device),
            "fake": NoiseEstimator(
                dimension,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                frequencies=config.frequencies,
            ).to(device),
        }
    elif kind == "shared_dsm":
        modules = {
            "shared": SharedDomainNoiseEstimator(
                dimension,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                frequencies=config.frequencies,
                domain_dim=config.domain_dim,
            ).to(device)
        }
    else:
        modules = {
            "factorized": FactorizedDomainNoiseEstimator(
                dimension,
                hidden_dim=config.hidden_dim,
                depth=config.depth,
                frequencies=config.frequencies,
            ).to(device)
        }
    parameters = [
        parameter for module in modules.values() for parameter in module.parameters()
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = cpu_generator(seed + 17)
    per_domain = config.batch_size // 2
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()
    for module in modules.values():
        module.train()
    for step in range(1, config.train_steps + 1):
        real = real_bank[
            sample_indices(
                len(real_bank), per_domain, generator=generator, device=device
            )
        ]
        fake = fake_bank[
            sample_indices(
                len(fake_bank), per_domain, generator=generator, device=device
            )
        ]
        sigma = draw_sigmas(
            per_domain,
            kind=kind,
            config=config,
            generator=generator,
            device=device,
            dtype=real.dtype,
        )
        real_noise = sample_noise_like(real, generator=generator)
        if kind == "factorized_dsm_coupled":
            fake_noise = real_noise.clone()
        else:
            fake_noise = sample_noise_like(fake, generator=generator)
        real_noisy = real + sigma[:, None] * real_noise
        fake_noisy = fake + sigma[:, None] * fake_noise
        if kind == "separate_dsm":
            real_prediction = modules["real"](real_noisy, sigma)
            fake_prediction = modules["fake"](fake_noisy, sigma)
        elif kind == "shared_dsm":
            states = torch.cat((real_noisy, fake_noisy), dim=0)
            sigmas = torch.cat((sigma, sigma), dim=0)
            domains = torch.cat(
                (
                    torch.ones(per_domain, device=device, dtype=torch.long),
                    torch.zeros(per_domain, device=device, dtype=torch.long),
                )
            )
            predictions = modules["shared"](states, sigmas, domains)
            real_prediction, fake_prediction = predictions.chunk(2, dim=0)
        else:
            states = torch.cat((real_noisy, fake_noisy), dim=0)
            sigmas = torch.cat((sigma, sigma), dim=0)
            domains = torch.cat(
                (
                    torch.ones(per_domain, device=device, dtype=torch.long),
                    torch.zeros(per_domain, device=device, dtype=torch.long),
                )
            )
            predictions = modules["factorized"](states, sigmas, domains)
            real_prediction, fake_prediction = predictions.chunk(2, dim=0)
        real_loss = F.mse_loss(real_prediction, real_noise)
        fake_loss = F.mse_loss(fake_prediction, fake_noise)
        loss = 0.5 * (real_loss + fake_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % config.log_every == 0 or step == config.train_steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "real_noise_mse": float(real_loss.detach()),
                    "fake_noise_mse": float(fake_loss.detach()),
                }
            )
    elapsed = time.perf_counter() - start
    return EstimatorBundle(
        label=kind,
        kind=kind,
        modules=modules,
        history=history,
        elapsed_seconds=elapsed,
        parameter_count=sum(parameter_count(module) for module in modules.values()),
    )


def estimate_field(
    bundle: EstimatorBundle,
    states: torch.Tensor,
    sigma: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    if bundle.kind in RATIO_METHODS:
        return ratio_score_difference(
            bundle.modules["ratio"],
            states,
            sigma,
            create_graph=create_graph,
        )
    if bundle.kind == "separate_dsm":
        return separate_dsm_score_difference(
            bundle.modules["real"],
            bundle.modules["fake"],
            states,
            sigma,
        )
    if bundle.kind == "shared_dsm":
        return shared_dsm_score_difference(bundle.modules["shared"], states, sigma)
    if bundle.kind in {"factorized_dsm", "factorized_dsm_coupled"}:
        return factorized_dsm_score_difference(
            bundle.modules["factorized"], states, sigma
        )
    raise ValueError(f"unknown bundle kind: {bundle.kind}")


def estimate_dsm_component_scores(
    bundle: EstimatorBundle,
    states: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the learned real and fake scores before taking their difference."""

    sigma = torch.as_tensor(sigma, device=states.device, dtype=states.dtype)
    if sigma.ndim == 0:
        sigma = sigma.expand(len(states))
    if torch.any(sigma <= 0):
        raise ValueError("DSM component scores require positive sigma")
    if bundle.kind == "separate_dsm":
        real_noise = bundle.modules["real"](states, sigma)
        fake_noise = bundle.modules["fake"](states, sigma)
    elif bundle.kind == "shared_dsm":
        real_domain = torch.ones(len(states), dtype=torch.long, device=states.device)
        fake_domain = torch.zeros_like(real_domain)
        real_noise = bundle.modules["shared"](states, sigma, real_domain)
        fake_noise = bundle.modules["shared"](states, sigma, fake_domain)
    elif bundle.kind in {"factorized_dsm", "factorized_dsm_coupled"}:
        real_domain = torch.ones(len(states), dtype=torch.long, device=states.device)
        fake_domain = torch.zeros_like(real_domain)
        real_noise = bundle.modules["factorized"](states, sigma, real_domain)
        fake_noise = bundle.modules["factorized"](states, sigma, fake_domain)
    else:
        raise ValueError(f"not a DSM bundle: {bundle.kind}")
    return -real_noise / sigma[:, None], -fake_noise / sigma[:, None]


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def quadrature_distribution_response(
    mixture: NearEquilibriumGaussianMixture,
    field_function: Callable[[torch.Tensor], torch.Tensor],
    *,
    sigma: float,
    epsilon: float,
    step_size: float,
    grid_size: int,
    batch_size: int,
) -> dict[str, float]:
    """Deterministically integrate local and finite distribution responses.

    The two-dimensional grid removes the Monte Carlo sign noise that otherwise
    dominates near equilibrium.  The finite-map KL uses change of variables
    and is reported only when the map has positive Jacobian over essentially
    all probability mass covered by the grid.
    """

    if mixture.dimension != 2:
        raise ValueError("quadrature response currently requires a 2D mixture")
    if grid_size < 16 or batch_size <= 0:
        raise ValueError("invalid quadrature configuration")
    component_scale = math.sqrt(mixture.component_std**2 + sigma**2)
    radius = float(mixture.means.norm(dim=1).max())
    extent = radius + 6.0 * component_scale
    axis = torch.linspace(
        -extent,
        extent,
        grid_size,
        device=mixture.means.device,
        dtype=mixture.means.dtype,
    )
    first, second = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((first.flatten(), second.flatten()), dim=1)
    cell_area = float((axis[1] - axis[0]).square())
    mass = torch.zeros((), dtype=torch.float64, device=grid.device)
    valid_mass = torch.zeros_like(mass)
    baseline_integral = torch.zeros_like(mass)
    after_integral = torch.zeros_like(mass)
    derivative_integral = torch.zeros_like(mass)
    oracle_derivative_integral = torch.zeros_like(mass)
    for start in range(0, len(grid), batch_size):
        states = grid[start : start + batch_size].detach().requires_grad_(True)
        log_q, score_q = mixture.log_prob_and_score(
            states, sigma=sigma, epsilon=epsilon, domain="fake"
        )
        log_p, score_p = mixture.log_prob_and_score(
            states, sigma=sigma, epsilon=epsilon, domain="real"
        )
        field = field_function(states)
        jacobian = field_jacobian(field, states)
        identity = torch.eye(
            mixture.dimension, device=states.device, dtype=states.dtype
        )[None]
        determinant = torch.linalg.det(identity + step_size * jacobian)
        valid = determinant > 0
        transported = states.detach() + step_size * field.detach()
        log_p_after, _ = mixture.log_prob_and_score(
            transported, sigma=sigma, epsilon=epsilon, domain="real"
        )
        weight = log_q.detach().double().exp() * cell_area
        mass = mass + weight.sum()
        baseline_integral = baseline_integral + (
            weight * (log_q.detach() - log_p.detach()).double()
        ).sum()
        derivative_integral = derivative_integral + (
            weight
            * (
                field.detach() * (score_q.detach() - score_p.detach())
            ).sum(dim=1).double()
        ).sum()
        oracle_derivative_integral = oracle_derivative_integral + (
            -weight * (score_p.detach() - score_q.detach()).square().sum(dim=1).double()
        ).sum()
        if valid.any():
            valid_weight = weight[valid]
            valid_mass = valid_mass + valid_weight.sum()
            after_integral = after_integral + (
                valid_weight
                * (
                    log_q.detach()[valid]
                    - determinant.detach()[valid].log()
                    - log_p_after.detach()[valid]
                ).double()
            ).sum()
    normalizer = mass.clamp_min(torch.finfo(torch.float64).tiny)
    valid_fraction = valid_mass / normalizer
    kl_before = baseline_integral / normalizer
    if float(valid_fraction) > 0.999999:
        kl_after = after_integral / valid_mass
        kl_change = kl_after - kl_before
    else:
        kl_after = torch.full_like(kl_before, float("nan"))
        kl_change = torch.full_like(kl_before, float("nan"))
    directional_derivative = derivative_integral / normalizer
    oracle_directional_derivative = oracle_derivative_integral / normalizer
    return {
        "quadrature_mass": float(mass.detach()),
        "kl_before": float(kl_before.detach()),
        "kl_after": float(kl_after.detach()),
        "kl_change": float(kl_change.detach()),
        "kl_directional_derivative": float(directional_derivative.detach()),
        "oracle_kl_directional_derivative": float(
            oracle_directional_derivative.detach()
        ),
        "kl_action_ratio": float(
            (directional_derivative / oracle_directional_derivative).detach()
        ),
        "positive_jacobian_mass_fraction": float(valid_fraction.detach()),
    }


def make_eval_states(
    mixture: NearEquilibriumGaussianMixture,
    *,
    count: int,
    epsilon: float,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_domain = count // 2
    generator = cpu_generator(seed)
    real = mixture.sample_clean(
        per_domain,
        epsilon=epsilon,
        domain="real",
        generator=generator,
    )
    fake = mixture.sample_clean(
        per_domain,
        epsilon=epsilon,
        domain="fake",
        generator=generator,
    )
    real = real + float(sigma) * sample_noise_like(real, generator=generator)
    fake = fake + float(sigma) * sample_noise_like(fake, generator=generator)
    states = torch.cat((real, fake), dim=0)
    labels = torch.cat(
        (
            torch.ones(per_domain, device=states.device, dtype=states.dtype),
            torch.zeros(per_domain, device=states.device, dtype=states.dtype),
        )
    )
    return states, labels


def make_fake_eval_states(
    mixture: NearEquilibriumGaussianMixture,
    *,
    count: int,
    epsilon: float,
    sigma: float,
    seed: int,
) -> torch.Tensor:
    generator = cpu_generator(seed)
    clean = mixture.sample_clean(
        count,
        epsilon=epsilon,
        domain="fake",
        generator=generator,
    )
    return clean + float(sigma) * sample_noise_like(clean, generator=generator)


def centered_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().mean().sqrt() * right.square().mean().sqrt()
    if float(denominator.detach()) == 0.0:
        return float("nan")
    return float(((left * right).mean() / denominator).detach())


def ratio_value_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bayes_logits: torch.Tensor,
) -> dict[str, float]:
    metrics = classifier_metrics(logits, labels)
    metrics.update(
        {
            "logit_mse": float(
                (logits - bayes_logits).square().mean().detach()
            ),
            "centered_logit_mse": float(
                (
                    (logits - logits.mean())
                    - (bayes_logits - bayes_logits.mean())
                )
                .square()
                .mean()
                .detach()
            ),
            "logit_correlation": centered_correlation(logits, bayes_logits),
        }
    )
    return metrics


def evaluate_oracle(
    mixture: NearEquilibriumGaussianMixture,
    *,
    epsilon: float,
    train_sample_count: int,
    sigma: float,
    config: ExperimentConfig,
    output_root: Path,
    eval_index: int,
) -> dict[str, Any]:
    states, labels = make_eval_states(
        mixture,
        count=config.eval_samples,
        epsilon=epsilon,
        sigma=sigma,
        seed=config.eval_seed + 1009 * eval_index,
    )
    states.requires_grad_(True)
    target = mixture.residual_score(states, sigma=sigma, epsilon=epsilon)
    bayes_logits = mixture.bayes_real_logit(states, sigma=sigma, epsilon=epsilon)
    metrics: dict[str, Any] = field_metrics(target, target)
    metrics.update(ratio_value_metrics(bayes_logits, labels, bayes_logits))
    metrics.update(
        quadrature_distribution_response(
            mixture,
            lambda grid_states: mixture.residual_score(
                grid_states, sigma=sigma, epsilon=epsilon
            ),
            sigma=sigma,
            epsilon=epsilon,
            step_size=config.pushforward_step_size,
            grid_size=config.quadrature_grid_size,
            batch_size=config.quadrature_batch_size,
        )
    )
    estimate_path = (
        output_root
        / "estimates"
        / f"eps{epsilon:g}_n{train_sample_count}"
        / "oracle"
        / f"sigma{sigma:g}.npz"
    )
    estimate_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        estimate_path,
        states=states.detach().cpu().numpy(),
        target=target.detach().cpu().numpy(),
        estimate=target.detach().cpu().numpy(),
    )
    metrics.update(
        {
            "epsilon": epsilon,
            "train_sample_count": train_sample_count,
            "seed": -1,
            "method": "oracle",
            "kind": "oracle",
            "sigma": sigma,
            "train_seconds": 0.0,
            "parameter_count": 0,
            "estimate_path": str(estimate_path.relative_to(output_root)),
        }
    )
    return metrics


def evaluate_bundle(
    bundle: EstimatorBundle,
    mixture: NearEquilibriumGaussianMixture,
    *,
    epsilon: float,
    train_sample_count: int,
    seed: int,
    sigma: float,
    config: ExperimentConfig,
    output_root: Path,
    eval_index: int,
) -> dict[str, Any]:
    if bundle.kind in DSM_METHODS and sigma <= 0:
        raise ValueError("DSM methods cannot be evaluated at sigma=0")
    if bundle.kind == "zero_ratio" and sigma != 0:
        raise ValueError("the zero-noise classifier is only defined at sigma=0")
    bundle.train(False)
    states, labels = make_eval_states(
        mixture,
        count=config.eval_samples,
        epsilon=epsilon,
        sigma=sigma,
        seed=config.eval_seed + 1009 * eval_index,
    )
    states.requires_grad_(True)
    sigma_batch = torch.full(
        (len(states),), sigma, device=states.device, dtype=states.dtype
    )
    target = mixture.residual_score(states, sigma=sigma, epsilon=epsilon)
    estimate = estimate_field(
        bundle,
        states,
        sigma_batch,
        create_graph=False,
    )
    metrics: dict[str, Any] = field_metrics(estimate.detach(), target.detach())
    if bundle.kind in RATIO_METHODS:
        with torch.no_grad():
            logits = bundle.modules["ratio"](states.detach(), sigma_batch)
            bayes_logits = mixture.bayes_real_logit(
                states.detach(), sigma=sigma, epsilon=epsilon
            )
        metrics.update(ratio_value_metrics(logits, labels, bayes_logits))
    if bundle.kind in DSM_METHODS:
        with torch.no_grad():
            predicted_real_score, predicted_fake_score = estimate_dsm_component_scores(
                bundle, states.detach(), sigma_batch
            )
            _, exact_real_score = mixture.log_prob_and_score(
                states.detach(), sigma=sigma, epsilon=epsilon, domain="real"
            )
            _, exact_fake_score = mixture.log_prob_and_score(
                states.detach(), sigma=sigma, epsilon=epsilon, domain="fake"
            )
            real_error = predicted_real_score - exact_real_score
            fake_error = predicted_fake_score - exact_fake_score
        metrics.update(
            prefixed_metrics(
                "real_component",
                field_metrics(predicted_real_score, exact_real_score),
            )
        )
        metrics.update(
            prefixed_metrics(
                "fake_component",
                field_metrics(predicted_fake_score, exact_fake_score),
            )
        )
        metrics.update(
            prefixed_metrics(
                "component_error",
                pairwise_field_metrics(real_error, fake_error),
            )
        )
        common_error_rms = (0.5 * (real_error + fake_error)).square().sum(dim=1).mean().sqrt()
        difference_error_rms = (real_error - fake_error).square().sum(dim=1).mean().sqrt()
        metrics["component_common_error_rms"] = float(common_error_rms)
        metrics["component_difference_error_rms"] = float(difference_error_rms)
        metrics["component_difference_to_common_error_ratio"] = float(
            difference_error_rms
            / common_error_rms.clamp_min(torch.finfo(common_error_rms.dtype).tiny)
        )
    def grid_field(grid_states: torch.Tensor) -> torch.Tensor:
        grid_sigma = torch.full(
            (len(grid_states),),
            sigma,
            device=grid_states.device,
            dtype=grid_states.dtype,
        )
        return estimate_field(
            bundle,
            grid_states,
            grid_sigma,
            create_graph=bundle.kind in RATIO_METHODS,
        )

    metrics.update(
        quadrature_distribution_response(
            mixture,
            grid_field,
            sigma=sigma,
            epsilon=epsilon,
            step_size=config.pushforward_step_size,
            grid_size=config.quadrature_grid_size,
            batch_size=config.quadrature_batch_size,
        )
    )
    estimate_path = (
        output_root
        / "estimates"
        / f"eps{epsilon:g}_n{train_sample_count}"
        / f"seed{seed}"
        / bundle.label
        / f"sigma{sigma:g}.npz"
    )
    estimate_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        estimate_path,
        states=states.detach().cpu().numpy(),
        target=target.detach().cpu().numpy(),
        estimate=estimate.detach().cpu().numpy(),
    )
    metrics.update(
        {
            "epsilon": epsilon,
            "train_sample_count": train_sample_count,
            "seed": seed,
            "method": bundle.label,
            "kind": bundle.kind,
            "sigma": sigma,
            "train_seconds": bundle.elapsed_seconds,
            "parameter_count": bundle.parameter_count,
            "sobolev_lambda": bundle.sobolev_lambda,
            "estimate_path": str(estimate_path.relative_to(output_root)),
        }
    )
    return metrics


def aggregate_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["epsilon", "train_sample_count", "method", "kind", "sigma"]
    numeric_columns = [
        column
        for column in rows.select_dtypes(include=[np.number]).columns
        if column not in {*group_columns, "seed"}
    ]
    aggregate = rows.groupby(group_columns, dropna=False)[numeric_columns].agg(
        ["mean", "std", "count"]
    )
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    return aggregate.reset_index()


def build_pairwise_seed_metrics(rows: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    learned = rows[rows["seed"] >= 0]
    group_columns = ["epsilon", "train_sample_count", "method", "kind", "sigma"]
    for key, group in learned.groupby(group_columns, dropna=False):
        group = group.sort_values("seed")
        for (_, left), (_, right) in combinations(group.iterrows(), 2):
            left_data = np.load(output_root / left["estimate_path"])
            right_data = np.load(output_root / right["estimate_path"])
            metrics = pairwise_field_metrics(
                torch.from_numpy(left_data["estimate"]),
                torch.from_numpy(right_data["estimate"]),
            )
            records.append(
                {
                    **dict(zip(group_columns, key)),
                    "left_seed": int(left["seed"]),
                    "right_seed": int(right["seed"]),
                    **metrics,
                }
            )
    return pd.DataFrame(records)


def plot_summary(aggregate: pd.DataFrame, output_root: Path) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    metric_specs = (
        ("relative_l2_mean", "Relative field L2", "log"),
        ("global_cosine_mean", "Field cosine with truth", "linear"),
        ("kl_directional_derivative_mean", "KL directional derivative", "linear"),
        ("kl_change_mean", "Finite-map KL change", "linear"),
        ("classifier_auc_mean", "Classifier AUC", "linear"),
        ("logit_correlation_mean", "Logit correlation with Bayes ratio", "linear"),
    )
    for (epsilon, sample_count), subset in aggregate.groupby(
        ["epsilon", "train_sample_count"]
    ):
        figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for axis, (metric, title, scale) in zip(axes.flat, metric_specs):
            if metric not in subset.columns:
                axis.set_visible(False)
                continue
            for method, method_rows in subset.groupby("method"):
                valid = method_rows.dropna(subset=[metric]).sort_values("sigma")
                if valid.empty:
                    continue
                axis.plot(valid["sigma"], valid[metric], marker="o", label=method)
                std_column = metric.replace("_mean", "_std")
                if std_column in valid and valid[std_column].notna().any():
                    mean = valid[metric].to_numpy()
                    std = valid[std_column].fillna(0).to_numpy()
                    x = valid["sigma"].to_numpy()
                    axis.fill_between(x, mean - std, mean + std, alpha=0.12)
            axis.set_title(title)
            axis.set_xlabel("Gaussian noise sigma")
            axis.set_xscale("symlog", linthresh=0.03)
            axis.set_yscale(scale)
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="outside lower center", ncol=3)
        figure.suptitle(
            f"Residual-score estimator audit: epsilon={epsilon:g}, n={sample_count}"
        )
        figure.savefig(
            figure_root / f"audit_eps{epsilon:g}_n{sample_count}.png", dpi=180
        )
        plt.close(figure)


def method_jobs(config: ExperimentConfig) -> Iterable[tuple[str, float]]:
    for method in config.methods:
        if method == "sobolev_ratio":
            for value in config.sobolev_lambdas:
                yield method, value
        else:
            yield method, 0.0


def run(config: ExperimentConfig, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    device = torch.device(config.device)
    mixture = NearEquilibriumGaussianMixture.ring(
        components=config.components,
        radius=config.radius,
        component_std=config.component_std,
        perturbation_amplitude=config.perturbation_amplitude,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    completed_oracles: set[tuple[float, int, float]] = set()
    total_jobs = (
        len(config.epsilons)
        * len(config.train_samples)
        * len(config.seeds)
        * sum(
            len(config.sobolev_lambdas) if method == "sobolev_ratio" else 1
            for method in config.methods
        )
    )
    job_index = 0
    for epsilon in config.epsilons:
        for train_sample_count in config.train_samples:
            for seed in config.seeds:
                bank_seed = (
                    config.data_seed
                    + 1_000_003 * seed
                    + 10_007 * train_sample_count
                    + int(round(epsilon * 1_000_000))
                )
                real_bank, fake_bank = build_clean_banks(
                    mixture,
                    count=train_sample_count,
                    epsilon=epsilon,
                    seed=bank_seed,
                )
                for method, sobolev_lambda in method_jobs(config):
                    job_index += 1
                    model_seed = bank_seed + 97
                    print(
                        f"[{job_index}/{total_jobs}] eps={epsilon:g} "
                        f"n={train_sample_count} seed={seed} method={method} "
                        f"lambda={sobolev_lambda:g}",
                        flush=True,
                    )
                    if method in RATIO_METHODS:
                        bundle = train_ratio_estimator(
                            kind=method,
                            sobolev_lambda=sobolev_lambda,
                            real_bank=real_bank,
                            fake_bank=fake_bank,
                            config=config,
                            seed=model_seed,
                            device=device,
                        )
                    else:
                        bundle = train_dsm_estimator(
                            kind=method,
                            real_bank=real_bank,
                            fake_bank=fake_bank,
                            config=config,
                            seed=model_seed,
                            device=device,
                        )
                    checkpoint_path = (
                        output_root
                        / "checkpoints"
                        / f"eps{epsilon:g}_n{train_sample_count}"
                        / f"seed{seed}"
                        / f"{bundle.label}.pt"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(bundle.state_dict(), checkpoint_path)
                    for history_row in bundle.history:
                        histories.append(
                            {
                                "epsilon": epsilon,
                                "train_sample_count": train_sample_count,
                                "seed": seed,
                                "method": bundle.label,
                                "kind": bundle.kind,
                                **history_row,
                            }
                        )
                    for eval_index, sigma in enumerate(config.eval_sigmas):
                        if bundle.kind in DSM_METHODS and sigma <= 0:
                            continue
                        if bundle.kind == "zero_ratio" and sigma != 0:
                            continue
                        oracle_key = (epsilon, train_sample_count, sigma)
                        if oracle_key not in completed_oracles:
                            rows.append(
                                evaluate_oracle(
                                    mixture,
                                    epsilon=epsilon,
                                    train_sample_count=train_sample_count,
                                    sigma=sigma,
                                    config=config,
                                    output_root=output_root,
                                    eval_index=eval_index,
                                )
                            )
                            completed_oracles.add(oracle_key)
                        rows.append(
                            evaluate_bundle(
                                bundle,
                                mixture,
                                epsilon=epsilon,
                                train_sample_count=train_sample_count,
                                seed=seed,
                                sigma=sigma,
                                config=config,
                                output_root=output_root,
                                eval_index=eval_index,
                            )
                        )
                    pd.DataFrame(rows).to_csv(output_root / "metrics.csv", index=False)
                    pd.DataFrame(histories).to_csv(
                        output_root / "training_curves.csv", index=False
                    )
                    print(
                        f"  finished {bundle.label} in {bundle.elapsed_seconds:.2f}s",
                        flush=True,
                    )
    metrics = pd.DataFrame(rows)
    aggregate = aggregate_metrics(metrics)
    aggregate.to_csv(output_root / "aggregate_metrics.csv", index=False)
    pairwise = build_pairwise_seed_metrics(metrics, output_root)
    pairwise.to_csv(output_root / "pairwise_seed_metrics.csv", index=False)
    if not pairwise.empty:
        pairwise.groupby(
            ["epsilon", "train_sample_count", "method", "kind", "sigma"],
            dropna=False,
        )[
            ["pairwise_cosine", "pairwise_relative_l2"]
        ].agg(["mean", "std", "count"]).to_csv(
            output_root / "pairwise_seed_summary.csv"
        )
    plot_summary(aggregate, output_root)
    print(f"wrote results to {output_root}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epsilons", type=parse_float_tuple, default=(0.3, 0.1, 0.03))
    parser.add_argument("--train-samples", type=parse_int_tuple, default=(2048, 8192))
    parser.add_argument("--seeds", type=parse_int_tuple, default=(0, 1, 2))
    parser.add_argument(
        "--methods",
        type=parse_str_tuple,
        default=(
            "zero_ratio",
            "ratio",
            "sobolev_ratio",
            "separate_dsm",
            "shared_dsm",
        ),
    )
    parser.add_argument(
        "--sobolev-lambdas", type=parse_float_tuple, default=(1e-4, 1e-3, 1e-2)
    )
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--frequencies", type=int, default=6)
    parser.add_argument("--domain-dim", type=int, default=16)
    parser.add_argument("--train-sigma-min", type=float, default=0.03)
    parser.add_argument("--train-sigma-max", type=float, default=1.5)
    parser.add_argument(
        "--eval-sigmas",
        type=parse_float_tuple,
        default=(0.0, 0.03, 0.1, 0.3, 0.7, 1.5),
    )
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--pushforward-step-size", type=float, default=0.01)
    parser.add_argument("--quadrature-grid-size", type=int, default=160)
    parser.add_argument("--quadrature-batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--components", type=int, default=12)
    parser.add_argument("--radius", type=float, default=2.5)
    parser.add_argument("--component-std", type=float, default=0.22)
    parser.add_argument("--perturbation-amplitude", type=float, default=0.85)
    parser.add_argument("--data-seed", type=int, default=20260824)
    parser.add_argument("--eval-seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        raise ValueError("batch-size must be an even integer >= 2")
    if args.train_steps <= 0 or args.eval_samples < 2 or args.eval_samples % 2:
        raise ValueError("invalid train/eval size")
    if args.quadrature_grid_size < 16 or args.quadrature_batch_size <= 0:
        raise ValueError("invalid quadrature configuration")
    if args.log_every <= 0:
        raise ValueError("invalid logging configuration")
    unknown = set(args.methods) - ALL_METHODS
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if "sobolev_ratio" in args.methods and any(
        value < 0 for value in args.sobolev_lambdas
    ):
        raise ValueError("Sobolev lambdas must be non-negative")
    config = ExperimentConfig(
        epsilons=args.epsilons,
        train_samples=args.train_samples,
        seeds=args.seeds,
        methods=args.methods,
        sobolev_lambdas=args.sobolev_lambdas,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        frequencies=args.frequencies,
        domain_dim=args.domain_dim,
        train_sigma_min=args.train_sigma_min,
        train_sigma_max=args.train_sigma_max,
        eval_sigmas=args.eval_sigmas,
        eval_samples=args.eval_samples,
        pushforward_step_size=args.pushforward_step_size,
        quadrature_grid_size=args.quadrature_grid_size,
        quadrature_batch_size=args.quadrature_batch_size,
        log_every=args.log_every,
        components=args.components,
        radius=args.radius,
        component_std=args.component_std,
        perturbation_amplitude=args.perturbation_amplitude,
        data_seed=args.data_seed,
        eval_seed=args.eval_seed,
        device=args.device,
    )
    run(config, args.output_root)


if __name__ == "__main__":
    main()
