#!/usr/bin/env python3
"""Test whether an adaptive Fréchet game can co-adapt while losing true modes."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from experiments.advfd_cleanroom.core import (
    batch_moments,
    fit_calibration,
    frechet_from_features,
    normalized_frechet_loss,
)
from experiments.advfd_cleanroom.run_critic_generalization_toy import (
    FeatureCritic,
    covariance_effective_rank,
)


Component = Literal["full", "mean"]


@dataclass(frozen=True)
class GameConfig:
    modes: int = 12
    radius: float = 2.5
    noise_std: float = 0.10
    train_batch: int = 240
    heldout_samples: int = 6000
    steps: int = 2000
    eval_every: int = 50
    critic_steps: int = 1
    critic_hidden: int = 128
    critic_depth: int = 4
    critic_features: int = 12
    critic_lr: float = 5e-4
    generator_lr: float = 2e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    whitening_epsilon: float = 1e-3
    static_features: int = 32


class RingGenerator(nn.Module):
    def __init__(self, config: GameConfig) -> None:
        super().__init__()
        indices = torch.arange(config.modes, dtype=torch.float32)
        angles = indices * (2.0 * math.pi / config.modes)
        # A mildly biased but mode-complete post-training initialization.
        angles = angles + 0.12 * (2.0 * math.pi / config.modes) * torch.sin(angles)
        radii = config.radius * (1.08 + 0.04 * torch.cos(2.0 * angles))
        centers = radii[:, None] * torch.stack([angles.cos(), angles.sin()], dim=1)
        self.centers = nn.Parameter(centers)
        self.log_stds = nn.Parameter(
            torch.full((config.modes, 1), math.log(config.noise_std * 1.15))
        )

    def forward(self, labels: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.centers[labels] + self.log_stds[labels].exp() * noise


class FixedFourierFeatures(nn.Module):
    def __init__(self, output_pairs: int, radius: float, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        directions = torch.randn(output_pairs, 2, generator=generator)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scales = torch.linspace(0.5, 5.0, output_pairs) / radius
        self.register_buffer("frequencies", directions * scales[:, None])
        self.radius = float(radius)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phases = inputs @ self.frequencies.mT
        return torch.cat([inputs / self.radius, phases.sin(), phases.cos()], dim=1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ring_centers(config: GameConfig, device: torch.device) -> torch.Tensor:
    angles = torch.arange(config.modes, device=device) * (
        2.0 * math.pi / config.modes
    )
    return config.radius * torch.stack([angles.cos(), angles.sin()], dim=1)


def balanced_labels(count: int, modes: int, seed: int, device: torch.device) -> torch.Tensor:
    labels = torch.arange(count) % modes
    generator = torch.Generator().manual_seed(seed)
    labels = labels[torch.randperm(count, generator=generator)]
    return labels.to(device)


def fixed_noise(count: int, seed: int, device: torch.device) -> torch.Tensor:
    return torch.randn(count, 2, generator=torch.Generator().manual_seed(seed)).to(device)


def sample_target(config: GameConfig, count: int, seed: int, device: torch.device) -> torch.Tensor:
    labels = balanced_labels(count, config.modes, seed, device)
    return ring_centers(config, device)[labels] + config.noise_std * fixed_noise(
        count, seed + 1, device
    )


def component_value(real: torch.Tensor, fake: torch.Tensor, component: Component) -> torch.Tensor:
    values = frechet_from_features(real, fake)
    return values.total if component == "full" else values.mean


def adaptive_distance(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    component: Component,
    epsilon: float,
    detach_real: bool,
) -> torch.Tensor:
    real_features = critic(real)
    if detach_real:
        real_features = real_features.detach()
    fake_features = critic(fake)
    calibration = fit_calibration(
        real_features,
        fake_features,
        mode="real",
        epsilon=epsilon,
        detach_statistics=True,
    )
    return component_value(
        calibration.apply(real_features),
        calibration.apply(fake_features),
        component,
    )


def target_nll(samples: torch.Tensor, config: GameConfig) -> float:
    centers = ring_centers(config, samples.device)
    squared = (samples[:, None, :] - centers[None, :, :]).square().sum(dim=2)
    log_components = (
        -0.5 * squared / (config.noise_std**2)
        - math.log(2.0 * math.pi * config.noise_std**2)
        - math.log(config.modes)
    )
    return float(-torch.logsumexp(log_components, dim=1).mean())


def mode_metrics(samples: torch.Tensor, config: GameConfig) -> dict[str, float]:
    centers = ring_centers(config, samples.device)
    nearest = torch.cdist(samples, centers).argmin(dim=1)
    counts = torch.bincount(nearest, minlength=config.modes).double()
    probabilities = counts / counts.sum()
    uniform = torch.full_like(probabilities, 1.0 / config.modes)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return {
        "mode_coverage_1pct": float((probabilities >= 0.01).sum()),
        "mode_entropy_normalized": float(entropy / math.log(config.modes)),
        "mode_mass_tv": float(0.5 * (probabilities - uniform).abs().sum()),
        "max_mode_fraction": float(probabilities.max()),
    }


def generator_center_metrics(generator: RingGenerator, config: GameConfig) -> dict[str, float]:
    centers = generator.centers.detach()
    true_centers = ring_centers(config, centers.device)
    nearest = torch.cdist(centers, true_centers)
    assigned = nearest.argmin(dim=1)
    pairwise = torch.cdist(centers, centers)
    pairwise.fill_diagonal_(float("inf"))
    return {
        "generator_unique_target_modes": float(assigned.unique().numel()),
        "generator_center_error": float(nearest.min(dim=1).values.mean()),
        "generator_min_center_distance": float(pairwise.min()),
        "generator_mean_std": float(generator.log_stds.detach().exp().mean()),
    }


@torch.no_grad()
def evaluate_game(
    generator: RingGenerator,
    critic: FeatureCritic,
    static_features: FixedFourierFeatures,
    config: GameConfig,
    *,
    component: Component,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    labels = balanced_labels(config.heldout_samples, config.modes, seed + 1, device)
    noise = fixed_noise(config.heldout_samples, seed + 2, device)
    fake = generator(labels, noise)
    real = sample_target(config, config.heldout_samples, seed + 3, device)
    static_fd = frechet_from_features(static_features(real), static_features(fake)).total
    adaptive_fd = adaptive_distance(
        critic,
        real,
        fake,
        component=component,
        epsilon=config.whitening_epsilon,
        detach_real=True,
    )
    critic_real = critic(real)
    calibration = fit_calibration(
        critic_real,
        critic(fake),
        mode="real",
        epsilon=config.whitening_epsilon,
        detach_statistics=True,
    )
    calibrated_real = calibration.apply(critic_real)
    return {
        "static_fd": float(static_fd),
        "adaptive_fd": float(adaptive_fd),
        "target_nll": target_nll(fake, config),
        "critic_feature_effective_rank": covariance_effective_rank(calibrated_real),
        **mode_metrics(fake, config),
        **generator_center_metrics(generator, config),
    }


def run_game(
    config: GameConfig,
    *,
    seed: int,
    condition: str,
    component: Component,
    static_weight: float,
    adaptive_weight: float,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], dict[int, tuple[dict, dict]]]:
    seed_everything(seed)
    generator = RingGenerator(config).to(device)
    critic = FeatureCritic(
        config.critic_hidden, config.critic_depth, config.critic_features
    ).to(device)
    static_features = FixedFourierFeatures(
        config.static_features, config.radius, seed + 91
    ).to(device)
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(), lr=config.generator_lr, weight_decay=0.0
    )
    critic_optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=config.critic_lr,
        weight_decay=config.weight_decay,
    )
    snapshot_steps = {0, config.steps // 4, config.steps // 2, config.steps}
    snapshots: dict[int, tuple[dict, dict]] = {}
    rows: list[dict[str, float | int | str]] = []

    for step in range(config.steps + 1):
        if step % config.eval_every == 0 or step == config.steps:
            metrics = evaluate_game(
                generator,
                critic,
                static_features,
                config,
                component=component,
                seed=seed + 100_000 + step,
                device=device,
            )
            rows.append(
                {
                    "condition": condition,
                    "component": component,
                    "seed": seed,
                    "step": step,
                    **metrics,
                }
            )
        if step in snapshot_steps:
            snapshots[step] = (
                copy.deepcopy(generator.state_dict()),
                copy.deepcopy(critic.state_dict()),
            )
        if step == config.steps:
            break

        labels = balanced_labels(config.train_batch, config.modes, seed + 17 * step, device)
        noise = fixed_noise(config.train_batch, seed + 17 * step + 1, device)
        real = sample_target(config, config.train_batch, seed + 17 * step + 2, device)

        if adaptive_weight > 0.0:
            for critic_step in range(config.critic_steps):
                critic.requires_grad_(True)
                critic_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    fake_detached = generator(labels, noise)
                distance = adaptive_distance(
                    critic,
                    real,
                    fake_detached,
                    component=component,
                    epsilon=config.whitening_epsilon,
                    detach_real=True,
                )
                (-distance).backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), config.gradient_clip
                )
                critic_optimizer.step()

        critic.requires_grad_(False)
        generator_optimizer.zero_grad(set_to_none=True)
        fake = generator(labels, noise)
        loss = fake.new_zeros(())
        if static_weight > 0.0:
            static_components = frechet_from_features(
                static_features(real), static_features(fake)
            )
            loss = loss + static_weight * normalized_frechet_loss(static_components)
        if adaptive_weight > 0.0:
            adaptive = adaptive_distance(
                critic,
                real,
                fake,
                component=component,
                epsilon=config.whitening_epsilon,
                detach_real=True,
            )
            loss = loss + adaptive_weight * adaptive / (adaptive.detach() + 0.01)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), config.gradient_clip)
        generator_optimizer.step()
        critic.requires_grad_(True)

    return rows, snapshots


@torch.no_grad()
def crossplay(
    config: GameConfig,
    snapshots: dict[int, tuple[dict, dict]],
    *,
    component: Component,
    seed: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    real = sample_target(config, config.heldout_samples, seed + 800_000, device)
    labels = balanced_labels(
        config.heldout_samples, config.modes, seed + 800_001, device
    )
    noise = fixed_noise(config.heldout_samples, seed + 800_002, device)
    rows = []
    for critic_step, (_, critic_state) in snapshots.items():
        critic = FeatureCritic(
            config.critic_hidden, config.critic_depth, config.critic_features
        ).to(device)
        critic.load_state_dict(critic_state)
        critic.eval()
        for generator_step, (generator_state, _) in snapshots.items():
            generator = RingGenerator(config).to(device)
            generator.load_state_dict(generator_state)
            generator.eval()
            fake = generator(labels, noise)
            distance = adaptive_distance(
                critic,
                real,
                fake,
                component=component,
                epsilon=config.whitening_epsilon,
                detach_real=True,
            )
            rows.append(
                {
                    "critic_step": critic_step,
                    "generator_step": generator_step,
                    "adaptive_fd": float(distance),
                    **mode_metrics(fake, config),
                }
            )
    return rows


def plot_metrics(frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = (
        ("target_nll", "Target NLL"),
        ("mode_mass_tv", "Mode mass TV"),
        ("adaptive_fd", "Paired adaptive FD"),
        ("critic_feature_effective_rank", "Critic feature effective rank"),
    )
    for axis, (metric, title) in zip(axes.flat, metrics):
        for condition, group in frame.groupby("condition"):
            summary = group.groupby("step")[metric].mean()
            axis.plot(summary.index, summary.values, label=condition)
        axis.set_title(title)
        axis.set_xlabel("step")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def parse_conditions(raw: str) -> list[tuple[str, Component, float, float]]:
    presets = {
        "static": ("full", 1.0, 0.0),
        "full_only": ("full", 0.0, 1.0),
        "full_w0.05": ("full", 1.0, 0.05),
        "full_w0.5": ("full", 1.0, 0.5),
        "full_w2": ("full", 1.0, 2.0),
        "mean_w0.5": ("mean", 1.0, 0.5),
    }
    result = []
    for name in raw.split(","):
        if name not in presets:
            raise ValueError(f"unknown condition {name!r}")
        component, static_weight, adaptive_weight = presets[name]
        result.append((name, component, static_weight, adaptive_weight))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="8301,8302,8303")
    parser.add_argument(
        "--conditions",
        default="static,full_only,full_w0.05,full_w0.5,full_w2,mean_w0.5",
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    config = GameConfig(steps=args.steps, eval_every=args.eval_every)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    conditions = parse_conditions(args.conditions)
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "config.json").write_text(
        json.dumps(
            {
                "config": asdict(config),
                "seeds": seeds,
                "conditions": conditions,
                "device": str(device),
                "interpretation_boundary": (
                    "Low critic rank or cross-play interaction is co-adaptation evidence. "
                    "Generator mode collapse additionally requires declining true mode coverage, "
                    "entropy, or target likelihood."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    all_rows = []
    all_crossplay = []
    for seed in seeds:
        for name, component, static_weight, adaptive_weight in conditions:
            print(f"running seed={seed} condition={name}", flush=True)
            rows, snapshots = run_game(
                config,
                seed=seed,
                condition=name,
                component=component,
                static_weight=static_weight,
                adaptive_weight=adaptive_weight,
                device=device,
            )
            all_rows.extend(rows)
            if adaptive_weight > 0.0:
                for row in crossplay(
                    config,
                    snapshots,
                    component=component,
                    seed=seed,
                    device=device,
                ):
                    all_crossplay.append(
                        {"seed": seed, "condition": name, "component": component, **row}
                    )
            pd.DataFrame(all_rows).to_csv(args.output_root / "metrics.csv", index=False)
            pd.DataFrame(all_crossplay).to_csv(
                args.output_root / "crossplay.csv", index=False
            )

    frame = pd.DataFrame(all_rows)
    final = frame.sort_values("step").groupby(["condition", "seed"], as_index=False).tail(1)
    final.to_csv(args.output_root / "summary.csv", index=False)
    plot_metrics(frame, args.output_root / "mode_collapse_dynamics.png")
    print(final.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
