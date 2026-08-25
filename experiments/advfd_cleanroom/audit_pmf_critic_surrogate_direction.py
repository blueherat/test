#!/usr/bin/env python3
"""Audit whether an AdvFD stop-gradient critic step raises its quotient value.

The paper defines the adaptive discrepancy after recalibrating the real-feature
whitening frame.  The official D-step instead differentiates with the real
features and whitening frame detached.  This audit computes that implemented
ascent direction on a fixed real/fake image pair, perturbs the actual critic
parameters along it, and then recomputes both:

* the frozen-frame surrogate used to form the direction; and
* the recalibrated current-pair AdvFD value from Equation (9).

Central finite differences avoid differentiating through a 2048-dimensional
eigendecomposition while still measuring the true directional derivative of
the recalibrated scalar objective.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


EQVAE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EQVAE_ROOT))

from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_gradients import (
    checkpoint_population_moments,
    select_adv_state,
)
from experiments.advfd_cleanroom.audit_pmf_generator_component_gradients import (
    build_generator,
    make_noise_and_labels,
    sample_generator,
)


DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument(
        "--packed-imagenet-root",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=0,
        help=(
            "fixed orthogonal feature projection for a robustness control; "
            "0 keeps the official full feature space"
        ),
    )
    parser.add_argument("--projection-seed", type=int, default=20260826)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument(
        "--relative-step-sizes",
        type=parse_floats,
        default=(1e-7, 3e-7, 1e-6, 3e-6),
    )
    parser.add_argument("--cfg-omega", type=float, default=8.5)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def population_moments(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.double()
    mean = features.mean(dim=0)
    second = features.T @ features / features.shape[0]
    covariance = second - mean[:, None] * mean[None, :]
    return mean, 0.5 * (covariance + covariance.T)


def population_mean_and_second(
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.double()
    return features.mean(dim=0), features.T @ features / features.shape[0]


def control_variate_moments(
    anchor: tuple[torch.Tensor, torch.Tensor],
    baseline_features: torch.Tensor,
    perturbed_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate perturbed population moments from an EMA anchor and paired batch."""

    anchor_mean, anchor_covariance = anchor
    baseline_mean, baseline_second = population_mean_and_second(baseline_features)
    perturbed_mean, perturbed_second = population_mean_and_second(perturbed_features)
    mean = anchor_mean + perturbed_mean - baseline_mean
    anchor_second = anchor_covariance + anchor_mean[:, None] * anchor_mean[None, :]
    second = anchor_second + perturbed_second - baseline_second
    covariance = second - mean[:, None] * mean[None, :]
    return mean, 0.5 * (covariance + covariance.T)


def fd_components(
    real_features: torch.Tensor,
    fake_features: torch.Tensor,
    *,
    epsilon: float,
    frozen_real_moments: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from experiments.advfd_cleanroom.temporal_gauge import (
        real_whitened_fd_components_from_stats,
    )

    if frozen_real_moments is None:
        real_mean, real_covariance = population_moments(real_features)
    else:
        real_mean, real_covariance = frozen_real_moments
    fake_mean, fake_covariance = population_moments(fake_features)
    mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
        real_mean,
        real_covariance,
        fake_mean,
        fake_covariance,
        epsilon=epsilon,
    )
    return mean_term, covariance_term


def fd_components_from_moments(
    real_moments: tuple[torch.Tensor, torch.Tensor],
    fake_moments: tuple[torch.Tensor, torch.Tensor],
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from experiments.advfd_cleanroom.temporal_gauge import (
        real_whitened_fd_components_from_stats,
    )

    mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
        *real_moments,
        *fake_moments,
        epsilon=epsilon,
    )
    return mean_term, covariance_term


def extract_features(
    critic: torch.nn.Module,
    images: torch.Tensor,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    features, _ = critic(images)
    return features if projection is None else features @ projection


def project_moments(
    moments: tuple[torch.Tensor, torch.Tensor],
    projection: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if projection is None:
        return moments
    mean, covariance = moments
    projection = projection.double()
    return mean @ projection, projection.T @ covariance @ projection


def parameter_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    return torch.stack(
        [parameter.detach().double().square().sum() for parameter in parameters]
    ).sum().sqrt()


def gradient_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    terms = [
        parameter.grad.detach().double().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not terms:
        raise RuntimeError("critic objective produced no parameter gradients")
    return torch.stack(terms).sum().sqrt()


@torch.no_grad()
def shift_along_gradient(
    parameters: list[torch.nn.Parameter],
    *,
    distance: float,
    gradient_l2: float,
) -> None:
    scale = float(distance) / float(gradient_l2)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.add_(parameter.grad, alpha=scale)


@torch.no_grad()
def evaluate_values(
    critic: torch.nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    *,
    projection: torch.Tensor | None,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    ema_beta: float,
    implemented_real_moments: tuple[torch.Tensor, torch.Tensor],
    epsilon: float,
) -> dict[str, dict[str, float]]:
    from experiments.advfd_cleanroom.temporal_gauge import blend_anchor_with_batch

    real_features = extract_features(critic, real_images, projection)
    fake_features = extract_features(critic, fake_images, projection)
    recalibrated_real_moments = blend_anchor_with_batch(
        *real_anchor,
        real_features,
        beta=ema_beta,
    )
    current_fake_moments = blend_anchor_with_batch(
        *fake_anchor,
        fake_features,
        beta=ema_beta,
    )
    values: dict[str, dict[str, float]] = {}
    for name, current_real_moments in (
        ("recalibrated_ema_quotient", recalibrated_real_moments),
        ("implemented_surrogate", implemented_real_moments),
    ):
        mean_term, covariance_term = fd_components_from_moments(
            current_real_moments,
            current_fake_moments,
            epsilon=epsilon,
        )
        values[name] = {
            "mean": float(mean_term),
            "covariance": float(covariance_term),
            "full": float(mean_term + covariance_term),
        }
    return values


def central_difference(
    plus: dict[str, dict[str, float]],
    minus: dict[str, dict[str, float]],
    distance: float,
) -> dict[str, dict[str, float]]:
    return {
        frame: {
            component: (plus[frame][component] - minus[frame][component])
            / (2.0 * distance)
            for component in plus[frame]
        }
        for frame in plus
    }


def audit_trial(
    critic: torch.nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    *,
    projection: torch.Tensor | None,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    ema_beta: float,
    relative_step_sizes: tuple[float, ...],
    epsilon: float,
) -> dict[str, Any]:
    from experiments.advfd_cleanroom.temporal_gauge import blend_anchor_with_batch

    parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    critic.zero_grad(set_to_none=True)
    with torch.no_grad():
        baseline_real_features = extract_features(critic, real_images, projection)
        implemented_real_moments = blend_anchor_with_batch(
            *real_anchor,
            baseline_real_features,
            beta=ema_beta,
        )
    fake_features = extract_features(critic, fake_images, projection)
    surrogate_fake_moments = blend_anchor_with_batch(
        *fake_anchor,
        fake_features,
        beta=ema_beta,
    )
    mean_term, covariance_term = fd_components_from_moments(
        implemented_real_moments,
        surrogate_fake_moments,
        epsilon=epsilon,
    )
    surrogate_full = mean_term + covariance_term
    surrogate_full.backward()
    grad_l2 = float(gradient_norm(parameters))
    parameter_l2 = float(parameter_norm(parameters))
    baseline = evaluate_values(
        critic,
        real_images,
        fake_images,
        projection=projection,
        real_anchor=real_anchor,
        fake_anchor=fake_anchor,
        ema_beta=ema_beta,
        implemented_real_moments=implemented_real_moments,
        epsilon=epsilon,
    )
    parameter_snapshots = [parameter.detach().cpu().clone() for parameter in parameters]

    @torch.no_grad()
    def set_distance(distance: float) -> None:
        for parameter, snapshot in zip(parameters, parameter_snapshots):
            parameter.copy_(snapshot.to(device=parameter.device, dtype=parameter.dtype))
        shift_along_gradient(parameters, distance=distance, gradient_l2=grad_l2)

    finite_steps = []
    for relative_step in relative_step_sizes:
        distance = float(relative_step) * parameter_l2
        set_distance(distance)
        plus = evaluate_values(
            critic,
            real_images,
            fake_images,
            projection=projection,
            real_anchor=real_anchor,
            fake_anchor=fake_anchor,
            ema_beta=ema_beta,
            implemented_real_moments=implemented_real_moments,
            epsilon=epsilon,
        )
        set_distance(-distance)
        minus = evaluate_values(
            critic,
            real_images,
            fake_images,
            projection=projection,
            real_anchor=real_anchor,
            fake_anchor=fake_anchor,
            ema_beta=ema_beta,
            implemented_real_moments=implemented_real_moments,
            epsilon=epsilon,
        )
        derivatives = central_difference(plus, minus, distance)
        implemented_derivative = derivatives["implemented_surrogate"]["full"]
        recalibrated_derivative = derivatives["recalibrated_ema_quotient"]["full"]
        finite_steps.append(
            {
                "relative_parameter_step": relative_step,
                "absolute_parameter_step_l2": distance,
                "plus": plus,
                "minus": minus,
                "directional_derivatives": derivatives,
                "implemented_derivative_to_autograd_ratio": (
                    implemented_derivative / grad_l2
                    if grad_l2 > 0.0
                    else float("nan")
                ),
                "recalibrated_to_implemented_directional_ratio": (
                    recalibrated_derivative / implemented_derivative
                    if abs(implemented_derivative) > 1e-30
                    else float("nan")
                ),
            }
        )
    set_distance(0.0)
    restored = evaluate_values(
        critic,
        real_images,
        fake_images,
        projection=projection,
        real_anchor=real_anchor,
        fake_anchor=fake_anchor,
        ema_beta=ema_beta,
        implemented_real_moments=implemented_real_moments,
        epsilon=epsilon,
    )
    restoration_error = max(
        abs(restored[frame][component] - baseline[frame][component])
        for frame in baseline
        for component in baseline[frame]
    )
    critic.zero_grad(set_to_none=True)
    return {
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_l2": parameter_l2,
        "surrogate_gradient_l2": grad_l2,
        "baseline": baseline,
        "finite_steps": finite_steps,
        "maximum_restoration_error": restoration_error,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("batch size must be at least two")
    if not args.relative_step_sizes or any(step <= 0.0 for step in args.relative_step_sizes):
        raise ValueError("relative step sizes must be positive")
    if args.projection_dim < 0:
        raise ValueError("projection dimension cannot be negative")
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))
    device = torch.device(args.device)

    from experiments.raev2_training_core import DeterministicImageNetPacked
    from experiments.advfd_cleanroom.temporal_gauge import torch_population_moments
    from frechet_distance.repr_models import load_repr_model
    import models

    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    adv_state = select_adv_state(checkpoint)
    real_anchor = torch_population_moments(
        checkpoint_population_moments(adv_state["real_stats"]), device=device
    )
    fake_anchor = torch_population_moments(
        checkpoint_population_moments(adv_state["fake_stats"]), device=device
    )
    feature_dim = int(real_anchor[0].numel())
    if args.projection_dim > feature_dim:
        raise ValueError("projection dimension exceeds critic feature dimension")
    if args.projection_dim:
        projection_generator = torch.Generator(device=device).manual_seed(
            args.projection_seed
        )
        random_projection = torch.randn(
            feature_dim,
            args.projection_dim,
            generator=projection_generator,
            device=device,
            dtype=torch.float64,
        )
        projection, _ = torch.linalg.qr(random_projection, mode="reduced")
        projection = projection.float()
    else:
        projection = None
    real_anchor = project_moments(real_anchor, projection)
    fake_anchor = project_moments(fake_anchor, projection)
    critic, _, _, _ = load_repr_model("inception", device=str(device))
    critic.load_state_dict(adv_state["model"], strict=True)
    critic.eval().requires_grad_(True)

    manifest = json.loads(
        args.adapter_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    generator = build_generator(models, manifest["official_arguments"], device)
    incompatible = generator.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"generator checkpoint mismatch: {incompatible}")
    generator.eval().requires_grad_(False)
    del checkpoint, adv_state

    dataset = DeterministicImageNetPacked(
        args.packed_imagenet_root,
        split="train",
        image_size=256,
        augmentation_seed=1,
        horizontal_flip=False,
    )
    rng = np.random.default_rng(args.seed + 91)
    indices = rng.choice(len(dataset), size=args.batch_size * args.trials, replace=False)
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trials = []
    for trial_index, real_batch in enumerate(loader):
        real_images = real_batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            noise, labels = make_noise_and_labels(
                generator,
                args.batch_size,
                seed=args.seed + trial_index,
                device=device,
            )
            fake_images = sample_generator(generator, noise, labels, args).detach()
        trials.append(
            audit_trial(
                critic,
                real_images,
                fake_images,
                projection=projection,
                real_anchor=real_anchor,
                fake_anchor=fake_anchor,
                ema_beta=args.ema_beta,
                relative_step_sizes=args.relative_step_sizes,
                epsilon=args.whiten_eps,
            )
        )
        print(f"completed trial {trial_index + 1}/{args.trials}", flush=True)

    ratio_by_step = {}
    for step_index, relative_step in enumerate(args.relative_step_sizes):
        values = [
            trial["finite_steps"][step_index][
                "recalibrated_to_implemented_directional_ratio"
            ]
            for trial in trials
        ]
        finite_values = [value for value in values if math.isfinite(value)]
        ratio_by_step[str(relative_step)] = {
            "mean": float(np.mean(finite_values)),
            "std": float(np.std(finite_values)),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
        }
    result = {
        "protocol": "advfd_pmf_critic_surrogate_direction_audit_v2",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "adapter_manifest": str(args.adapter_manifest.expanduser().resolve()),
        "batch_size": args.batch_size,
        "trials": len(trials),
        "relative_step_sizes": list(args.relative_step_sizes),
        "projection_dim": args.projection_dim or feature_dim,
        "projection_seed": args.projection_seed,
        "interpretation_boundary": (
            "The direction matches the raw checkpoint-EMA stop-gradient critic "
            "gradient and is normalized as gradient clipping would do. At the "
            "unperturbed checkpoint, implemented_surrogate and "
            "recalibrated_ema_quotient use identical EMA-blended moments. Under "
            "a critic perturbation, the implemented surrogate freezes the real "
            "whitening frame while the quotient recalibrates it from the paired "
            "real batch. Their derivative difference therefore isolates the "
            "detached real-frame approximation. Projection_dim=0 retains the "
            "official full Inception feature space; a positive dimension is "
            "only a projected robustness control. Adam preconditioning is "
            "deliberately excluded."
        ),
        "aggregate_recalibrated_to_implemented_ratio": ratio_by_step,
        "trials_detail": trials,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate_recalibrated_to_implemented_ratio"], indent=2))


if __name__ == "__main__":
    main()
