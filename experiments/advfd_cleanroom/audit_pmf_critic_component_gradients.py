#!/usr/bin/env python3
"""Compare full, mean, and covariance AdvFD critic gradients on paired data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


EQVAE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EQVAE_ROOT))

from experiments.advfd_cleanroom.audit_advfd_temporal_gauge_gradients import (
    aggregate_scalars,
    checkpoint_population_moments,
    flatten_parameter_gradients,
    pair_metrics,
    select_adv_state,
)
from experiments.advfd_cleanroom.audit_pmf_generator_component_gradients import (
    build_generator,
    make_noise_and_labels,
    sample_generator,
)


DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


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
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--cfg-omega", type=float, default=8.5)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def component_objectives(
    critic: torch.nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    *,
    ema_beta: float,
    epsilon: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    from experiments.advfd_cleanroom.temporal_gauge import (
        blend_anchor_with_batch,
        real_whitened_fd_components_from_stats,
    )

    with torch.no_grad():
        real_features, _ = critic(real_images)
        real_moments = blend_anchor_with_batch(
            *real_anchor,
            real_features,
            beta=ema_beta,
        )
    fake_features, _ = critic(fake_images)
    fake_moments = blend_anchor_with_batch(
        *fake_anchor,
        fake_features,
        beta=ema_beta,
    )
    mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
        *real_moments,
        *fake_moments,
        epsilon=epsilon,
    )
    components = {
        "mean": mean_term,
        "covariance": covariance_term,
        "full": mean_term + covariance_term,
    }
    return components, {name: float(value.detach()) for name, value in components.items()}


def gradient_trial(
    critic: torch.nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    *,
    ema_beta: float,
    epsilon: float,
) -> dict[str, Any]:
    parameters = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    gradients: dict[str, torch.Tensor] = {}
    raw_components: dict[str, float] | None = None
    for name in ("mean", "covariance", "full"):
        components, current_raw = component_objectives(
            critic,
            real_images,
            fake_images,
            real_anchor,
            fake_anchor,
            ema_beta=ema_beta,
            epsilon=epsilon,
        )
        gradients[name] = flatten_parameter_gradients(components[name], parameters)
        raw_components = current_raw
    assert raw_components is not None
    pairs = {}
    names = tuple(gradients)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            pairs[f"{first}_vs_{second}"] = pair_metrics(
                gradients[first].unsqueeze(0),
                gradients[second].unsqueeze(0),
            )
    additivity_residual = gradients["full"] - (
        gradients["mean"] + gradients["covariance"]
    )
    full_norm = torch.linalg.vector_norm(gradients["full"].double())
    return {
        "raw_components": raw_components,
        "component_fraction": {
            "mean": raw_components["mean"] / raw_components["full"],
            "covariance": raw_components["covariance"] / raw_components["full"],
        },
        "gradient_norms": {
            name: float(torch.linalg.vector_norm(gradient.double()))
            for name, gradient in gradients.items()
        },
        "full_gradient_additivity_relative_error": float(
            torch.linalg.vector_norm(additivity_residual.double())
            / full_norm.clamp_min(1e-30)
        ),
        "pairs": pairs,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 2 or args.trials < 1:
        raise ValueError("batch size must be >=2 and trials must be positive")
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))
    device = torch.device(args.device)

    from experiments.advfd_cleanroom.temporal_gauge import torch_population_moments
    from experiments.raev2_training_core import DeterministicImageNetPacked
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
            gradient_trial(
                critic,
                real_images,
                fake_images,
                real_anchor,
                fake_anchor,
                ema_beta=args.ema_beta,
                epsilon=args.whiten_eps,
            )
        )
        critic.zero_grad(set_to_none=True)
        print(f"completed trial {trial_index + 1}/{args.trials}", flush=True)

    result = {
        "protocol": "advfd_pmf_critic_component_gradient_audit_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "adapter_manifest": str(args.adapter_manifest.expanduser().resolve()),
        "batch_size": args.batch_size,
        "trials": len(trials),
        "interpretation_boundary": (
            "All components use the same checkpoint EMA anchors and paired "
            "real/fake images. Real features and their whitening frame are "
            "detached exactly as in the official D-step. Raw parameter "
            "gradients are compared before clipping and AdamW preconditioning."
        ),
        "aggregate_component_fraction": {
            name: aggregate_scalars(
                [{name: trial["component_fraction"][name]} for trial in trials]
            )[name]
            for name in ("mean", "covariance")
        },
        "aggregate_gradient_norms": {
            name: aggregate_scalars(
                [{name: trial["gradient_norms"][name]} for trial in trials]
            )[name]
            for name in ("mean", "covariance", "full")
        },
        "aggregate_full_gradient_additivity_relative_error": aggregate_scalars(
            [
                {
                    "relative_error": trial[
                        "full_gradient_additivity_relative_error"
                    ]
                }
                for trial in trials
            ]
        )["relative_error"],
        "aggregate_pairs": {
            pair_name: aggregate_scalars(
                [trial["pairs"][pair_name] for trial in trials]
            )
            for pair_name in trials[0]["pairs"]
        },
        "trials_detail": trials,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["aggregate_pairs"], indent=2))


if __name__ == "__main__":
    main()
