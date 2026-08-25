#!/usr/bin/env python3
"""Compare AdvFD image gradients from historical EMA and current-frame moments."""

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
DEFAULT_OFFICIAL_ROOT = Path("/data/users/zhoushunyu/research_repos/AdvFD")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-moments", type=Path, required=True)
    parser.add_argument("--fake-moments", type=Path, required=True)
    parser.add_argument(
        "--packed-imagenet-root",
        type=Path,
        default=Path("/data/shared/imagenet-1k/random_access_v1"),
    )
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--parameter-gradient-trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument(
        "--refresh-steps",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Expected fixed-frame EMA refresh counts to audit. Each refresh "
            "interpolates historical and current uncentered moments with beta**n."
        ),
    )
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--fid-norm-eps", type=float, default=0.01)
    parser.add_argument("--cfg-omega", type=float, default=8.5)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=0.7)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def select_adv_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    states = checkpoint.get("fd_adv_states")
    if not isinstance(states, list) or len(states) != 1:
        raise ValueError("expected exactly one adaptive critic state")
    return states[0]


def checkpoint_population_moments(stats: dict[str, torch.Tensor]):
    from experiments.advfd_cleanroom.temporal_gauge import PopulationMoments

    mean = stats["mu_ema"].detach().double().cpu().numpy()
    second = stats["m2_ema"].detach().double().cpu().numpy()
    covariance = second - np.outer(mean, mean)
    return PopulationMoments(
        mean=mean,
        covariance=0.5 * (covariance + covariance.T),
        count=-1,
    )


def load_fresh_moment_variants(path: Path):
    from experiments.advfd_cleanroom.temporal_gauge import (
        PopulationMoments,
        merge_population_moments,
    )

    payload = np.load(path, allow_pickle=False)

    def load(prefix: str) -> PopulationMoments:
        return PopulationMoments(
            mean=np.asarray(payload[f"{prefix}_mean"], dtype=np.float64),
            covariance=np.asarray(
                payload[f"{prefix}_covariance"], dtype=np.float64
            ),
            count=int(payload[f"{prefix}_count"]),
        )

    first = load("fresh_a")
    second = load("fresh_b")
    return {"current_a": first, "current_b": second, "current_full": merge_population_moments(first, second)}


def pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first_flat = first.double().reshape(-1)
    second_flat = second.double().reshape(-1)
    first_norm = torch.linalg.vector_norm(first_flat)
    second_norm = torch.linalg.vector_norm(second_flat)
    cosine = torch.dot(first_flat, second_flat) / (
        first_norm * second_norm
    ).clamp_min(1e-30)
    difference = torch.linalg.vector_norm(first_flat - second_flat)
    first_samples = first.double().flatten(1)
    second_samples = second.double().flatten(1)
    sample_cosines = torch.nn.functional.cosine_similarity(
        first_samples, second_samples, dim=1
    )
    return {
        "cosine": float(cosine),
        "first_norm": float(first_norm),
        "second_norm": float(second_norm),
        "second_to_first_norm_ratio": float(second_norm / first_norm.clamp_min(1e-30)),
        "difference_to_first_norm_ratio": float(difference / first_norm.clamp_min(1e-30)),
        "per_sample_cosine_mean": float(sample_cosines.mean()),
        "per_sample_cosine_min": float(sample_cosines.min()),
        "per_sample_cosine_max": float(sample_cosines.max()),
    }


def gradient_concentration_metrics(gradient: torch.Tensor) -> dict[str, float]:
    """Summarize how much of a batch/image carries squared gradient energy."""

    squared = gradient.detach().double().square()
    sample_energy = squared.flatten(1).sum(dim=1)
    spatial_energy = squared.sum(dim=1).flatten()
    coordinate_energy = squared.flatten()

    def effective_fraction(energy: torch.Tensor) -> float:
        numerator = energy.sum().square()
        denominator = energy.numel() * energy.square().sum()
        return float(numerator / denominator.clamp_min(1e-30))

    sample_norm = sample_energy.sqrt()
    return {
        "sample_effective_fraction": effective_fraction(sample_energy),
        "spatial_effective_fraction": effective_fraction(spatial_energy),
        "coordinate_effective_fraction": effective_fraction(coordinate_energy),
        "sample_norm_min": float(sample_norm.min()),
        "sample_norm_median": float(sample_norm.median()),
        "sample_norm_max": float(sample_norm.max()),
        "sample_norm_max_to_median": float(
            sample_norm.max() / sample_norm.median().clamp_min(1e-30)
        ),
    }


def aggregate_scalars(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = records[0].keys()
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "std": float(np.std([record[key] for record in records])),
            "min": float(np.min([record[key] for record in records])),
            "max": float(np.max([record[key] for record in records])),
        }
        for key in keys
    }


def flatten_parameter_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        allow_unused=True,
    )
    chunks = [
        (
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().cpu().reshape(-1)
        )
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(chunks)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EQVAE_ROOT))
    device = torch.device("cuda")
    official_root = args.official_root.expanduser().resolve()
    sys.path.insert(0, str(official_root))
    from experiments.advfd_cleanroom.temporal_gauge import (
        blend_anchor_with_batch,
        interpolate_population_moments,
        real_whitened_fd_components_from_stats,
        torch_population_moments,
    )
    from experiments.raev2_training_core import DeterministicImageNetPacked
    from frechet_distance.repr_models import load_repr_model
    import models

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    adv_state = select_adv_state(checkpoint)
    real_anchors = {"ema": checkpoint_population_moments(adv_state["real_stats"])}
    fake_anchors = {"ema": checkpoint_population_moments(adv_state["fake_stats"])}
    real_anchors.update(load_fresh_moment_variants(args.real_moments))
    fake_anchors.update(load_fresh_moment_variants(args.fake_moments))
    refresh_names = []
    for refresh_step in sorted(set(args.refresh_steps)):
        if refresh_step < 0:
            raise ValueError("refresh steps must be nonnegative")
        refresh_name = f"refresh_{refresh_step:04d}"
        historical_weight = float(args.ema_beta**refresh_step)
        real_anchors[refresh_name] = interpolate_population_moments(
            real_anchors["ema"],
            real_anchors["current_full"],
            historical_weight=historical_weight,
        )
        fake_anchors[refresh_name] = interpolate_population_moments(
            fake_anchors["ema"],
            fake_anchors["current_full"],
            historical_weight=historical_weight,
        )
        refresh_names.append(refresh_name)

    critic, _, _, _ = load_repr_model("inception", device="cuda")
    critic.load_state_dict(adv_state["model"], strict=True)
    critic.eval().requires_grad_(False)
    del adv_state

    manifest = json.loads(
        args.adapter_manifest.expanduser().resolve().read_text(encoding="utf-8")
    )
    model_args = manifest["official_arguments"]
    model_name = str(model_args["model"])
    if model_name not in models.pMFDenoiser_models:
        raise ValueError(f"gradient audit currently expects pMF, got {model_name!r}")
    generator = models.pMFDenoiser_models[model_name](
        img_size=int(model_args["img_size"]),
        patch_size=int(model_args["patch_size"]),
        in_channels=int(model_args["token_channels"]),
        tokenizer_patch_size=int(model_args["tokenizer_patch_size"]),
        num_classes=int(model_args["num_classes"]),
        label_drop_prob=float(model_args["label_drop_prob"]),
        P_mean=float(model_args["P_mean"]),
        P_std=float(model_args["P_std"]),
        ratio_r_neq_t=float(model_args["ratio_r_neq_t"]),
        cfg_beta=float(model_args["cfg_beta"]),
        tr_uniform=bool(model_args["tr_uniform"]),
        cfg_omega_max=float(model_args["cfg_omega_max"]),
        aux_head_depth=int(model_args["aux_head_depth"]),
        class_tokens=int(model_args["class_tokens"]),
        time_tokens=int(model_args["time_tokens"]),
        guidance_tokens=int(model_args["guidance_tokens"]),
        interval_tokens=int(model_args["interval_tokens"]),
        t_eps=float(model_args["t_eps"]),
        perceptual_threshold=float(model_args["perceptual_threshold"]),
        perceptual_loss_on_aux=bool(model_args["perceptual_loss_on_aux"]),
        rope_2d=bool(model_args["rope_2d"]),
        learned_pe=bool(model_args["learned_pe"]),
        disable_v_head=bool(model_args["disable_v_head"]),
        noise_scale=float(model_args["noise_scale"]),
        norm_eps=float(model_args["norm_eps"]),
        norm_p=float(model_args["norm_p"]),
        grad_checkpointing=False,
    ).to(device)
    incompatible = generator.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"generator checkpoint mismatch: {incompatible}")
    generator.eval().requires_grad_(False)

    del checkpoint

    real_anchor_tensors = {
        name: torch_population_moments(moments, device=device)
        for name, moments in real_anchors.items()
    }
    fake_anchor_tensors = {
        name: torch_population_moments(moments, device=device)
        for name, moments in fake_anchors.items()
    }
    variant_anchor_names = {
        "ema": ("ema", "ema"),
        "current_real_only": ("current_full", "ema"),
        "current_fake_only": ("ema", "current_full"),
        "current_a": ("current_a", "current_a"),
        "current_b": ("current_b", "current_b"),
        "current_full": ("current_full", "current_full"),
    }
    variant_anchor_names.update(
        {name: (name, name) for name in refresh_names}
    )
    variants = tuple(variant_anchor_names)
    component_variants = ("ema", "current_full")

    dataset = DeterministicImageNetPacked(
        args.packed_imagenet_root,
        split="train",
        image_size=256,
        augmentation_seed=1,
        horizontal_flip=False,
    )
    rng = np.random.default_rng(args.seed + 91)
    real_indices = rng.choice(
        len(dataset), size=args.batch_size * args.trials, replace=False
    )
    real_loader = DataLoader(
        Subset(dataset, real_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    trials = []
    parameter_trial_inputs = []
    pair_names = (
        ("ema", "current_full"),
        ("ema", "current_real_only"),
        ("ema", "current_fake_only"),
        ("current_a", "current_b"),
        ("current_full", "current_a"),
        ("current_full", "current_b"),
        ("current_full", "current_real_only"),
        ("current_full", "current_fake_only"),
        *[("ema", name) for name in refresh_names],
        *[(name, "current_full") for name in refresh_names],
    )
    for trial_index, real_batch in enumerate(real_loader):
        real_images = real_batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            real_features, _ = critic(real_images)
            noise_generator = torch.Generator(device=device).manual_seed(
                args.seed + trial_index
            )
            noise = torch.randn(
                args.batch_size,
                generator.in_channels,
                generator.input_size,
                generator.input_size,
                generator=noise_generator,
                device=device,
            ) * float(generator.noise_scale)
            label_generator = torch.Generator(device=device).manual_seed(
                args.seed + 10_000 + trial_index
            )
            labels = torch.randint(
                0,
                1000,
                (args.batch_size,),
                generator=label_generator,
                device=device,
            )
            model_space = generator.sample_images_with_grad(
                noise,
                labels,
                sampling_args={
                    "cfg": args.cfg_omega,
                    "t_min": args.interval_min,
                    "t_max": args.interval_max,
                    "num_steps": 1,
                },
            )
            generated_images = model_space.mul(0.5).add(0.5)
        if trial_index < args.parameter_gradient_trials:
            parameter_trial_inputs.append(
                {
                    "trial": trial_index,
                    "real_features": real_features.detach().cpu(),
                    "noise_seed": args.seed + trial_index,
                    "label_seed": args.seed + 10_000 + trial_index,
                }
            )

        gradients: dict[str, torch.Tensor] = {}
        component_gradients: dict[str, dict[str, torch.Tensor]] = {}
        losses: dict[str, dict[str, float]] = {}
        for variant in variants:
            images = generated_images.detach().clone().requires_grad_(True)
            fake_features, _ = critic(images)
            real_anchor_name, fake_anchor_name = variant_anchor_names[variant]
            real_mu, real_covariance = blend_anchor_with_batch(
                *real_anchor_tensors[real_anchor_name],
                real_features,
                beta=args.ema_beta,
            )
            fake_mu, fake_covariance = blend_anchor_with_batch(
                *fake_anchor_tensors[fake_anchor_name],
                fake_features,
                beta=args.ema_beta,
            )
            mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
                real_mu,
                real_covariance,
                fake_mu,
                fake_covariance,
                epsilon=args.whiten_eps,
            )
            raw_loss = mean_term + covariance_term
            normalized_loss = raw_loss / (raw_loss.detach() + args.fid_norm_eps)
            retain_for_components = variant in component_variants
            gradient = torch.autograd.grad(
                normalized_loss,
                images,
                retain_graph=retain_for_components,
            )[0]
            if retain_for_components:
                denominator = raw_loss.detach() + args.fid_norm_eps
                mean_gradient = torch.autograd.grad(
                    mean_term / denominator,
                    images,
                    retain_graph=True,
                )[0]
                covariance_gradient = torch.autograd.grad(
                    covariance_term / denominator,
                    images,
                )[0]
                component_gradients[variant] = {
                    "mean": mean_gradient.detach().cpu(),
                    "covariance": covariance_gradient.detach().cpu(),
                }
            gradient = gradient.detach().cpu()
            gradients[variant] = gradient
            losses[variant] = {
                "raw_fd": float(raw_loss.detach()),
                "raw_mean_term": float(mean_term.detach()),
                "raw_covariance_term": float(covariance_term.detach()),
                "normalized_fd": float(normalized_loss.detach()),
                "image_gradient_norm": float(torch.linalg.vector_norm(gradient.double())),
                "gradient_concentration": gradient_concentration_metrics(gradient),
            }
            del (
                images,
                fake_features,
                mean_term,
                covariance_term,
                raw_loss,
                normalized_loss,
            )

        pairs = {
            f"{first}_vs_{second}": pair_metrics(
                gradients[first], gradients[second]
            )
            for first, second in pair_names
        }
        component_metrics = {}
        for variant in component_variants:
            mean_gradient = component_gradients[variant]["mean"]
            covariance_gradient = component_gradients[variant]["covariance"]
            component_metrics[variant] = {
                "mean_vs_covariance": pair_metrics(
                    mean_gradient,
                    covariance_gradient,
                ),
                "total_vs_component_sum": pair_metrics(
                    gradients[variant],
                    mean_gradient + covariance_gradient,
                ),
                "mean_concentration": gradient_concentration_metrics(
                    mean_gradient
                ),
                "covariance_concentration": gradient_concentration_metrics(
                    covariance_gradient
                ),
                "full_concentration": gradient_concentration_metrics(
                    gradients[variant]
                ),
            }
        component_cross_pairs = {
            "ema_mean_vs_current_mean": pair_metrics(
                component_gradients["ema"]["mean"],
                component_gradients["current_full"]["mean"],
            ),
            "ema_covariance_vs_current_covariance": pair_metrics(
                component_gradients["ema"]["covariance"],
                component_gradients["current_full"]["covariance"],
            ),
        }
        trials.append(
            {
                "trial": trial_index,
                "noise_seed": args.seed + trial_index,
                "label_seed": args.seed + 10_000 + trial_index,
                "generated_fraction_below_zero": float(
                    (generated_images < 0.0).float().mean()
                ),
                "generated_fraction_above_one": float(
                    (generated_images > 1.0).float().mean()
                ),
                "losses": losses,
                "pairs": pairs,
                "component_metrics": component_metrics,
                "component_cross_pairs": component_cross_pairs,
            }
        )
        print(
            f"trial {trial_index + 1}/{args.trials}: "
            f"EMA/current cosine={pairs['ema_vs_current_full']['cosine']:.6f}, "
            f"fresh A/B cosine={pairs['current_a_vs_current_b']['cosine']:.6f}",
            flush=True,
        )
        if trial_index + 1 >= args.trials:
            break

    aggregate_pairs = {
        pair_name: aggregate_scalars(
            [trial["pairs"][pair_name] for trial in trials]
        )
        for pair_name in trials[0]["pairs"]
    }
    aggregate_component_metrics = {
        variant: {
            pair_name: aggregate_scalars(
                [trial["component_metrics"][variant][pair_name] for trial in trials]
            )
            for pair_name in trials[0]["component_metrics"][variant]
        }
        for variant in component_variants
    }
    aggregate_component_cross_pairs = {
        pair_name: aggregate_scalars(
            [trial["component_cross_pairs"][pair_name] for trial in trials]
        )
        for pair_name in trials[0]["component_cross_pairs"]
    }
    parameter_trials = []
    if parameter_trial_inputs:
        generator.requires_grad_(True)
        parameters = [parameter for parameter in generator.parameters() if parameter.requires_grad]
        for trial_input in parameter_trial_inputs:
            gradients: dict[str, torch.Tensor] = {}
            losses: dict[str, float] = {}
            real_features = trial_input["real_features"].to(device)
            for variant in variants:
                real_anchor_name, fake_anchor_name = variant_anchor_names[variant]
                noise_generator = torch.Generator(device=device).manual_seed(
                    trial_input["noise_seed"]
                )
                noise = torch.randn(
                    args.batch_size,
                    generator.in_channels,
                    generator.input_size,
                    generator.input_size,
                    generator=noise_generator,
                    device=device,
                ) * float(generator.noise_scale)
                label_generator = torch.Generator(device=device).manual_seed(
                    trial_input["label_seed"]
                )
                labels = torch.randint(
                    0,
                    1000,
                    (args.batch_size,),
                    generator=label_generator,
                    device=device,
                )
                model_space = generator.sample_images_with_grad(
                    noise,
                    labels,
                    sampling_args={
                        "cfg": args.cfg_omega,
                        "t_min": args.interval_min,
                        "t_max": args.interval_max,
                        "num_steps": 1,
                    },
                )
                generated_images = model_space.mul(0.5).add(0.5)
                fake_features, _ = critic(generated_images)
                real_mu, real_covariance = blend_anchor_with_batch(
                    *real_anchor_tensors[real_anchor_name],
                    real_features,
                    beta=args.ema_beta,
                )
                fake_mu, fake_covariance = blend_anchor_with_batch(
                    *fake_anchor_tensors[fake_anchor_name],
                    fake_features,
                    beta=args.ema_beta,
                )
                mean_term, covariance_term, _ = real_whitened_fd_components_from_stats(
                    real_mu,
                    real_covariance,
                    fake_mu,
                    fake_covariance,
                    epsilon=args.whiten_eps,
                )
                raw_loss = mean_term + covariance_term
                normalized_loss = raw_loss / (raw_loss.detach() + args.fid_norm_eps)
                gradients[variant] = flatten_parameter_gradients(
                    normalized_loss, parameters
                )
                losses[variant] = float(raw_loss.detach())
                del (
                    noise,
                    labels,
                    model_space,
                    generated_images,
                    fake_features,
                    mean_term,
                    covariance_term,
                    raw_loss,
                    normalized_loss,
                )
                torch.cuda.empty_cache()
            pairs = {
                f"{first}_vs_{second}": pair_metrics(
                    gradients[first].unsqueeze(0), gradients[second].unsqueeze(0)
                )
                for first, second in pair_names
            }
            parameter_trials.append(
                {
                    "trial": int(trial_input["trial"]),
                    "parameter_count": int(gradients["ema"].numel()),
                    "raw_fd": losses,
                    "pairs": pairs,
                }
            )
            print(
                f"parameter trial {trial_input['trial'] + 1}: "
                f"EMA/current cosine={pairs['ema_vs_current_full']['cosine']:.6f}, "
                f"fresh A/B cosine={pairs['current_a_vs_current_b']['cosine']:.6f}",
                flush=True,
            )
            del gradients
        generator.requires_grad_(False)

    aggregate_parameter_pairs = (
        {
            pair_name: aggregate_scalars(
                [trial["pairs"][pair_name] for trial in parameter_trials]
            )
            for pair_name in parameter_trials[0]["pairs"]
        }
        if parameter_trials
        else {}
    )
    result = {
        "protocol": "advfd_temporal_gauge_image_gradient_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "adapter_manifest": str(args.adapter_manifest.expanduser().resolve()),
        "real_moments": str(args.real_moments.expanduser().resolve()),
        "fake_moments": str(args.fake_moments.expanduser().resolve()),
        "batch_size": int(args.batch_size),
        "trials": len(trials),
        "parameter_gradient_trials": len(parameter_trials),
        "ema_beta": float(args.ema_beta),
        "refresh_steps": sorted(set(args.refresh_steps)),
        "whiten_epsilon": float(args.whiten_eps),
        "fid_normalization_epsilon": float(args.fid_norm_eps),
        "gradient_target": "generated RGB image before clipping, after [-1,1] to [0,1] map",
        "trials_detail": trials,
        "aggregate_pairs": aggregate_pairs,
        "aggregate_component_metrics": aggregate_component_metrics,
        "aggregate_component_cross_pairs": aggregate_component_cross_pairs,
        "parameter_trials_detail": parameter_trials,
        "aggregate_parameter_pairs": aggregate_parameter_pairs,
        "interpretation_boundary": (
            "Image-gradient cosine measures the signal delivered at the generator output. "
            "It does not yet include the generator parameter-space Jacobian."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["aggregate_pairs"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
