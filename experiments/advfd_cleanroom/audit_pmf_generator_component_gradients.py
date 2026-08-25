#!/usr/bin/env python3
"""Compare AdvFD full- and mean-component generator gradients on paired inputs.

The training objective normalizes its selected discrepancy by a detached copy of
that discrepancy.  Consequently, changing the selected component changes both
the gradient direction and its scalar normalization.  This audit reports the
two effects separately at the generated-image and generator-parameter levels.
"""

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
    checkpoint_population_moments,
    flatten_parameter_gradients,
    gradient_concentration_metrics,
    pair_metrics,
    select_adv_state,
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--parameter-gradient-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ema-beta", type=float, default=0.99)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    parser.add_argument("--fid-norm-eps", type=float, default=0.01)
    parser.add_argument("--cfg-omega", type=float, default=8.5)
    parser.add_argument("--interval-min", type=float, default=0.1)
    parser.add_argument("--interval-max", type=float, default=0.7)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def aggregate_scalars(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "std": float(np.std([record[key] for record in records])),
            "min": float(np.min([record[key] for record in records])),
            "max": float(np.max([record[key] for record in records])),
        }
        for key in records[0]
    }


def build_generator(models: Any, model_args: dict[str, Any], device: torch.device):
    model_name = str(model_args["model"])
    if model_name not in models.pMFDenoiser_models:
        raise ValueError(f"expected a pMF checkpoint, got {model_name!r}")
    return models.pMFDenoiser_models[model_name](
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


def make_noise_and_labels(
    generator: torch.nn.Module,
    batch_size: int,
    *,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        batch_size,
        generator.in_channels,
        generator.input_size,
        generator.input_size,
        generator=noise_generator,
        device=device,
    ) * float(generator.noise_scale)
    label_generator = torch.Generator(device=device).manual_seed(seed + 10_000)
    labels = torch.randint(
        0,
        1000,
        (batch_size,),
        generator=label_generator,
        device=device,
    )
    return noise, labels


def sample_generator(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
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
    return model_space.mul(0.5).add(0.5)


def component_losses(
    images: torch.Tensor,
    real_features: torch.Tensor,
    critic: torch.nn.Module,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    from experiments.advfd_cleanroom.temporal_gauge import (
        blend_anchor_with_batch,
        real_whitened_fd_components_from_stats,
    )

    fake_features, _ = critic(images)
    real_mu, real_covariance = blend_anchor_with_batch(
        *real_anchor,
        real_features,
        beta=args.ema_beta,
    )
    fake_mu, fake_covariance = blend_anchor_with_batch(
        *fake_anchor,
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
    full_term = mean_term + covariance_term
    full_denominator = full_term.detach() + args.fid_norm_eps
    mean_denominator = mean_term.detach() + args.fid_norm_eps
    covariance_denominator = covariance_term.detach() + args.fid_norm_eps
    losses = {
        "full_own_denominator": full_term / full_denominator,
        "mean_full_denominator": mean_term / full_denominator,
        "covariance_full_denominator": covariance_term / full_denominator,
        "mean_own_denominator": mean_term / mean_denominator,
        "covariance_own_denominator": covariance_term / covariance_denominator,
    }
    scalars = {
        "raw_full": float(full_term.detach()),
        "raw_mean": float(mean_term.detach()),
        "raw_covariance": float(covariance_term.detach()),
        "full_to_mean_denominator_ratio": float(
            full_denominator / mean_denominator
        ),
        "full_to_covariance_denominator_ratio": float(
            full_denominator / covariance_denominator
        ),
    }
    return losses, scalars


def image_gradient_trial(
    generated_images: torch.Tensor,
    real_features: torch.Tensor,
    critic: torch.nn.Module,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gradients = {}
    scalars = None
    names = (
        "full_own_denominator",
        "mean_full_denominator",
        "covariance_full_denominator",
        "mean_own_denominator",
        "covariance_own_denominator",
    )
    for name in names:
        images = generated_images.detach().clone().requires_grad_(True)
        losses, current_scalars = component_losses(
            images,
            real_features,
            critic,
            real_anchor,
            fake_anchor,
            args,
        )
        gradients[name] = torch.autograd.grad(losses[name], images)[0].detach().cpu()
        scalars = current_scalars
    pairs = {}
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            pairs[f"{first}_vs_{second}"] = pair_metrics(
                gradients[first], gradients[second]
            )
    return {
        "raw_terms": scalars,
        "gradient_metrics": {
            name: {
                "norm": float(torch.linalg.vector_norm(gradient.double())),
                "concentration": gradient_concentration_metrics(gradient),
            }
            for name, gradient in gradients.items()
        },
        "pairs": pairs,
    }


def parameter_gradient_trial(
    generator: torch.nn.Module,
    noise: torch.Tensor,
    labels: torch.Tensor,
    real_features: torch.Tensor,
    critic: torch.nn.Module,
    real_anchor: tuple[torch.Tensor, torch.Tensor],
    fake_anchor: tuple[torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    parameters = [parameter for parameter in generator.parameters() if parameter.requires_grad]
    gradients = {}
    scalars = None
    # Denominator changes only scale a component gradient, so two exact update
    # directions are sufficient for the parameter-space causal comparison.
    for name in ("full_own_denominator", "mean_own_denominator"):
        images = sample_generator(generator, noise, labels, args)
        losses, current_scalars = component_losses(
            images,
            real_features,
            critic,
            real_anchor,
            fake_anchor,
            args,
        )
        gradients[name] = flatten_parameter_gradients(losses[name], parameters)
        scalars = current_scalars
        del images, losses
        torch.cuda.empty_cache()
    pair = pair_metrics(
        gradients["full_own_denominator"].unsqueeze(0),
        gradients["mean_own_denominator"].unsqueeze(0),
    )
    return {"raw_terms": scalars, "full_vs_mean": pair}


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(EQVAE_ROOT))
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
    critic.eval().requires_grad_(False)

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
    indices = rng.choice(
        len(dataset), size=args.batch_size * args.trials, replace=False
    )
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    image_trials = []
    parameter_trials = []
    for trial_index, real_batch in enumerate(loader):
        real_images = real_batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            real_features, _ = critic(real_images)
            noise, labels = make_noise_and_labels(
                generator,
                args.batch_size,
                seed=args.seed + trial_index,
                device=device,
            )
            generated_images = sample_generator(generator, noise, labels, args)
        image_trials.append(
            image_gradient_trial(
                generated_images,
                real_features,
                critic,
                real_anchor,
                fake_anchor,
                args,
            )
        )
        if trial_index < args.parameter_gradient_trials:
            generator.requires_grad_(True)
            parameter_trials.append(
                parameter_gradient_trial(
                    generator,
                    noise,
                    labels,
                    real_features,
                    critic,
                    real_anchor,
                    fake_anchor,
                    args,
                )
            )
            generator.requires_grad_(False)
        print(f"completed trial {trial_index + 1}/{args.trials}", flush=True)

    pair_names = image_trials[0]["pairs"]
    aggregate_image_pairs = {
        name: aggregate_scalars([trial["pairs"][name] for trial in image_trials])
        for name in pair_names
    }
    gradient_names = image_trials[0]["gradient_metrics"]
    aggregate_image_norms = {
        name: aggregate_scalars(
            [
                {"norm": trial["gradient_metrics"][name]["norm"]}
                for trial in image_trials
            ]
        )["norm"]
        for name in gradient_names
    }
    result = {
        "protocol": "advfd_pmf_generator_component_gradient_audit_v1",
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "adapter_manifest": str(args.adapter_manifest.expanduser().resolve()),
        "batch_size": args.batch_size,
        "trials": len(image_trials),
        "parameter_gradient_trials": len(parameter_trials),
        "normalization_note": (
            "Official full-G uses full_own_denominator; the decoupled run uses "
            "mean_own_denominator. mean_full_denominator isolates component "
            "direction under the official full scalar denominator."
        ),
        "image_trials": image_trials,
        "aggregate_image_pairs": aggregate_image_pairs,
        "aggregate_image_norms": aggregate_image_norms,
        "parameter_trials": parameter_trials,
        "aggregate_parameter_full_vs_mean": (
            aggregate_scalars(
                [trial["full_vs_mean"] for trial in parameter_trials]
            )
            if parameter_trials
            else {}
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "aggregate_image_norms": aggregate_image_norms,
        "full_vs_mean_image": aggregate_image_pairs[
            "full_own_denominator_vs_mean_own_denominator"
        ],
        "full_vs_mean_parameter": result["aggregate_parameter_full_vs_mean"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
