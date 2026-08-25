#!/usr/bin/env python3
"""Short frozen-estimator residual-score post-training pilot for pMF-B."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from experiments.advfd_cleanroom.audit_pmf_residual_estimators import (
    frechet_distance,
)
from experiments.advfd_cleanroom.audit_pmf_residual_generator_vjp import (
    EstimatorSpec,
    fixed_pmf_inputs,
    load_estimator,
    transformed_features,
)
from experiments.advfd_cleanroom.feature_extractors import (
    DifferentiableInception2048,
    generator_output_to_unit_interval,
)
from experiments.advfd_cleanroom.generators import (
    load_pmf_b16,
    pmf_one_step,
    pmf_state_dict_for_advfd,
)
from experiments.advfd_cleanroom.run_pmf_pilot import autocast_context
from experiments.run_residual_score_estimator_toy import (
    EstimatorBundle,
    estimate_field,
    parse_float_tuple,
    parse_int_tuple,
)


def load_shared_ensemble(
    estimator_root: Path,
    seeds: tuple[int, ...],
    *,
    dimension: int,
    sigmas: tuple[float, ...],
    device: torch.device,
) -> list[EstimatorBundle]:
    ensemble = []
    for seed in seeds:
        seed_root = estimator_root / f"seed{seed}"
        config = json.loads((seed_root / "config.json").read_text(encoding="utf-8"))
        checkpoint = seed_root / "checkpoints" / f"seed{seed}" / "shared_dsm.pt"
        spec = EstimatorSpec(
            seed=seed,
            method="shared_dsm",
            checkpoint=checkpoint,
            sigmas=sigmas,
        )
        bundle = load_estimator(
            spec,
            config=config,
            dimension=dimension,
            device=device,
        )
        if bundle.kind != "shared_dsm":
            raise ValueError(f"expected shared_dsm, got {bundle.kind}")
        ensemble.append(bundle)
    return ensemble


def ensemble_residual_field(
    ensemble: list[EstimatorBundle],
    features: torch.Tensor,
    sigmas: tuple[float, ...],
    *,
    noise_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    clean = features.detach().float()
    sigma_fields = []
    diagnostics: dict[str, float] = {}
    for sigma_index, sigma in enumerate(sigmas):
        generator = torch.Generator(device=clean.device).manual_seed(
            noise_seed + 10007 * sigma_index
        )
        noise = torch.randn(
            clean.shape,
            generator=generator,
            device=clean.device,
            dtype=clean.dtype,
        )
        states = clean + float(sigma) * noise
        sigma_batch = torch.full(
            (len(states),), float(sigma), device=clean.device, dtype=clean.dtype
        )
        with torch.no_grad():
            per_seed = [
                estimate_field(
                    bundle,
                    states,
                    sigma_batch,
                    create_graph=False,
                )
                for bundle in ensemble
            ]
        stacked = torch.stack(per_seed)
        field = stacked.mean(dim=0)
        sigma_fields.append(field)
        diagnostics[f"field_rms_sigma_{sigma:g}"] = float(
            field.square().mean().sqrt()
        )
        if len(per_seed) > 1:
            normalized = torch.nn.functional.normalize(
                stacked.double().flatten(1), dim=1
            )
            pairwise = normalized @ normalized.mT
            upper = torch.triu_indices(len(per_seed), len(per_seed), offset=1)
            diagnostics[f"seed_cosine_sigma_{sigma:g}"] = float(
                pairwise[upper[0], upper[1]].mean()
            )
    return torch.stack(sigma_fields).mean(dim=0), diagnostics


def apply_moment_tangent_projection(
    features: torch.Tensor,
    field: torch.Tensor,
    projection: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    normal = moment_normal_field(features, projection)
    tangent = field - normal
    full_norm = field.double().flatten(1).norm(dim=1).mean()
    tangent_norm = tangent.double().flatten(1).norm(dim=1).mean()
    normal_norm = normal.double().flatten(1).norm(dim=1).mean()
    return tangent, {
        "full_field_sample_norm": float(full_norm),
        "tangent_field_sample_norm": float(tangent_norm),
        "normal_field_sample_norm": float(normal_norm),
    }


def moment_normal_field(
    features: torch.Tensor,
    projection: dict[str, Any],
) -> torch.Tensor:
    clean = features.detach()
    source_mean = projection["source_mean"].to(clean)
    translation = projection["translation"].to(clean)
    symmetric_linear = projection["symmetric_linear"].to(clean)
    return translation + (clean - source_mean) @ symmetric_linear.mT


def vector_dot(
    left: list[torch.Tensor | None],
    right: list[torch.Tensor | None],
) -> torch.Tensor:
    terms = [
        (a.detach() * b.detach()).sum(dtype=torch.float64)
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None
    ]
    if not terms:
        raise ValueError("cannot take a dot product of empty parameter vectors")
    return torch.stack(terms).sum()


@torch.no_grad()
def projected_adam_step(
    parameters: list[torch.nn.Parameter],
    constraint_gradients: list[torch.Tensor | None],
    state: list[dict[str, torch.Tensor]],
    *,
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Apply Adam's actual step after projecting it off one constraint gradient."""

    if step <= 0:
        raise ValueError("Adam step must be positive")
    if len(parameters) != len(constraint_gradients) or len(parameters) != len(state):
        raise ValueError("parameter, constraint, and optimizer-state lengths differ")
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    directions: list[torch.Tensor | None] = []
    raw_gradients: list[torch.Tensor | None] = []
    for parameter, constraint, parameter_state in zip(
        parameters, constraint_gradients, state, strict=True
    ):
        gradient = parameter.grad
        raw_gradients.append(gradient)
        if gradient is None:
            directions.append(None)
            continue
        if constraint is None:
            raise ValueError("missing constraint gradient for a trainable parameter")
        exp_avg = parameter_state.setdefault("exp_avg", torch.zeros_like(parameter))
        exp_avg_sq = parameter_state.setdefault(
            "exp_avg_sq", torch.zeros_like(parameter)
        )
        exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        direction = (exp_avg / bias_correction1) / (
            (exp_avg_sq / bias_correction2).sqrt() + eps
        )
        directions.append(direction)

    constraint_norm_sq = vector_dot(constraint_gradients, constraint_gradients)
    unprojected_norm_sq = vector_dot(directions, directions)
    raw_gradient_norm_sq = vector_dot(raw_gradients, raw_gradients)
    dot_before = vector_dot(directions, constraint_gradients)
    coefficient = dot_before / constraint_norm_sq.clamp_min(1e-30)
    for index, (direction, constraint) in enumerate(
        zip(directions, constraint_gradients, strict=True)
    ):
        if direction is not None and constraint is not None:
            directions[index] = direction - coefficient.to(direction) * constraint
    projected_norm_sq = vector_dot(directions, directions)
    dot_after = vector_dot(directions, constraint_gradients)
    for parameter, direction in zip(parameters, directions, strict=True):
        if direction is not None:
            parameter.add_(direction, alpha=-learning_rate)
    cosine_before = dot_before / (
        unprojected_norm_sq.sqrt() * constraint_norm_sq.sqrt()
    ).clamp_min(1e-30)
    cosine_after = dot_after / (
        projected_norm_sq.sqrt() * constraint_norm_sq.sqrt()
    ).clamp_min(1e-30)
    return {
        "raw_parameter_gradient_norm": float(raw_gradient_norm_sq.sqrt()),
        "normal_parameter_gradient_norm": float(constraint_norm_sq.sqrt()),
        "adam_unprojected_direction_norm": float(unprojected_norm_sq.sqrt()),
        "adam_projected_direction_norm": float(projected_norm_sq.sqrt()),
        "adam_projection_coefficient": float(coefficient),
        "adam_constraint_cosine_before": float(cosine_before),
        "adam_constraint_cosine_after": float(cosine_after),
        "adam_removed_energy_fraction": float(
            1.0 - projected_norm_sq / unprojected_norm_sq.clamp_min(1e-30)
        ),
    }


def gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    norm_sq = sum(
        float(parameter.grad.detach().double().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return math.sqrt(norm_sq)


def evaluate_projected_fd(
    model: torch.nn.Module,
    encoder: DifferentiableInception2048,
    bank: dict[str, Any],
    *,
    count: int,
    batch_size: int,
    noise_seed: int,
    label_seed: int,
    amp: bool,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    if count > len(bank["real_heldout"]):
        raise ValueError("evaluation count exceeds held-out real feature bank")
    was_training = model.training
    model.eval()
    parts = []
    completed = 0
    while completed < count:
        batch = min(batch_size, count - completed)
        noise, labels = fixed_pmf_inputs(
            model,
            batch_size=batch,
            noise_seed=noise_seed + completed,
            label_seed=label_seed + completed,
            device=device,
        )
        with torch.no_grad(), autocast_context(device, amp):
            raw = pmf_one_step(model, noise, labels)
            images = generator_output_to_unit_interval(raw.float())
            features = transformed_features(encoder, images, bank, amp=amp)
        parts.append(features.cpu())
        completed += batch
    model.train(was_training)
    fake = torch.cat(parts)
    real = bank["real_heldout"][:count].float()
    return frechet_distance(real, fake), fake


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    step: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Save in the public AdvFD layout so the official evaluator can
            # consume the post-trained checkpoint without a permissive load.
            "model": pmf_state_dict_for_advfd(model.state_dict()),
            "step": step,
            "config": config,
            "model_state_format": "advfd_pmf_denoiser_v1",
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--estimator-root", type=Path, required=True)
    parser.add_argument(
        "--moment-projection",
        type=Path,
        help="Frozen moment-tangent projection fitted on the fake feature bank.",
    )
    parser.add_argument(
        "--moment-projection-space",
        choices=("feature", "parameter"),
        default="feature",
        help=(
            "Apply the frozen decomposition to feature fields, or project Adam's "
            "realized parameter update off the induced moment-normal gradient."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--pmf-repo",
        type=Path,
        default=Path("/data/users/zhoushunyu/research_repos/pMF"),
    )
    parser.add_argument(
        "--pmf-checkpoint",
        type=Path,
        default=Path(
            "/data/users/zhoushunyu/research_repos/FD-Loss-assets/pMF-B_256.pth"
        ),
    )
    parser.add_argument("--estimator-seeds", type=parse_int_tuple, default=(0, 1, 2))
    parser.add_argument(
        "--sigmas", type=parse_float_tuple, default=(0.1, 0.3, 0.7, 1.5)
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--update-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save the final model-only checkpoint; disabled for screening runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.steps <= 0
        or args.batch_size <= 0
        or args.gradient_accumulation <= 0
        or args.learning_rate <= 0
    ):
        raise ValueError("invalid training configuration")
    if not args.sigmas or any(sigma <= 0 for sigma in args.sigmas):
        raise ValueError("diffusive score training requires positive sigma values")
    if args.eval_every <= 0 or args.eval_samples <= 1:
        raise ValueError("invalid evaluation configuration")
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    device = torch.device(args.device)
    amp = not args.no_amp
    bank = torch.load(args.feature_bank, map_location="cpu", weights_only=False)
    if bank.get("protocol") != "frozen_pmf_b_inception64_residual_feature_bank_v1":
        raise ValueError("unexpected feature-bank protocol")
    model = load_pmf_b16(
        repo=args.pmf_repo, checkpoint=args.pmf_checkpoint, device=device
    )
    # The objective is defined on the deterministic inference map. Parameters
    # remain trainable even though stochastic training-mode behavior is disabled.
    model.eval().requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    ensemble = load_shared_ensemble(
        args.estimator_root,
        args.estimator_seeds,
        dimension=int(bank["projection"].shape[1]),
        sigmas=args.sigmas,
        device=device,
    )
    moment_projection = None
    if args.moment_projection is not None:
        moment_projection = torch.load(
            args.moment_projection, map_location="cpu", weights_only=False
        )
        if (
            moment_projection.get("protocol")
            != "pmf_inception64_moment_tangent_score_projection_v1"
        ):
            raise ValueError("unexpected moment projection protocol")
        if tuple(moment_projection["sigmas"]) != tuple(args.sigmas):
            raise ValueError("moment projection sigma schedule does not match")
        if tuple(moment_projection["estimator_seeds"]) != tuple(
            args.estimator_seeds
        ):
            raise ValueError("moment projection estimator seeds do not match")
    parameter_projection = (
        moment_projection is not None and args.moment_projection_space == "parameter"
    )
    optimizer = None
    adam_state: list[dict[str, torch.Tensor]] | None = None
    if parameter_projection:
        adam_state = [{} for _ in parameters]
    else:
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
    config = {
        **vars(args),
        "feature_bank": str(args.feature_bank),
        "estimator_root": str(args.estimator_root),
        "moment_projection": (
            str(args.moment_projection) if args.moment_projection else None
        ),
        "output_root": str(args.output_root),
        "pmf_repo": str(args.pmf_repo),
        "pmf_checkpoint": str(args.pmf_checkpoint),
        "protocol": (
            "pmf_b_frozen_shared_dsm_parameter_tangent_posttrain_pilot_v1"
            if parameter_projection
            else "pmf_b_frozen_shared_dsm_moment_tangent_posttrain_pilot_v1"
            if moment_projection is not None
            else "pmf_b_frozen_shared_dsm_residual_posttrain_pilot_v1"
        ),
        "objective": (
            "project Adam(full_score) orthogonal to the parameter gradient induced "
            "by the frozen moment-normal score component"
            if parameter_projection
            else "loss=-sign*E[stopgrad(P_moment_tangent(s_real-s_fake))^T "
            "whitened_inception64]"
            if moment_projection is not None
            else "loss=-sign*E[stopgrad(s_real-s_fake)^T whitened_inception64]"
        ),
        "estimator_refresh": False,
        "amp": amp,
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    def evaluate(step: int) -> None:
        fd, features = evaluate_projected_fd(
            model,
            encoder,
            bank,
            count=args.eval_samples,
            batch_size=args.eval_batch_size,
            noise_seed=args.seed + 900001,
            label_seed=args.seed + 900007,
            amp=amp,
            device=device,
        )
        torch.save(features, args.output_root / f"eval_features_step{step:06d}.pt")
        rows.append(
            {
                "record_type": "evaluation",
                "step": step,
                "projected_feature_fd": fd,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            }
        )
        pd.DataFrame(rows).to_csv(args.output_root / "metrics.csv", index=False)
        print(f"eval step={step} projected_feature_fd={fd:.6f}", flush=True)

    evaluate(0)
    for step in range(1, args.steps + 1):
        for parameter in parameters:
            parameter.grad = None
        normal_gradient_sums: list[torch.Tensor | None] | None = (
            [None for _ in parameters] if parameter_projection else None
        )
        loss_sum = 0.0
        image_rms_sum = 0.0
        diagnostic_sums: dict[str, float] = {}
        for accumulation_index in range(args.gradient_accumulation):
            micro_index = (step - 1) * args.gradient_accumulation + accumulation_index + 1
            noise, labels = fixed_pmf_inputs(
                model,
                batch_size=args.batch_size,
                noise_seed=args.seed + 1009 * micro_index,
                label_seed=args.seed + 2003 * micro_index,
                device=device,
            )
            # First obtain the image-space update while pMF is frozen.
            # Recomputing pMF below keeps the large pMF and Inception graphs out
            # of memory simultaneously without changing the detached-field VJP.
            with torch.no_grad(), autocast_context(device, amp):
                raw = pmf_one_step(model, noise, labels)
                baseline_images = generator_output_to_unit_interval(raw.float())
            image_leaf = baseline_images.detach().requires_grad_(True)
            features = transformed_features(encoder, image_leaf, bank, amp=amp)
            field, diagnostics = ensemble_residual_field(
                ensemble,
                features,
                args.sigmas,
                noise_seed=args.seed + 3001 * micro_index,
            )
            normal_field = None
            if moment_projection is not None and not parameter_projection:
                field, projection_diagnostics = apply_moment_tangent_projection(
                    features, field, moment_projection
                )
                diagnostics.update(projection_diagnostics)
            elif parameter_projection:
                assert moment_projection is not None
                normal_field = moment_normal_field(features, moment_projection)
                tangent_field = field - normal_field
                diagnostics.update(
                    {
                        "full_field_sample_norm": float(
                            field.double().flatten(1).norm(dim=1).mean()
                        ),
                        "tangent_field_sample_norm": float(
                            tangent_field.double().flatten(1).norm(dim=1).mean()
                        ),
                        "normal_field_sample_norm": float(
                            normal_field.double().flatten(1).norm(dim=1).mean()
                        ),
                    }
                )
            positive_surrogate = (features * field.detach()).sum(dim=1).mean()
            image_update = torch.autograd.grad(
                positive_surrogate,
                image_leaf,
                retain_graph=parameter_projection,
            )[0].detach()
            normal_image_update = None
            if parameter_projection:
                assert normal_field is not None
                normal_surrogate = (
                    features * normal_field.detach()
                ).sum(dim=1).mean()
                normal_image_update = torch.autograd.grad(
                    normal_surrogate, image_leaf
                )[0].detach()
                del normal_surrogate
            del (
                raw,
                baseline_images,
                image_leaf,
                features,
                field,
                normal_field,
                positive_surrogate,
            )

            with autocast_context(device, amp):
                raw = pmf_one_step(model, noise, labels)
                images = generator_output_to_unit_interval(raw.float())
                raw_loss = -args.update_sign * (images * image_update).sum()
                loss = raw_loss / args.gradient_accumulation
                normal_loss = (
                    -args.update_sign * (images * normal_image_update).sum()
                    / args.gradient_accumulation
                    if parameter_projection
                    else None
                )
            if parameter_projection:
                assert normal_loss is not None
                assert normal_gradient_sums is not None
                normal_gradients = torch.autograd.grad(
                    normal_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                for index, gradient in enumerate(normal_gradients):
                    if gradient is None:
                        continue
                    if normal_gradient_sums[index] is None:
                        normal_gradient_sums[index] = gradient.detach().clone()
                    else:
                        normal_gradient_sums[index].add_(gradient.detach())
            loss.backward()
            loss_sum += float(raw_loss.detach())
            image_rms_sum += float(image_update.float().square().mean().sqrt())
            for name, value in diagnostics.items():
                diagnostic_sums[name] = diagnostic_sums.get(name, 0.0) + value
            del (
                noise,
                labels,
                raw,
                images,
                raw_loss,
                loss,
                normal_loss,
                image_update,
                normal_image_update,
            )
        optimizer_diagnostics: dict[str, float] = {}
        if parameter_projection:
            assert normal_gradient_sums is not None
            assert adam_state is not None
            optimizer_diagnostics = projected_adam_step(
                parameters,
                normal_gradient_sums,
                adam_state,
                step=step,
                learning_rate=args.learning_rate,
            )
            parameter_gradient_norm = optimizer_diagnostics[
                "raw_parameter_gradient_norm"
            ]
        else:
            assert optimizer is not None
            parameter_gradient_norm = gradient_norm(parameters)
            optimizer.step()
        rows.append(
            {
                "record_type": "training",
                "step": step,
                "loss": loss_sum / args.gradient_accumulation,
                "image_update_coordinate_rms": image_rms_sum
                / args.gradient_accumulation,
                "parameter_gradient_norm": parameter_gradient_norm,
                "learning_rate": args.learning_rate,
                "update_sign": args.update_sign,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size
                * args.gradient_accumulation,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                **{
                    name: value / args.gradient_accumulation
                    for name, value in diagnostic_sums.items()
                },
                **optimizer_diagnostics,
            }
        )
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            pd.DataFrame(rows).to_csv(args.output_root / "metrics.csv", index=False)
            print(
                f"train step={step}/{args.steps} loss={rows[-1]['loss']:.5g} "
                f"grad={parameter_gradient_norm:.5g} "
                f"image_rms={rows[-1]['image_update_coordinate_rms']:.5g}",
                flush=True,
            )
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)

    if args.save_checkpoint:
        save_checkpoint(
            args.output_root / "checkpoint_final.pt",
            model=model,
            step=args.steps,
            config=config,
        )
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "steps": args.steps,
        "update_sign": args.update_sign,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
