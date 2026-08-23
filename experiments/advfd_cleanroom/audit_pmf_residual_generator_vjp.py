#!/usr/bin/env python3
"""Audit residual-score VJPs through a frozen pMF-B generator.

The estimator is never allowed to update pMF in this program.  For fixed pMF
noise and labels, the script computes the residual field in the established
whitened Inception-64 space, propagates it to pixels, and finally propagates it
to pMF parameters.  Independent estimator seeds are compared at all three
levels so that pointwise field noise is not confused with a stable shared-
parameter update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from experiments.advfd_cleanroom.feature_extractors import (
    DifferentiableInception2048,
    generator_output_to_unit_interval,
)
from experiments.advfd_cleanroom.generators import load_pmf_b16, pmf_one_step
from experiments.advfd_cleanroom.run_pmf_pilot import autocast_context
from experiments.residual_score_toy import (
    FactorizedDomainNoiseEstimator,
    RatioEstimator,
    SharedDomainNoiseEstimator,
)
from experiments.run_residual_score_estimator_toy import (
    DSM_METHODS,
    RATIO_METHODS,
    EstimatorBundle,
    estimate_field,
    parse_float_tuple,
    parse_int_tuple,
    parse_str_tuple,
)


@dataclass(frozen=True)
class EstimatorSpec:
    seed: int
    method: str
    checkpoint: Path
    sigmas: tuple[float, ...]


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.double().reshape(-1)
    right = second.double().reshape(-1)
    denominator = left.norm() * right.norm()
    return float(torch.dot(left, right) / denominator.clamp_min(1e-30))


def relative_l2(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.double().reshape(-1)
    right = second.double().reshape(-1)
    scale = 0.5 * (left.norm() + right.norm())
    return float((left - right).norm() / scale.clamp_min(1e-30))


def parameter_group(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[1] in {"shared_blocks", "aux_blocks"}:
        return ".".join(parts[:3])
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def stable_seed(name: str, base_seed: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + int(base_seed)) % (2**63 - 1)


def gradient_sketch(
    named_gradients: list[tuple[str, torch.Tensor | None, int]],
    *,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    """Return a reproducible coordinate sketch and exact norm diagnostics.

    Coordinates are sampled with replacement using the same name-derived RNG
    for every estimator condition.  Multiplication by ``sqrt(n / k)`` makes
    sketch inner products unbiased estimates of full parameter inner products.
    """

    if budget <= 0:
        raise ValueError("sketch budget must be positive")
    total = sum(numel for _, gradient, numel in named_gradients if gradient is not None)
    if total <= 0:
        raise ValueError("no parameter gradients were produced")
    sketches: list[torch.Tensor] = []
    group_sketches: dict[str, list[torch.Tensor]] = {}
    group_norm_sq: dict[str, float] = {}
    exact_norm_sq = 0.0
    sampled = 0
    for name, gradient, numel in named_gradients:
        if gradient is None:
            continue
        values = gradient.detach().float().reshape(-1).cpu()
        if not torch.isfinite(values).all():
            raise FloatingPointError(f"non-finite gradient in {name}")
        norm_sq = float(values.double().square().sum())
        exact_norm_sq += norm_sq
        group = parameter_group(name)
        group_norm_sq[group] = group_norm_sq.get(group, 0.0) + norm_sq
        count = max(1, int(round(budget * numel / total)))
        if count >= numel:
            indices = torch.arange(numel)
            count = numel
        else:
            generator = torch.Generator(device="cpu").manual_seed(
                stable_seed(name, seed)
            )
            indices = torch.randint(numel, (count,), generator=generator)
        part = values[indices] * math.sqrt(numel / count)
        sketches.append(part)
        group_sketches.setdefault(group, []).append(part)
        sampled += count
    return {
        "sketch": torch.cat(sketches),
        "group_sketches": {
            group: torch.cat(parts) for group, parts in group_sketches.items()
        },
        "exact_norm": math.sqrt(exact_norm_sq),
        "group_exact_norms": {
            group: math.sqrt(value) for group, value in group_norm_sq.items()
        },
        "sampled_coordinates": sampled,
        "parameter_coordinates": total,
    }


def load_estimator(
    spec: EstimatorSpec,
    *,
    config: dict[str, Any],
    dimension: int,
    device: torch.device,
) -> EstimatorBundle:
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    kind = str(payload["kind"])
    common = {
        "dimension": dimension,
        "hidden_dim": int(config["hidden_dim"]),
        "depth": int(config["depth"]),
        "frequencies": int(config["frequencies"]),
    }
    if kind in RATIO_METHODS:
        modules = {"ratio": RatioEstimator(**common)}
    elif kind == "shared_dsm":
        modules = {
            "shared": SharedDomainNoiseEstimator(
                **common, domain_dim=int(config["domain_dim"])
            )
        }
    elif kind in {"factorized_dsm", "factorized_dsm_coupled"}:
        modules = {"factorized": FactorizedDomainNoiseEstimator(**common)}
    else:
        raise ValueError(f"unsupported estimator kind for VJP audit: {kind}")
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name], strict=True)
        module.to(device).eval().requires_grad_(False)
    return EstimatorBundle(
        label=str(payload["label"]),
        kind=kind,
        modules=modules,
        history=[],
        elapsed_seconds=0.0,
        parameter_count=sum(
            parameter.numel()
            for module in modules.values()
            for parameter in module.parameters()
        ),
        sobolev_lambda=payload.get("sobolev_lambda"),
    )


def method_sigmas(method: str, requested: tuple[float, ...]) -> tuple[float, ...]:
    if method == "zero_ratio":
        return (0.0,)
    if not requested:
        raise ValueError("nonzero-noise estimators require at least one sigma")
    if any(sigma <= 0 for sigma in requested):
        raise ValueError("DSM/diffusive ratio sigma values must be positive")
    return requested


def build_specs(args: argparse.Namespace) -> tuple[list[EstimatorSpec], dict[int, dict]]:
    specs: list[EstimatorSpec] = []
    configs: dict[int, dict] = {}
    for seed in args.seeds:
        seed_root = args.estimator_root / f"seed{seed}"
        config = json.loads((seed_root / "config.json").read_text(encoding="utf-8"))
        configs[seed] = config
        for method in args.methods:
            checkpoint = seed_root / "checkpoints" / f"seed{seed}" / f"{method}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            specs.append(
                EstimatorSpec(
                    seed=seed,
                    method=method,
                    checkpoint=checkpoint,
                    sigmas=method_sigmas(method, args.sigmas),
                )
            )
    return specs, configs


def transformed_features(
    encoder: DifferentiableInception2048,
    images: torch.Tensor,
    bank: dict[str, Any],
    *,
    amp: bool,
) -> torch.Tensor:
    with autocast_context(images.device, amp):
        full = encoder(images)
    projection = bank["projection"].to(images.device, dtype=torch.float32)
    mean = bank["real_mean"].to(images.device, dtype=torch.float32)
    whitening = bank["whitening"].to(images.device, dtype=torch.float32)
    return ((full.float() @ projection) - mean) @ whitening


def integrated_field(
    bundle: EstimatorBundle,
    features: torch.Tensor,
    sigmas: tuple[float, ...],
    *,
    noise_seed: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    fields = []
    rms: dict[str, float] = {}
    clean = features.detach().float()
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
        states = (clean + float(sigma) * noise).requires_grad_(True)
        sigma_batch = torch.full(
            (len(states),), float(sigma), device=clean.device, dtype=clean.dtype
        )
        field = estimate_field(
            bundle,
            states,
            sigma_batch,
            create_graph=False,
        ).detach()
        fields.append(field)
        rms[f"sigma_{sigma:g}"] = float(field.square().mean().sqrt())
    return torch.stack(fields).mean(dim=0), rms


def fixed_pmf_inputs(
    model: torch.nn.Module,
    *,
    batch_size: int,
    noise_seed: int,
    label_seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    label_generator = torch.Generator(device=device).manual_seed(label_seed)
    noise = torch.randn(
        batch_size,
        model.img_channels,
        model.img_size,
        model.img_size,
        generator=noise_generator,
        device=device,
        dtype=torch.float32,
    ) * float(model.noise_scale)
    labels = torch.randint(
        0, 1000, (batch_size,), generator=label_generator, device=device
    )
    return noise, labels


def save_pairwise_summaries(output_root: Path, records: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for (method, trial), group in records.groupby(["method", "trial"]):
        for (_, left), (_, right) in combinations(group.sort_values("seed").iterrows(), 2):
            left_data = torch.load(left["artifact"], map_location="cpu", weights_only=False)
            right_data = torch.load(right["artifact"], map_location="cpu", weights_only=False)
            rows.append(
                {
                    "method": method,
                    "trial": trial,
                    "left_seed": int(left["seed"]),
                    "right_seed": int(right["seed"]),
                    "feature_field_cosine": cosine(
                        left_data["feature_field"], right_data["feature_field"]
                    ),
                    "feature_field_relative_l2": relative_l2(
                        left_data["feature_field"], right_data["feature_field"]
                    ),
                    "image_vjp_cosine": cosine(
                        left_data["image_update"], right_data["image_update"]
                    ),
                    "image_vjp_relative_l2": relative_l2(
                        left_data["image_update"], right_data["image_update"]
                    ),
                    "parameter_vjp_sketch_cosine": cosine(
                        left_data["parameter_sketch"],
                        right_data["parameter_sketch"],
                    ),
                    "parameter_vjp_sketch_relative_l2": relative_l2(
                        left_data["parameter_sketch"],
                        right_data["parameter_sketch"],
                    ),
                }
            )
            common_groups = sorted(
                set(left_data["group_sketches"]) & set(right_data["group_sketches"])
            )
            for parameter_group_name in common_groups:
                group_rows.append(
                    {
                        "method": method,
                        "trial": trial,
                        "left_seed": int(left["seed"]),
                        "right_seed": int(right["seed"]),
                        "parameter_group": parameter_group_name,
                        "sketch_cosine": cosine(
                            left_data["group_sketches"][parameter_group_name],
                            right_data["group_sketches"][parameter_group_name],
                        ),
                        "left_exact_norm": left_data["group_exact_norms"].get(
                            parameter_group_name, 0.0
                        ),
                        "right_exact_norm": right_data["group_exact_norms"].get(
                            parameter_group_name, 0.0
                        ),
                    }
                )
    pairwise = pd.DataFrame(rows)
    pairwise.to_csv(output_root / "pairwise_vjp.csv", index=False)
    if not pairwise.empty:
        pairwise.groupby(["method"])[
            [
                "feature_field_cosine",
                "image_vjp_cosine",
                "parameter_vjp_sketch_cosine",
            ]
        ].agg(["mean", "std", "min", "max"]).to_csv(
            output_root / "pairwise_vjp_summary.csv"
        )
    pd.DataFrame(group_rows).to_csv(
        output_root / "pairwise_parameter_groups.csv", index=False
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--estimator-root", type=Path, required=True)
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
    parser.add_argument(
        "--methods",
        type=parse_str_tuple,
        default=("zero_ratio", "sobolev_ratio_lam0.1", "shared_dsm"),
    )
    parser.add_argument("--seeds", type=parse_int_tuple, default=(0, 1, 2))
    parser.add_argument(
        "--sigmas", type=parse_float_tuple, default=(0.1, 0.3, 0.7, 1.5)
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sketch-coordinates", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trials <= 0 or args.batch_size <= 0 or args.sketch_coordinates <= 0:
        raise ValueError("trials, batch size and sketch size must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    specs, configs = build_specs(args)
    bank = torch.load(args.feature_bank, map_location="cpu", weights_only=False)
    if bank.get("protocol") != "frozen_pmf_b_inception64_residual_feature_bank_v1":
        raise ValueError("unexpected feature-bank protocol")
    dimension = int(bank["projection"].shape[1])
    device = torch.device(args.device)
    model = load_pmf_b16(
        repo=args.pmf_repo, checkpoint=args.pmf_checkpoint, device=device
    )
    model.eval().requires_grad_(True)
    encoder = DifferentiableInception2048(trainable=False).to(device).eval()
    parameters = [(name, parameter) for name, parameter in model.named_parameters()]
    config_payload = {
        **vars(args),
        "feature_bank": str(args.feature_bank),
        "estimator_root": str(args.estimator_root),
        "output_root": str(args.output_root),
        "pmf_repo": str(args.pmf_repo),
        "pmf_checkpoint": str(args.pmf_checkpoint),
        "protocol": "frozen_pmf_b_residual_generator_vjp_v1",
        "update_sign": "positive VJP J_theta^T(s_real-s_fake)",
    }
    (args.output_root / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=str) + "\n", encoding="utf-8"
    )
    bundles = {
        (spec.seed, spec.method): load_estimator(
            spec,
            config=configs[spec.seed],
            dimension=dimension,
            device=device,
        )
        for spec in specs
    }
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for trial in range(args.trials):
        noise_seed = args.seed + 1009 * trial
        label_seed = args.seed + 2003 * trial
        feature_noise_seed = args.seed + 3001 * trial
        noise, labels = fixed_pmf_inputs(
            model,
            batch_size=args.batch_size,
            noise_seed=noise_seed,
            label_seed=label_seed,
            device=device,
        )
        with torch.no_grad(), autocast_context(device, not args.no_amp):
            baseline_raw = pmf_one_step(model, noise, labels)
            baseline_images = generator_output_to_unit_interval(
                baseline_raw.float()
            )
        image_leaf = baseline_images.detach().requires_grad_(True)
        features = transformed_features(
            encoder, image_leaf, bank, amp=not args.no_amp
        )
        condition_updates: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, dict]] = {}
        for condition_index, spec in enumerate(specs):
            bundle = bundles[(spec.seed, spec.method)]
            field, sigma_rms = integrated_field(
                bundle,
                features,
                spec.sigmas,
                noise_seed=feature_noise_seed,
            )
            surrogate = (features * field.detach()).sum(dim=1).mean()
            image_update = torch.autograd.grad(
                surrogate,
                image_leaf,
                retain_graph=condition_index + 1 < len(specs),
            )[0].detach()
            condition_updates[(spec.seed, spec.method)] = (
                field.cpu(),
                image_update.cpu(),
                sigma_rms,
            )
        del features, image_leaf, baseline_images, baseline_raw
        for condition_index, spec in enumerate(specs):
            condition_started = time.perf_counter()
            feature_field, image_update_cpu, sigma_rms = condition_updates[
                (spec.seed, spec.method)
            ]
            image_update = image_update_cpu.to(device)
            with autocast_context(device, not args.no_amp):
                raw = pmf_one_step(model, noise, labels)
                images = generator_output_to_unit_interval(raw.float())
                parameter_surrogate = (images * image_update).sum()
            gradients = torch.autograd.grad(
                parameter_surrogate,
                [parameter for _, parameter in parameters],
                allow_unused=True,
            )
            named_gradients = [
                (name, gradient, parameter.numel())
                for (name, parameter), gradient in zip(parameters, gradients)
            ]
            sketch = gradient_sketch(
                named_gradients,
                budget=args.sketch_coordinates,
                seed=args.seed + 7919,
            )
            artifact = (
                args.output_root
                / "artifacts"
                / spec.method
                / f"seed{spec.seed}_trial{trial}.pt"
            )
            artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "seed": spec.seed,
                    "method": spec.method,
                    "trial": trial,
                    "sigmas": spec.sigmas,
                    "feature_field": feature_field,
                    "image_update": image_update_cpu,
                    "parameter_sketch": sketch["sketch"],
                    "group_sketches": sketch["group_sketches"],
                    "parameter_exact_norm": sketch["exact_norm"],
                    "group_exact_norms": sketch["group_exact_norms"],
                    "sampled_coordinates": sketch["sampled_coordinates"],
                    "parameter_coordinates": sketch["parameter_coordinates"],
                },
                artifact,
            )
            rows.append(
                {
                    "seed": spec.seed,
                    "method": spec.method,
                    "trial": trial,
                    "sigmas": "+".join(f"{sigma:g}" for sigma in spec.sigmas),
                    "feature_field_coordinate_rms": float(
                        feature_field.float().square().mean().sqrt()
                    ),
                    "image_vjp_coordinate_rms": float(
                        image_update_cpu.float().square().mean().sqrt()
                    ),
                    "parameter_vjp_exact_norm": sketch["exact_norm"],
                    "sampled_coordinates": sketch["sampled_coordinates"],
                    "parameter_coordinates": sketch["parameter_coordinates"],
                    "condition_seconds": time.perf_counter() - condition_started,
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device)
                    / 2**30,
                    "artifact": str(artifact),
                    **sigma_rms,
                }
            )
            pd.DataFrame(rows).to_csv(args.output_root / "vjp_metrics.csv", index=False)
            print(
                f"trial={trial} seed={spec.seed} method={spec.method} "
                f"field_rms={rows[-1]['feature_field_coordinate_rms']:.4g} "
                f"param_norm={sketch['exact_norm']:.4g} "
                f"elapsed={time.perf_counter() - condition_started:.1f}s",
                flush=True,
            )
            del raw, images, parameter_surrogate, gradients, named_gradients, image_update
        del condition_updates, noise, labels
        torch.cuda.empty_cache()
    metrics = pd.DataFrame(rows)
    save_pairwise_summaries(args.output_root, metrics)
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "conditions": len(specs),
        "trials": args.trials,
        "parameter_count": sum(parameter.numel() for _, parameter in parameters),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
