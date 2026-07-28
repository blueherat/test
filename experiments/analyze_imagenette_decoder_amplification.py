"""Diagnose whether a frozen decoder amplifies latent-prior mismatch directions.

The intervention is performed in the centered, unit-RMS condition embedding that
the pixel decoder actually consumes.  Encoder, decoder, and latent prior remain
frozen throughout the analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_imagenette_latent_prior_tradeoff import (  # noqa: E402
    condition_embeddings,
    distribution_comparison,
    load_run_config,
)
from experiments.imagenette_latent_prior_tradeoff import (  # noqa: E402
    INTERFACE_DIM,
    OrthogonalLatentInterface,
    ResNet18Evaluator,
    build_prior,
    fixed_orthogonal_basis,
    load_frozen_models,
    sample_prior_coordinates,
    state_dict_sha256,
)
from experiments.mnist_spectral_rollout_toy import (  # noqa: E402
    configure_fp32,
    descending_time_grid,
    sinusoidal_time_embedding,
)


DEFAULT_ROOT = Path.home() / "data/eqvae/imagenette_latent_prior_tradeoff"
FIXED_ANGLE = 0.15
PROBE_TIMES = (0.9, 0.5, 0.1)
BRANCHES = ("prior_direction", "empirical_direction", "random_direction")


def project_condition_sphere(value: torch.Tensor) -> torch.Tensor:
    """Project rows to the decoder's zero-mean, unit-RMS condition sphere."""
    centered = value - value.mean(dim=1, keepdim=True)
    return centered * torch.rsqrt(
        centered.square().mean(dim=1, keepdim=True).clamp_min(1e-12)
    )


def condition_angles(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_unit = F.normalize(left.double(), dim=1)
    right_unit = F.normalize(right.double(), dim=1)
    cosine = (left_unit * right_unit).sum(dim=1).clamp(-1.0, 1.0)
    return cosine.acos().float()


def hungarian_sphere_match(
    base: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target rows optimally assigned to base rows by cosine distance."""
    if base.ndim != 2 or target.ndim != 2 or base.shape != target.shape:
        raise ValueError("Hungarian sphere matching requires equal rank-two tensors")
    cosine = F.normalize(base.double(), dim=1) @ F.normalize(target.double(), dim=1).T
    rows, columns = linear_sum_assignment((1.0 - cosine).cpu().numpy())
    expected = np.arange(len(base))
    if not np.array_equal(rows, expected):
        raise RuntimeError("Hungarian assignment did not preserve base row order")
    indices = torch.as_tensor(columns, dtype=torch.long)
    return target[indices], indices


def tangent_toward(base: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Unit tangent at each base point pointing along the target geodesic."""
    base_unit = F.normalize(base.double(), dim=1)
    target_unit = F.normalize(target.double(), dim=1)
    projection = (base_unit * target_unit).sum(dim=1, keepdim=True)
    tangent = target_unit - projection * base_unit
    norm = tangent.norm(dim=1, keepdim=True)
    if bool((norm < 1e-8).any()):
        raise RuntimeError("matched condition pair has an undefined sphere tangent")
    return (tangent / norm).float()


def random_tangent(base: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(base.shape, generator=generator, dtype=torch.float64)
    random = random - random.mean(dim=1, keepdim=True)
    base_unit = F.normalize(base.double(), dim=1)
    random = random - (random * base_unit).sum(dim=1, keepdim=True) * base_unit
    norm = random.norm(dim=1, keepdim=True)
    if bool((norm < 1e-8).any()):
        raise RuntimeError("random sphere tangent is numerically degenerate")
    return (random / norm).float()


def geodesic_step(
    base: torch.Tensor,
    tangent: torch.Tensor,
    angle: float,
) -> torch.Tensor:
    radius = math.sqrt(base.shape[1])
    base_unit = F.normalize(base.double(), dim=1)
    tangent_unit = F.normalize(tangent.double(), dim=1)
    stepped = radius * (
        math.cos(float(angle)) * base_unit + math.sin(float(angle)) * tangent_unit
    )
    return project_condition_sphere(stepped.float())


def decoder_forward_with_embedding(
    model: torch.nn.Module,
    value: torch.Tensor,
    time: torch.Tensor,
    condition_embedding: torch.Tensor,
) -> torch.Tensor:
    """Run ImagenetteConditionalUNet after its condition_mlp interface."""
    embedding = model.time_mlp(
        sinusoidal_time_embedding(time, model.embedding_dim)
    ) + condition_embedding
    skip0 = model._run(model.down0, model.input(value), embedding)
    skip1 = model._run(model.down1, model.downsample0(skip0), embedding)
    skip2 = model._run(model.down2, model.downsample1(skip1), embedding)
    hidden = model._run(model.middle, model.downsample2(skip2), embedding)

    hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="nearest")
    hidden = model.upsample2(hidden)
    hidden = model._run(model.up2, torch.cat([hidden, skip2], dim=1), embedding)
    hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="nearest")
    hidden = model.upsample1(hidden)
    hidden = model._run(model.up1, torch.cat([hidden, skip1], dim=1), embedding)
    hidden = F.interpolate(hidden, size=skip0.shape[-2:], mode="nearest")
    hidden = model.upsample0(hidden)
    hidden = model._run(model.up0, torch.cat([hidden, skip0], dim=1), embedding)
    return model.output(F.silu(model.output_norm(hidden)))


@torch.no_grad()
def sample_with_embedding(
    model: torch.nn.Module,
    initial: torch.Tensor,
    embedding: torch.Tensor,
    steps: int,
    *,
    record_times: tuple[float, ...] = (),
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    state = initial.clone()
    grid = descending_time_grid(int(steps), device=state.device)
    selected = {
        float(target): int(torch.argmin((grid[:-1] - float(target)).abs()))
        for target in record_times
    }
    recorded: dict[float, torch.Tensor] = {}
    for step_index, (current, following) in enumerate(zip(grid[:-1], grid[1:])):
        for target, selected_index in selected.items():
            if step_index == selected_index:
                recorded[target] = state.clone()
        time = torch.full((len(state),), float(current), device=state.device)
        velocity = decoder_forward_with_embedding(model, state, time, embedding)
        state = state + (following - current) * velocity
    return state.clamp(-1.0, 1.0), recorded


def two_sample_accuracy(real: torch.Tensor, generated: torch.Tensor, seed: int) -> dict[str, float]:
    features = torch.cat([real, generated]).double().numpy()
    labels = np.concatenate(
        [np.zeros(len(real), dtype=np.int64), np.ones(len(generated), dtype=np.int64)]
    )
    split = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(seed))
    linear = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2_000, random_state=int(seed)),
    )
    nonlinear = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64,),
            alpha=1e-3,
            batch_size="auto",
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            random_state=int(seed),
        ),
    )
    return {
        "condition_linear_c2st_accuracy": float(
            cross_val_score(linear, features, labels, cv=split, scoring="accuracy").mean()
        ),
        "condition_mlp_c2st_accuracy": float(
            cross_val_score(nonlinear, features, labels, cv=split, scoring="accuracy").mean()
        ),
    }


def _mean_sem(values: torch.Tensor) -> tuple[float, float]:
    values = values.double().flatten()
    mean = float(values.mean())
    sem = float(values.std(unbiased=True) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, sem


@torch.no_grad()
def decoder_response_metrics(
    decoder: torch.nn.Module,
    evaluator: torch.nn.Module,
    embeddings: dict[str, torch.Tensor],
    *,
    image_size: int,
    ode_steps: int,
    batch_size: int,
    pixel_seed: int,
) -> dict[str, float]:
    if "base" not in embeddings:
        raise ValueError("embeddings must contain a base branch")
    count = len(embeddings["base"])
    if any(len(value) != count for value in embeddings.values()):
        raise ValueError("all embedding branches must contain the same sample count")
    device = next(decoder.parameters()).device
    generator = torch.Generator(device=device).manual_seed(int(pixel_seed))
    sample_values: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"pixel_rms": [], "feature_rms": [], "feature_cosine": []}
        for name in embeddings
        if name != "base"
    }
    velocity_values: dict[str, dict[float, list[torch.Tensor]]] = {
        name: {time: [] for time in PROBE_TIMES}
        for name in BRANCHES
        if name in embeddings
    }

    for start in range(0, count, int(batch_size)):
        end = min(start + int(batch_size), count)
        initial = torch.randn(
            (end - start, 3, int(image_size), int(image_size)),
            generator=generator,
            device=device,
        )
        local = {
            name: value[start:end].to(device, non_blocking=True)
            for name, value in embeddings.items()
        }
        base_image, trajectory = sample_with_embedding(
            decoder,
            initial,
            local["base"],
            int(ode_steps),
            record_times=PROBE_TIMES,
        )
        base_feature, _ = evaluator(base_image)
        for name, embedding in local.items():
            if name == "base":
                continue
            image, _ = sample_with_embedding(
                decoder, initial, embedding, int(ode_steps)
            )
            feature, _ = evaluator(image)
            sample_values[name]["pixel_rms"].append(
                (image - base_image).flatten(1).square().mean(dim=1).sqrt().cpu()
            )
            sample_values[name]["feature_rms"].append(
                (feature - base_feature).square().mean(dim=1).sqrt().cpu()
            )
            sample_values[name]["feature_cosine"].append(
                (1.0 - F.cosine_similarity(feature, base_feature, dim=1)).cpu()
            )

        for target_time, state in trajectory.items():
            time = torch.full((len(state),), float(target_time), device=device)
            base_velocity = decoder_forward_with_embedding(
                decoder, state, time, local["base"]
            )
            for name in velocity_values:
                velocity = decoder_forward_with_embedding(
                    decoder, state, time, local[name]
                )
                velocity_values[name][target_time].append(
                    (velocity - base_velocity)
                    .flatten(1)
                    .square()
                    .mean(dim=1)
                    .sqrt()
                    .cpu()
                )

    result: dict[str, float] = {}
    for name, metrics in sample_values.items():
        for metric, parts in metrics.items():
            mean, sem = _mean_sem(torch.cat(parts))
            result[f"{name}_{metric}_mean"] = mean
            result[f"{name}_{metric}_sem"] = sem
    for name, time_values in velocity_values.items():
        for target_time, parts in time_values.items():
            mean, sem = _mean_sem(torch.cat(parts))
            suffix = str(target_time).replace(".", "p")
            result[f"{name}_velocity_rms_t{suffix}_mean"] = mean
            result[f"{name}_velocity_rms_t{suffix}_sem"] = sem
    return result


def _sphere_checks(embeddings: dict[str, torch.Tensor], angle: float) -> dict[str, float]:
    result: dict[str, float] = {}
    base = embeddings["base"]
    for name, value in embeddings.items():
        result[f"{name}_condition_abs_mean_max"] = float(value.mean(dim=1).abs().max())
        result[f"{name}_condition_rms_max_error"] = float(
            (value.square().mean(dim=1).sqrt() - 1.0).abs().max()
        )
        if name in BRANCHES:
            errors = (condition_angles(base, value) - float(angle)).abs()
            result[f"{name}_fixed_angle_max_error"] = float(errors.max())
    return result


def _independent_empirical_indices(
    train_count: int,
    excluded: torch.Tensor,
    count: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(int(train_count), generator=generator)
    excluded_set = set(int(value) for value in excluded.tolist())
    selected = [int(value) for value in order.tolist() if int(value) not in excluded_set]
    if len(selected) < int(count):
        raise RuntimeError("not enough disjoint empirical conditions")
    return torch.as_tensor(selected[: int(count)], dtype=torch.long)


@torch.no_grad()
def analyze_run(
    run: Path,
    *,
    device_name: str,
    count: int = 256,
    fixed_angle: float = FIXED_ANGLE,
    overwrite: bool = False,
) -> Path:
    output = run / "decoder_amplification_audit.json"
    if output.is_file() and not overwrite:
        print(f"decoder amplification audit already complete: {output}", flush=True)
        return output

    config = load_run_config(run, device_name)
    configure_fp32(config.prior_seed)
    device = torch.device(device_name)
    _encoder, decoder, frozen = load_frozen_models(config, device)
    frozen_hash_before = state_dict_sha256(decoder)
    cache = torch.load(run / "latent_cache.pt", map_location="cpu", weights_only=True)
    prior_state = torch.load(run / "prior_state.pt", map_location="cpu", weights_only=True)
    prior = build_prior(config, device)
    prior.load_state_dict(prior_state["prior_ema"])
    prior.eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    interface = OrthogonalLatentInterface(
        config.latent_dim,
        fixed_orthogonal_basis(INTERFACE_DIM, config.basis_seed),
    ).to(device)

    formal_count = min(int(config.quality_count), len(cache["val_latent"]))
    if not 2 <= int(count) <= formal_count:
        raise ValueError(f"count must lie in [2, {formal_count}]")
    empirical_generator = torch.Generator(device="cpu").manual_seed(
        config.prior_seed + 1_101
    )
    formal_empirical_indices = torch.randint(
        len(cache["train_latent"]),
        (formal_count,),
        generator=empirical_generator,
    )
    empirical_indices = formal_empirical_indices[: int(count)]
    empirical_latent = cache["train_latent"][empirical_indices]
    second_indices = _independent_empirical_indices(
        len(cache["train_latent"]),
        empirical_indices,
        int(count),
        config.prior_seed + 4_101,
    )
    second_empirical_latent = cache["train_latent"][second_indices]
    prior_latent = sample_prior_coordinates(
        prior,
        interface,
        formal_count,
        config.prior_ode_steps,
        seed=config.prior_seed + 1_201,
        batch_size=config.prior_batch_size,
    )[: int(count)]

    base = project_condition_sphere(
        condition_embeddings(decoder, empirical_latent, batch_size=config.eval_batch_size)
    )
    prior_condition = project_condition_sphere(
        condition_embeddings(decoder, prior_latent, batch_size=config.eval_batch_size)
    )
    second_empirical = project_condition_sphere(
        condition_embeddings(
            decoder, second_empirical_latent, batch_size=config.eval_batch_size
        )
    )
    matched_prior, _prior_assignment = hungarian_sphere_match(base, prior_condition)
    matched_empirical, _empirical_assignment = hungarian_sphere_match(
        base, second_empirical
    )
    prior_tangent = tangent_toward(base, matched_prior)
    empirical_tangent = tangent_toward(base, matched_empirical)
    isotropic_tangent = random_tangent(base, seed=config.prior_seed + 4_301)
    embeddings = {
        "base": base,
        "prior_direction": geodesic_step(base, prior_tangent, fixed_angle),
        "empirical_direction": geodesic_step(base, empirical_tangent, fixed_angle),
        "random_direction": geodesic_step(base, isotropic_tangent, fixed_angle),
        "prior_endpoint": matched_prior,
    }
    checks = _sphere_checks(embeddings, fixed_angle)
    if max(value for key, value in checks.items() if key.endswith("rms_max_error")) > 2e-5:
        raise RuntimeError("condition sphere RMS check failed")
    if max(value for key, value in checks.items() if key.endswith("angle_max_error")) > 2e-5:
        raise RuntimeError("fixed-angle intervention check failed")

    evaluator = ResNet18Evaluator().to(device).eval()
    responses = decoder_response_metrics(
        decoder,
        evaluator,
        embeddings,
        image_size=config.image_size,
        ode_steps=config.pixel_ode_steps,
        batch_size=min(int(config.eval_batch_size), 32),
        pixel_seed=33_001,
    )
    prior_angles = condition_angles(base, matched_prior)
    empirical_angles = condition_angles(base, matched_empirical)
    prior_angle_mean, prior_angle_sem = _mean_sem(prior_angles)
    empirical_angle_mean, empirical_angle_sem = _mean_sem(empirical_angles)
    summary = json.loads((run / "summary.json").read_text())
    payload: dict[str, float | int | bool | str] = {
        "latent_dim": int(config.latent_dim),
        "frozen_seed": int(config.frozen_seed),
        "count": int(count),
        "fixed_angle": float(fixed_angle),
        "condition_dim": int(base.shape[1]),
        "modeling_gap": float(summary["modeling_gap"]),
        "end_to_end_feature_fid": float(summary["end_to_end_feature_fid"]),
        "condition_prior_matched_angle_mean": prior_angle_mean,
        "condition_prior_matched_angle_sem": prior_angle_sem,
        "condition_prior_matched_angle_median": float(prior_angles.median()),
        "condition_empirical_matched_angle_mean": empirical_angle_mean,
        "condition_empirical_matched_angle_sem": empirical_angle_sem,
        "condition_empirical_matched_angle_median": float(empirical_angles.median()),
        **distribution_comparison(
            base,
            prior_condition,
            seed=config.prior_seed + 4_501,
            prefix="condition_prior",
        ),
        **distribution_comparison(
            base,
            second_empirical,
            seed=config.prior_seed + 4_503,
            prefix="condition_empirical_control",
        ),
        **two_sample_accuracy(base, prior_condition, config.prior_seed + 4_701),
        **checks,
        **responses,
    }
    for metric in ("pixel_rms", "feature_rms", "feature_cosine"):
        prior_value = float(payload[f"prior_direction_{metric}_mean"])
        random_value = float(payload[f"random_direction_{metric}_mean"])
        empirical_value = float(payload[f"empirical_direction_{metric}_mean"])
        payload[f"{metric}_alignment_ratio"] = prior_value / max(random_value, 1e-12)
        payload[f"{metric}_manifold_ratio"] = prior_value / max(empirical_value, 1e-12)
    for target_time in PROBE_TIMES:
        suffix = str(target_time).replace(".", "p")
        prior_value = float(payload[f"prior_direction_velocity_rms_t{suffix}_mean"])
        random_value = float(payload[f"random_direction_velocity_rms_t{suffix}_mean"])
        empirical_value = float(payload[f"empirical_direction_velocity_rms_t{suffix}_mean"])
        payload[f"velocity_t{suffix}_alignment_ratio"] = prior_value / max(
            random_value, 1e-12
        )
        payload[f"velocity_t{suffix}_manifold_ratio"] = prior_value / max(
            empirical_value, 1e-12
        )
    payload["prior_endpoint_feature_secant"] = float(
        payload["prior_endpoint_feature_rms_mean"] / max(prior_angle_mean, 1e-12)
    )
    payload["decoder_weighted_mismatch"] = float(
        prior_angle_mean
        * float(payload["prior_direction_feature_rms_mean"])
        / float(fixed_angle)
    )
    payload["frozen_decoder_sha256"] = state_dict_sha256(decoder)
    payload["frozen_decoder_matches_formal"] = bool(
        payload["frozen_decoder_sha256"] == frozen["frozen_decoder_sha256"]
        == summary["frozen_decoder_sha256"]
        == frozen_hash_before
    )
    if not all(
        math.isfinite(float(value))
        for value in payload.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ):
        raise FloatingPointError("non-finite decoder amplification metric")
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--fixed-angle", type=float, default=FIXED_ANGLE)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    return analyze_run(
        args.run,
        device_name=args.device,
        count=args.count,
        fixed_angle=args.fixed_angle,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
