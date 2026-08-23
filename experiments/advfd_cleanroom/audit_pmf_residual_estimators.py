#!/usr/bin/env python3
"""Audit residual-score estimators on frozen pMF Inception features."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from experiments.residual_score_toy import (
    classifier_metrics,
    pairwise_field_metrics,
)
from experiments.run_residual_score_estimator_toy import (
    DSM_METHODS,
    RATIO_METHODS,
    EstimatorBundle,
    ExperimentConfig,
    estimate_field,
    parse_float_tuple,
    parse_int_tuple,
    parse_str_tuple,
    train_dsm_estimator,
    train_ratio_estimator,
)


def covariance(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = features.double()
    mean = values.mean(dim=0)
    centered = values - mean
    cov = centered.mT @ centered / max(len(values) - 1, 1)
    return mean, 0.5 * (cov + cov.mT)


def matrix_square_root(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (matrix + matrix.mT))
    return eigenvectors @ torch.diag(eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.mT


def frechet_distance(real: torch.Tensor, fake: torch.Tensor) -> float:
    real_mean, real_covariance = covariance(real)
    fake_mean, fake_covariance = covariance(fake)
    real_root = matrix_square_root(real_covariance)
    middle = real_root @ fake_covariance @ real_root
    covariance_root = matrix_square_root(middle)
    value = (
        (real_mean - fake_mean).square().sum()
        + torch.trace(real_covariance)
        + torch.trace(fake_covariance)
        - 2.0 * torch.trace(covariance_root)
    )
    return float(value.clamp_min(0))


def coordinate_rms(values: torch.Tensor) -> torch.Tensor:
    return values.square().mean().sqrt()


def normalized_displacement(
    direction: torch.Tensor,
    target_coordinate_rms: float,
) -> torch.Tensor:
    rms = coordinate_rms(direction)
    if float(rms) == 0.0:
        return torch.zeros_like(direction)
    return direction * (float(target_coordinate_rms) / rms)


def affine_update(
    clean_fake: torch.Tensor,
    residual_field: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the shared affine-generator update induced by a residual field."""

    matrix = torch.einsum("bo,bi->oi", residual_field, clean_fake) / len(clean_fake)
    bias = residual_field.mean(dim=0)
    direction = clean_fake @ matrix.mT + bias
    vector = torch.cat((matrix.flatten(), bias))
    return vector, direction, matrix


def fixed_noised_heldout(
    real: torch.Tensor,
    fake: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real_noise = torch.randn(real.shape, generator=generator, dtype=real.dtype)
    fake_noise = torch.randn(fake.shape, generator=generator, dtype=fake.dtype)
    real_noisy = real + float(sigma) * real_noise
    fake_noisy = fake + float(sigma) * fake_noise
    states = torch.cat((real_noisy, fake_noisy), dim=0)
    labels = torch.cat(
        (
            torch.ones(len(real), dtype=states.dtype),
            torch.zeros(len(fake), dtype=states.dtype),
        )
    )
    return states, labels, fake_noisy, fake_noise


def training_config(args: argparse.Namespace, seed: int) -> ExperimentConfig:
    return ExperimentConfig(
        epsilons=(),
        train_samples=(),
        seeds=(seed,),
        methods=args.methods,
        sobolev_lambdas=(args.sobolev_lambda,),
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        frequencies=args.frequencies,
        domain_dim=args.domain_dim,
        train_sigma_min=args.train_sigma_min,
        train_sigma_max=args.train_sigma_max,
        eval_sigmas=args.eval_sigmas,
        eval_samples=0,
        pushforward_step_size=0.0,
        quadrature_grid_size=16,
        quadrature_batch_size=16,
        log_every=args.log_every,
        components=0,
        radius=0.0,
        component_std=0.0,
        perturbation_amplitude=0.0,
        data_seed=args.data_seed,
        eval_seed=args.eval_seed,
        device=args.device,
    )


def train_bundle(
    method: str,
    *,
    real_train: torch.Tensor,
    fake_train: torch.Tensor,
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
) -> EstimatorBundle:
    if method in RATIO_METHODS:
        return train_ratio_estimator(
            kind=method,
            sobolev_lambda=(
                config.sobolev_lambdas[0] if method == "sobolev_ratio" else 0.0
            ),
            real_bank=real_train,
            fake_bank=fake_train,
            config=config,
            seed=seed,
            device=device,
        )
    if method in DSM_METHODS:
        return train_dsm_estimator(
            kind=method,
            real_bank=real_train,
            fake_bank=fake_train,
            config=config,
            seed=seed,
            device=device,
        )
    raise ValueError(f"unknown method: {method}")


def evaluate_bundle(
    bundle: EstimatorBundle,
    *,
    real_heldout_cpu: torch.Tensor,
    fake_heldout_cpu: torch.Tensor,
    sigma: float,
    displacement_rms: tuple[float, ...],
    eval_seed: int,
    device: torch.device,
    output_root: Path,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states_cpu, labels_cpu, fake_noisy_cpu, _ = fixed_noised_heldout(
        real_heldout_cpu,
        fake_heldout_cpu,
        sigma=sigma,
        seed=eval_seed,
    )
    states = states_cpu.to(device).requires_grad_(True)
    sigma_batch = torch.full(
        (len(states),), sigma, device=device, dtype=states.dtype
    )
    bundle.train(False)
    field = estimate_field(
        bundle,
        states,
        sigma_batch,
        create_graph=False,
    ).detach().cpu()
    fake_field = field[len(real_heldout_cpu) :]
    clean_fake = fake_heldout_cpu
    affine_vector, affine_direction, _ = affine_update(clean_fake, fake_field)
    baseline_fd = frechet_distance(real_heldout_cpu, clean_fake)
    row: dict[str, Any] = {
        "seed": seed,
        "method": bundle.label,
        "kind": bundle.kind,
        "sigma": sigma,
        "field_coordinate_rms": float(coordinate_rms(fake_field)),
        "field_vector_rms": float(fake_field.square().sum(dim=1).mean().sqrt()),
        "affine_gradient_norm": float(affine_vector.norm()),
        "baseline_feature_fd": baseline_fd,
        "train_seconds": bundle.elapsed_seconds,
        "parameter_count": bundle.parameter_count,
    }
    if bundle.kind in RATIO_METHODS:
        with torch.no_grad():
            logits = bundle.modules["ratio"](states.detach(), sigma_batch).cpu()
        row.update(classifier_metrics(logits, labels_cpu))
    intervention_rows: list[dict[str, Any]] = []
    for target_rms in displacement_rms:
        for sign in (-1.0, 1.0):
            direct_delta = normalized_displacement(fake_field, target_rms) * sign
            affine_delta = normalized_displacement(affine_direction, target_rms) * sign
            intervention_rows.extend(
                (
                    {
                        "seed": seed,
                        "method": bundle.label,
                        "kind": bundle.kind,
                        "sigma": sigma,
                        "intervention": "sample_field",
                        "sign": sign,
                        "target_coordinate_rms": target_rms,
                        "actual_coordinate_rms": float(coordinate_rms(direct_delta)),
                        "feature_fd": frechet_distance(
                            real_heldout_cpu, clean_fake + direct_delta
                        ),
                        "baseline_feature_fd": baseline_fd,
                    },
                    {
                        "seed": seed,
                        "method": bundle.label,
                        "kind": bundle.kind,
                        "sigma": sigma,
                        "intervention": "shared_affine_vjp",
                        "sign": sign,
                        "target_coordinate_rms": target_rms,
                        "actual_coordinate_rms": float(coordinate_rms(affine_delta)),
                        "feature_fd": frechet_distance(
                            real_heldout_cpu, clean_fake + affine_delta
                        ),
                        "baseline_feature_fd": baseline_fd,
                    },
                )
            )
    artifact_path = (
        output_root
        / "fields"
        / f"seed{seed}"
        / bundle.label
        / f"sigma{sigma:g}.npz"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        field=field.numpy(),
        fake_field=fake_field.numpy(),
        affine_vector=affine_vector.numpy(),
        affine_direction=affine_direction.numpy(),
        states=states_cpu.numpy(),
    )
    row["artifact_path"] = str(artifact_path.relative_to(output_root))
    return row, intervention_rows


def pairwise_metrics(metrics: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (method, kind, sigma), group in metrics.groupby(["method", "kind", "sigma"]):
        group = group.sort_values("seed")
        for (_, left), (_, right) in combinations(group.iterrows(), 2):
            left_data = np.load(output_root / left["artifact_path"])
            right_data = np.load(output_root / right["artifact_path"])
            field_stats = pairwise_field_metrics(
                torch.from_numpy(left_data["fake_field"]),
                torch.from_numpy(right_data["fake_field"]),
            )
            affine_stats = pairwise_field_metrics(
                torch.from_numpy(left_data["affine_vector"])[None],
                torch.from_numpy(right_data["affine_vector"])[None],
            )
            records.append(
                {
                    "method": method,
                    "kind": kind,
                    "sigma": sigma,
                    "left_seed": int(left["seed"]),
                    "right_seed": int(right["seed"]),
                    "field_cosine": field_stats["pairwise_cosine"],
                    "field_relative_l2": field_stats["pairwise_relative_l2"],
                    "affine_vjp_cosine": affine_stats["pairwise_cosine"],
                    "affine_vjp_relative_l2": affine_stats["pairwise_relative_l2"],
                }
            )
    return pd.DataFrame(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=parse_int_tuple, default=(0, 1, 2))
    parser.add_argument(
        "--methods",
        type=parse_str_tuple,
        default=(
            "zero_ratio",
            "ratio",
            "sobolev_ratio",
            "shared_dsm",
            "factorized_dsm_coupled",
        ),
    )
    parser.add_argument("--sobolev-lambda", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--frequencies", type=int, default=6)
    parser.add_argument("--domain-dim", type=int, default=16)
    parser.add_argument("--train-sigma-min", type=float, default=0.05)
    parser.add_argument("--train-sigma-max", type=float, default=2.0)
    parser.add_argument(
        "--eval-sigmas", type=parse_float_tuple, default=(0.0, 0.1, 0.3, 0.7, 1.5)
    )
    parser.add_argument(
        "--displacement-rms", type=parse_float_tuple, default=(0.01, 0.03, 0.1)
    )
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--data-seed", type=int, default=20260824)
    parser.add_argument("--eval-seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda:3")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_steps <= 0 or args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("invalid training configuration")
    if args.sobolev_lambda < 0 or any(value <= 0 for value in args.displacement_rms):
        raise ValueError("invalid regularization/intervention configuration")
    device = torch.device(args.device)
    bank = torch.load(args.feature_bank, map_location="cpu", weights_only=False)
    if bank.get("protocol") != "frozen_pmf_b_inception64_residual_feature_bank_v1":
        raise ValueError("unexpected feature-bank protocol")
    real_train = bank["real_train"].to(device)
    fake_train = bank["fake_train"].to(device)
    real_heldout = bank["real_heldout"].float()
    fake_heldout = bank["fake_heldout"].float()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "feature_bank": str(args.feature_bank),
                "output_root": str(args.output_root),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    metric_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    total_jobs = len(args.seeds) * len(args.methods)
    job = 0
    for seed in args.seeds:
        config = training_config(args, seed)
        for method in args.methods:
            job += 1
            print(f"[{job}/{total_jobs}] seed={seed} method={method}", flush=True)
            bundle = train_bundle(
                method,
                real_train=real_train,
                fake_train=fake_train,
                config=config,
                seed=args.data_seed + 1009 * seed,
                device=device,
            )
            checkpoint_path = (
                args.output_root / "checkpoints" / f"seed{seed}" / f"{bundle.label}.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bundle.state_dict(), checkpoint_path)
            for history in bundle.history:
                history_rows.append(
                    {
                        "seed": seed,
                        "method": bundle.label,
                        "kind": bundle.kind,
                        **history,
                    }
                )
            for sigma_index, sigma in enumerate(args.eval_sigmas):
                if bundle.kind in DSM_METHODS and sigma <= 0:
                    continue
                if bundle.kind == "zero_ratio" and sigma != 0:
                    continue
                metric, interventions = evaluate_bundle(
                    bundle,
                    real_heldout_cpu=real_heldout,
                    fake_heldout_cpu=fake_heldout,
                    sigma=sigma,
                    displacement_rms=args.displacement_rms,
                    eval_seed=args.eval_seed + 10007 * sigma_index,
                    device=device,
                    output_root=args.output_root,
                    seed=seed,
                )
                metric_rows.append(metric)
                intervention_rows.extend(interventions)
            pd.DataFrame(metric_rows).to_csv(
                args.output_root / "metrics.csv", index=False
            )
            pd.DataFrame(intervention_rows).to_csv(
                args.output_root / "interventions.csv", index=False
            )
            pd.DataFrame(history_rows).to_csv(
                args.output_root / "training_curves.csv", index=False
            )
    metrics = pd.DataFrame(metric_rows)
    pairwise = pairwise_metrics(metrics, args.output_root)
    pairwise.to_csv(args.output_root / "pairwise_metrics.csv", index=False)
    metrics.groupby(["method", "kind", "sigma"], dropna=False).mean(
        numeric_only=True
    ).to_csv(args.output_root / "metrics_mean.csv")
    pd.DataFrame(intervention_rows).groupby(
        [
            "method",
            "kind",
            "sigma",
            "intervention",
            "sign",
            "target_coordinate_rms",
        ],
        dropna=False,
    ).agg(
        feature_fd_mean=("feature_fd", "mean"),
        feature_fd_std=("feature_fd", "std"),
        baseline_feature_fd=("baseline_feature_fd", "mean"),
    ).reset_index().to_csv(args.output_root / "interventions_mean.csv", index=False)
    print(f"wrote results to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
