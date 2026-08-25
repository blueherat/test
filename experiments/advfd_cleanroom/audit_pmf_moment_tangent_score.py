#!/usr/bin/env python3
"""Fit and audit a moment-tangent residual score on the pMF feature bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from experiments.advfd_cleanroom.run_pmf_residual_score_posttrain import (
    ensemble_residual_field,
    load_shared_ensemble,
)
from experiments.frechet_residual_score_toy import (
    gaussian_transport_field,
    project_onto_fixed_moment_tangent,
    weighted_inner,
    weighted_moments,
)
from experiments.run_residual_score_estimator_toy import (
    parse_float_tuple,
    parse_int_tuple,
)


def estimate_bank_field(
    features: torch.Tensor,
    ensemble,
    sigmas: tuple[float, ...],
    *,
    batch_size: int,
    noise_seed: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device)
        field, _ = ensemble_residual_field(
            ensemble,
            batch,
            sigmas,
            noise_seed=noise_seed + start,
        )
        parts.append(field.cpu())
    return torch.cat(parts)


def apply_projection(
    states: torch.Tensor,
    field: torch.Tensor,
    *,
    source_mean: torch.Tensor,
    translation: torch.Tensor,
    symmetric_linear: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal = translation + (states - source_mean) @ symmetric_linear.mT
    return field - normal, normal


def moment_derivatives(
    states: torch.Tensor, field: torch.Tensor
) -> tuple[float, float]:
    weights = torch.ones(len(states), dtype=states.dtype, device=states.device)
    moments = weighted_moments(states, weights)
    centered = states - moments.mean
    mean_derivative = field.mean(dim=0)
    cross = centered.mT @ field / len(states)
    covariance_derivative = cross + cross.mT
    return float(mean_derivative.norm()), float(covariance_derivative.norm())


def summarize_split(
    split: str,
    states: torch.Tensor,
    full: torch.Tensor,
    tangent: torch.Tensor,
    normal: torch.Tensor,
    static: torch.Tensor,
) -> dict[str, Any]:
    weights = torch.ones(len(states), dtype=states.dtype, device=states.device)
    full_norm = weighted_inner(full, full, weights).sqrt()
    tangent_norm = weighted_inner(tangent, tangent, weights).sqrt()
    normal_norm = weighted_inner(normal, normal, weights).sqrt()
    static_norm = weighted_inner(static, static, weights).sqrt()
    full_mean, full_cov = moment_derivatives(states, full)
    tangent_mean, tangent_cov = moment_derivatives(states, tangent)
    return {
        "split": split,
        "samples": len(states),
        "full_rms": float(full.square().mean().sqrt()),
        "tangent_rms": float(tangent.square().mean().sqrt()),
        "normal_rms": float(normal.square().mean().sqrt()),
        "tangent_energy_fraction": float(
            tangent_norm.square() / full_norm.square().clamp_min(1e-20)
        ),
        "full_tangent_cosine": float(
            weighted_inner(full, tangent, weights)
            / (full_norm * tangent_norm).clamp_min(1e-20)
        ),
        "tangent_static_cosine": float(
            weighted_inner(tangent, static, weights)
            / (tangent_norm * static_norm).clamp_min(1e-20)
        ),
        "tangent_normal_cosine": float(
            weighted_inner(tangent, normal, weights)
            / (tangent_norm * normal_norm).clamp_min(1e-20)
        ),
        "full_mean_derivative_norm": full_mean,
        "full_covariance_derivative_norm": full_cov,
        "tangent_mean_derivative_norm": tangent_mean,
        "tangent_covariance_derivative_norm": tangent_cov,
    }


def run(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    device = torch.device(args.device)
    bank = torch.load(args.feature_bank, map_location="cpu", weights_only=False)
    if bank.get("protocol") != "frozen_pmf_b_inception64_residual_feature_bank_v1":
        raise ValueError("unexpected feature-bank protocol")
    dimension = int(bank["projection"].shape[1])
    ensemble = load_shared_ensemble(
        args.estimator_root,
        args.estimator_seeds,
        dimension=dimension,
        sigmas=args.sigmas,
        device=device,
    )
    fake_train = bank["fake_train"].double()
    fake_heldout = bank["fake_heldout"].double()
    real_train = bank["real_train"].double()
    full_train = estimate_bank_field(
        fake_train.float(),
        ensemble,
        args.sigmas,
        batch_size=args.batch_size,
        noise_seed=args.seed,
        device=device,
    ).double()
    full_heldout = estimate_bank_field(
        fake_heldout.float(),
        ensemble,
        args.sigmas,
        batch_size=args.batch_size,
        noise_seed=args.seed + 1000003,
        device=device,
    ).double()
    train_weights = torch.ones(len(fake_train), dtype=torch.float64)
    projection = project_onto_fixed_moment_tangent(
        fake_train, full_train, train_weights
    )
    source_moments = weighted_moments(fake_train, train_weights)
    target_moments = weighted_moments(
        real_train, torch.ones(len(real_train), dtype=torch.float64)
    )
    tangent_train = projection.tangent
    normal_train = projection.normal
    tangent_heldout, normal_heldout = apply_projection(
        fake_heldout,
        full_heldout,
        source_mean=source_moments.mean,
        translation=projection.translation,
        symmetric_linear=projection.symmetric_linear,
    )
    static_train = gaussian_transport_field(
        source_moments, target_moments, fake_train
    )
    static_heldout = gaussian_transport_field(
        source_moments, target_moments, fake_heldout
    )
    rows = [
        summarize_split(
            "train",
            fake_train,
            full_train,
            tangent_train,
            normal_train,
            static_train,
        ),
        summarize_split(
            "heldout",
            fake_heldout,
            full_heldout,
            tangent_heldout,
            normal_heldout,
            static_heldout,
        ),
    ]
    pd.DataFrame(rows).to_csv(args.output_root / "projection_audit.csv", index=False)
    artifact = {
        "protocol": "pmf_inception64_moment_tangent_score_projection_v1",
        "feature_bank": str(args.feature_bank),
        "estimator_root": str(args.estimator_root),
        "estimator_seeds": list(args.estimator_seeds),
        "sigmas": list(args.sigmas),
        "noise_seed": args.seed,
        "source_mean": source_moments.mean.float(),
        "translation": projection.translation.float(),
        "symmetric_linear": projection.symmetric_linear.float(),
    }
    torch.save(artifact, args.output_root / "moment_tangent_projection.pt")
    summary = {
        "protocol": artifact["protocol"],
        "projection_fit_samples": len(fake_train),
        "projection_heldout_samples": len(fake_heldout),
        "train_projection_constraints_pass": bool(
            projection.mean_derivative.norm() < 1e-8
            and projection.covariance_derivative.norm() < 1e-8
            and projection.orthogonality_error < 1e-8
        ),
        "rows": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-bank", type=Path, required=True)
    parser.add_argument("--estimator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--estimator-seeds", type=parse_int_tuple, default=(0, 1, 2))
    parser.add_argument(
        "--sigmas", type=parse_float_tuple, default=(0.1, 0.3, 0.7, 1.5)
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--device", default="cuda:3")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
