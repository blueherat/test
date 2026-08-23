#!/usr/bin/env python3
"""Summarize saved AdvFD critic moments without evaluating the generator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch


STEP_RE = re.compile(r"step_(\d+)\.pth$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--reference-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--whiten-eps", type=float, default=1e-3)
    return parser.parse_args()


def covariance_from_state(stats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mu = stats["mu_ema"].detach().double()
    m2 = stats["m2_ema"].detach().double()
    cov = m2 - mu[:, None] * mu[None, :]
    cov = 0.5 * (cov + cov.T)
    return mu, cov


def moment_summary(prefix: str, stats: dict[str, torch.Tensor]) -> dict[str, float | int]:
    initialized = int(stats["initialized"].reshape(-1)[0].item())
    mu, cov = covariance_from_state(stats)
    dim = int(mu.numel())
    second_moment_trace = float(torch.trace(cov) + mu.square().sum())
    covariance_trace = float(torch.trace(cov))
    covariance_frobenius_sq = float(cov.square().sum())
    participation_rank = (
        covariance_trace**2 / covariance_frobenius_sq
        if covariance_frobenius_sq > 0.0
        else 0.0
    )
    return {
        f"{prefix}_initialized": initialized,
        f"{prefix}_feature_dim": dim,
        f"{prefix}_mean_norm": float(torch.linalg.vector_norm(mu)),
        f"{prefix}_feature_rms": math.sqrt(max(second_moment_trace / dim, 0.0)),
        f"{prefix}_vector_rms": math.sqrt(max(second_moment_trace, 0.0)),
        f"{prefix}_covariance_trace": covariance_trace,
        f"{prefix}_covariance_frobenius": math.sqrt(max(covariance_frobenius_sq, 0.0)),
        f"{prefix}_covariance_participation_rank": participation_rank,
        f"{prefix}_covariance_diag_min": float(torch.diagonal(cov).min()),
        f"{prefix}_covariance_diag_max": float(torch.diagonal(cov).max()),
    }


def critic_parameter_norm(model_state: dict[str, torch.Tensor]) -> float:
    squared_norm = 0.0
    for value in model_state.values():
        if torch.is_floating_point(value):
            squared_norm += float(value.detach().double().square().sum())
    return math.sqrt(squared_norm)


def real_whitened_fd_from_stats(
    real_mu: torch.Tensor,
    real_covariance: torch.Tensor,
    fake_mu: torch.Tensor,
    fake_covariance: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[float, float, float]:
    """Match the official regularized real-whitened Gaussian FD exactly."""

    if epsilon <= 0.0:
        raise ValueError("whiten epsilon must be positive")
    dimension = int(real_mu.numel())
    identity = torch.eye(dimension, dtype=torch.float64)
    real_regularized = 0.5 * (real_covariance + real_covariance.T) + (
        epsilon * identity
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(real_regularized)
    inverse_roots = eigenvalues.clamp_min(epsilon).rsqrt()

    fake_mean_white = ((fake_mu - real_mu) @ eigenvectors) * inverse_roots
    fake_regularized = 0.5 * (fake_covariance + fake_covariance.T) + (
        epsilon * identity
    )
    fake_covariance_eigenbasis = eigenvectors.T @ fake_regularized @ eigenvectors
    fake_covariance_white = (
        fake_covariance_eigenbasis
        * inverse_roots[:, None]
        * inverse_roots[None, :]
    )
    fake_covariance_white = 0.5 * (
        fake_covariance_white + fake_covariance_white.T
    )

    mean_term = float(fake_mean_white.square().sum())
    covariance_term = float(
        (
            torch.trace(fake_covariance_white)
            + dimension
            - 2.0
            * torch.linalg.eigvalsh(fake_covariance_white)
            .clamp_min(0.0)
            .sqrt()
            .sum()
        ).clamp_min(0.0)
    )
    return mean_term + covariance_term, mean_term, covariance_term


def reference_summary(path: Path) -> dict[str, float | int]:
    with np.load(path) as data:
        mu = np.asarray(data["mu"], dtype=np.float64)
        cov = np.asarray(data["sigma"], dtype=np.float64)
    second_moment_trace = float(np.trace(cov) + np.dot(mu, mu))
    return {
        "reference_feature_dim": int(mu.size),
        "reference_mean_norm": float(np.linalg.norm(mu)),
        "reference_feature_rms": math.sqrt(max(second_moment_trace / mu.size, 0.0)),
        "reference_vector_rms": math.sqrt(max(second_moment_trace, 0.0)),
        "reference_covariance_trace": float(np.trace(cov)),
    }


def summarize_checkpoint(
    path: Path, reference: dict[str, float | int], *, whiten_epsilon: float
) -> dict[str, object]:
    match = STEP_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse checkpoint step from {path}")

    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    adv_states = checkpoint.get("fd_adv_states")
    if not isinstance(adv_states, list) or not adv_states:
        raise ValueError(f"Checkpoint has no adaptive FD state: {path}")

    result: dict[str, object] = {
        "checkpoint": str(path.resolve()),
        "checkpoint_bytes": path.stat().st_size,
        "filename_step": int(match.group(1)),
        "saved_step": int(checkpoint.get("step", -1)),
        "current_step": int(checkpoint.get("current_step", -1)),
        "samples_seen": int(checkpoint.get("samples_seen", -1)),
        **reference,
    }

    for index, state in enumerate(adv_states):
        name = str(state.get("name", f"critic_{index}"))
        prefix = f"critic_{index}_{name}"
        result[f"{prefix}_parameter_norm"] = critic_parameter_norm(state["model"])
        result.update(moment_summary(f"{prefix}_real", state["real_stats"]))
        result.update(moment_summary(f"{prefix}_fake", state["fake_stats"]))

        real_mu, real_cov = covariance_from_state(state["real_stats"])
        fake_mu, fake_cov = covariance_from_state(state["fake_stats"])
        whitened_total, whitened_mean, whitened_covariance = (
            real_whitened_fd_from_stats(
                real_mu,
                real_cov,
                fake_mu,
                fake_cov,
                epsilon=whiten_epsilon,
            )
        )
        result[f"{prefix}_ema_real_whitened_fd"] = whitened_total
        result[f"{prefix}_ema_real_whitened_fd_mean"] = whitened_mean
        result[f"{prefix}_ema_real_whitened_fd_covariance"] = whitened_covariance
        result[f"{prefix}_mean_gap_norm"] = float(torch.linalg.vector_norm(real_mu - fake_mu))
        result[f"{prefix}_mean_gap_squared"] = float((real_mu - fake_mu).square().sum())
        result[f"{prefix}_covariance_gap_frobenius"] = float(
            torch.linalg.matrix_norm(real_cov - fake_cov, ord="fro")
        )

        ref_rms = float(reference["reference_feature_rms"])
        for split in ("real", "fake"):
            rms_key = f"{prefix}_{split}_feature_rms"
            result[f"{rms_key}_ratio_to_reference"] = float(result[rms_key]) / ref_rms

    return result


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: int(row["filename_step"]))
    fieldnames = sorted({key for row in rows for key in row})

    with (output_dir / "checkpoint_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "checkpoint_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    reference = reference_summary(args.reference_stats)
    rows = [
        summarize_checkpoint(
            path, reference, whiten_epsilon=args.whiten_eps
        )
        for path in args.checkpoints
    ]
    write_outputs(rows, args.output_dir)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
