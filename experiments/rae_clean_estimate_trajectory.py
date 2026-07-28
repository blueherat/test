"""Geometry and exploratory gates for RAE clean-estimate trajectories."""

from __future__ import annotations

import pandas as pd
import torch

from experiments.rae_cycle_direction_intervention import match_sample_rms, sample_rms
from experiments.rae_layerwise_path import path_coefficients, spatial_center


PATHS = ("static", "random", "annealed", "reverse")


def clean_estimate(
    state: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor | float,
) -> torch.Tensor:
    """Recover the linear-flow clean endpoint estimate z_t - t v(z_t,t)."""

    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have equal shape")
    if isinstance(time, torch.Tensor):
        if time.ndim == 0:
            scale = time.to(state)
        elif time.ndim == 1 and len(time) == len(state):
            scale = time.to(state).reshape(-1, *([1] * (state.ndim - 1)))
        else:
            raise ValueError("time must be scalar or have one value per sample")
    else:
        scale = state.new_tensor(float(time))
    return state.float() - scale * velocity.float()


def project_to_endpoint_chord(
    query: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project each query onto its paired start-to-end Euclidean chord."""

    if query.shape != start.shape or start.shape != end.shape or query.ndim < 2:
        raise ValueError("query, start, and end must share shape [B,...]")
    chord = end.float() - start.float()
    displacement = query.float() - start.float()
    denominator = chord.square().flatten(1).sum(dim=1).clamp_min(1e-12)
    progress = (displacement * chord).flatten(1).sum(dim=1) / denominator
    expanded = progress.reshape(-1, *([1] * (query.ndim - 1)))
    projection = start.float() + expanded * chord
    curvature = sample_rms(query.float() - projection) / sample_rms(chord, 1e-12)
    return progress, projection, curvature


def rms_matched_projection(query: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    if query.shape != projection.shape:
        raise ValueError("query and projection must have equal shape")
    return match_sample_rms(projection.float(), sample_rms(query.float()))


def endpoint_observation_factors(
    time: torch.Tensor,
    mode: str,
    *,
    power: float,
    family: str = "power",
    floor: float = 0.0,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return factors relating z_t - t v_t to semantic/detail endpoints."""

    sem_c, detail_c, sem_d, detail_d = path_coefficients(
        time.float(),
        mode,
        power=float(power),
        family=family,
        floor=float(floor),
        alpha=float(alpha),
    )
    multiplier = time.float() * (1.0 - time.float())
    return sem_c - multiplier * sem_d, detail_c - multiplier * detail_d


def invert_path_endpoint_observation(
    observation: torch.Tensor,
    time: torch.Tensor,
    basis: torch.Tensor,
    *,
    mode: str,
    power: float,
    family: str = "power",
    floor: float = 0.0,
    alpha: float = 1.0,
    detail_scale: float,
) -> torch.Tensor:
    """Invert the exact layerwise-path observation z_t - t v_t.

    The detail operator is ``detail_scale * Pi`` on spatial residuals, so the
    basis-subspace factor differs from both the semantic and detail factors
    when ``detail_scale != 1``.
    """

    if observation.ndim != 4 or time.ndim != 1 or len(time) != len(observation):
        raise ValueError("expected observation [B,C,H,W] and time [B]")
    if basis.ndim != 2 or basis.shape[0] != observation.shape[1]:
        raise ValueError("basis must have shape [C,K]")
    if float(detail_scale) <= 0:
        raise ValueError("detail_scale must be positive")
    semantic_factor, detail_factor = endpoint_observation_factors(
        time,
        mode,
        power=power,
        family=family,
        floor=floor,
        alpha=alpha,
    )
    scale = float(detail_scale)
    basis_factor = semantic_factor * (1.0 - scale) + detail_factor * scale
    if bool((semantic_factor.abs() < 1e-8).any() or (basis_factor.abs() < 1e-8).any()):
        raise ValueError("path endpoint observation is singular at this time")

    mean, residual = spatial_center(observation.float())
    basis = basis.to(device=observation.device, dtype=observation.dtype)
    rows = residual.permute(0, 2, 3, 1).reshape(-1, observation.shape[1])
    projected_rows = (rows @ basis) @ basis.transpose(0, 1)
    projected = projected_rows.reshape(
        observation.shape[0], observation.shape[2], observation.shape[3], observation.shape[1]
    ).permute(0, 3, 1, 2).contiguous()
    complement = residual - projected
    expand = (-1,) + (1,) * (observation.ndim - 1)
    semantic_factor = semantic_factor.to(observation).reshape(expand)
    basis_factor = basis_factor.to(observation).reshape(expand)
    return mean / semantic_factor + complement / semantic_factor + projected / basis_factor


def trajectory_prediction_summary(
    distribution: pd.DataFrame,
    latent_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Evaluate the exploratory predictions fixed before the trajectory run."""

    required_distribution = {"path", "step_index", "method", "projected_frechet"}
    required_latent = {"path", "step_index", "kind", "curvature_ratio", "progress"}
    if missing := required_distribution.difference(distribution.columns):
        raise KeyError(f"distribution is missing {sorted(missing)}")
    if missing := required_latent.difference(latent_metrics.columns):
        raise KeyError(f"latent_metrics is missing {sorted(missing)}")

    rows = []
    for path in PATHS:
        metrics = latent_metrics[
            (latent_metrics.path == path)
            & (latent_metrics.kind == "actual")
            & (latent_metrics.step_index > 0)
            & (latent_metrics.step_index < 49)
        ]
        curvature_by_step = metrics.groupby("step_index").curvature_ratio.median()
        progress_by_step = metrics.groupby("step_index").progress.median()
        table = distribution[
            (distribution.path == path)
            & (distribution.step_index > 0)
            & (distribution.step_index < 49)
        ].pivot(index="step_index", columns="method", values="projected_frechet")
        line_advantage = table["chord"] - table["actual"]
        rms_advantage = table["rms_chord"] - table["actual"]
        positive_line = float((line_advantage > 0).mean())
        positive_rms = float((rms_advantage > 0).mean())
        mean_line = float(line_advantage.mean())
        mean_rms = float(rms_advantage.mean())
        retention = mean_rms / mean_line if mean_line > 1e-12 else float("nan")
        rows.append(
            {
                "path": path,
                "max_median_curvature": float(curvature_by_step.max()),
                "final_intermediate_progress": float(progress_by_step.iloc[-1]),
                "actual_beats_chord_fraction": positive_line,
                "actual_beats_rms_chord_fraction": positive_rms,
                "mean_chord_advantage": mean_line,
                "mean_rms_chord_advantage": mean_rms,
                "rms_advantage_retention": retention,
                "p1_curved": bool(float(curvature_by_step.max()) >= 0.15),
                "p3_chord_advantage": bool(positive_line > 0.5),
                "p4_rms_robust": bool(positive_rms > 0.5 and retention >= 0.5),
            }
        )
    summary = pd.DataFrame(rows)
    counts = {
        "p1_curved_paths": int(summary.p1_curved.sum()),
        "p3_chord_advantage_paths": int(summary.p3_chord_advantage.sum()),
        "p4_rms_robust_paths": int(summary.p4_rms_robust.sum()),
    }
    predictions = {
        "p1_curvature": counts["p1_curved_paths"] >= 3,
        "p3_decoder_advantage": counts["p3_chord_advantage_paths"] >= 3,
        "p4_not_only_rms": counts["p4_rms_robust_paths"] >= 3,
    }
    return summary, {
        "pass": bool(all(predictions.values())),
        "predictions": predictions,
        "counts": counts,
    }
