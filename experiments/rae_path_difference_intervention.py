"""Core controls and gates for paired RAE path-difference interventions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from experiments.rae_cycle_direction_intervention import (
    cyclic_derangement,
    match_sample_rms,
    sample_rms,
)


PATH_PAIRS = (
    ("static", "reverse"),
    ("static", "random"),
    ("random", "annealed"),
    ("annealed", "reverse"),
)
KNOWN_5K_FID = {
    "static": 123.53,
    "random": 128.99,
    "annealed": 143.54,
    "reverse": 159.05,
}


def fit_global_direction(calibration_delta: torch.Tensor) -> torch.Tensor:
    """Fit one path-level direction without using held-out samples."""

    if calibration_delta.ndim != 4 or len(calibration_delta) < 1:
        raise ValueError("calibration_delta must have shape [B,C,H,W]")
    return calibration_delta.float().mean(dim=0, keepdim=True)


def matched_path_directions(
    delta: torch.Tensor,
    global_direction: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return own, shuffled, random, global, and opposite equal-RMS directions."""

    if delta.ndim != 4 or len(delta) < 2:
        raise ValueError("delta must have shape [B,C,H,W] with B >= 2")
    if global_direction.shape != (1, *delta.shape[1:]):
        raise ValueError("global_direction must have shape [1,C,H,W]")
    delta = delta.float()
    target = sample_rms(delta)
    permutation = cyclic_derangement(len(delta), seed)
    shuffled = match_sample_rms(delta[permutation], target)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 130_363)
    random = torch.randn(delta.shape, generator=generator, dtype=torch.float32)
    random = match_sample_rms(random, target)
    global_matched = match_sample_rms(global_direction.expand_as(delta), target)
    return {
        "own": delta,
        "shuffled": shuffled,
        "random": random,
        "global": global_matched,
        "opposite": -delta,
    }


def spatial_components(delta: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split a path difference into token-common and centered spatial parts."""

    if delta.ndim != 4:
        raise ValueError("delta must have shape [B,C,H,W]")
    token_mean = delta.float().mean(dim=(-2, -1), keepdim=True).expand_as(delta)
    residual = delta.float() - token_mean
    return {"token_mean": token_mean, "spatial_residual": residual}


def rms_preserving_lerp(
    start: torch.Tensor,
    end: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Linearly interpolate direction while preserving interpolated sample RMS."""

    if start.shape != end.shape or start.ndim < 2:
        raise ValueError("start and end must share shape [B,...]")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha == 0.0:
        return start.float().clone()
    if alpha == 1.0:
        return end.float().clone()
    start = start.float()
    end = end.float()
    linear = torch.lerp(start, end, alpha)
    target = torch.lerp(sample_rms(start), sample_rms(end), alpha)
    return match_sample_rms(linear, target)


def spherical_interpolate(
    start: torch.Tensor,
    end: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Interpolate each flattened sample on a sphere with linearly varying radius."""

    if start.shape != end.shape or start.ndim < 2:
        raise ValueError("start and end must share shape [B,...]")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if alpha == 0.0:
        return start.float().clone()
    if alpha == 1.0:
        return end.float().clone()

    start = start.float()
    end = end.float()
    shape = start.shape
    first = start.flatten(1)
    second = end.flatten(1)
    first_norm = first.norm(dim=1, keepdim=True).clamp_min(1e-12)
    second_norm = second.norm(dim=1, keepdim=True).clamp_min(1e-12)
    first_unit = first / first_norm
    second_unit = second / second_norm
    cosine = (first_unit * second_unit).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    sine = torch.sin(angle)
    nearly_parallel = sine.abs() < 1e-6
    first_weight = torch.sin((1.0 - alpha) * angle) / sine.clamp_min(1e-12)
    second_weight = torch.sin(alpha * angle) / sine.clamp_min(1e-12)
    direction = first_weight * first_unit + second_weight * second_unit
    fallback = (1.0 - alpha) * first_unit + alpha * second_unit
    direction = torch.where(nearly_parallel, fallback, direction)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    radius = torch.lerp(first_norm, second_norm, alpha)
    return (direction * radius).reshape(shape)


def component_energy_fraction(component: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
    if component.shape != total.shape:
        raise ValueError("component and total must have equal shape")
    numerator = component.float().square().flatten(1).sum(dim=1)
    denominator = total.float().square().flatten(1).sum(dim=1).clamp_min(1e-12)
    return numerator / denominator


def feature_progress(
    candidate: torch.Tensor,
    good: torch.Tensor,
    bad: torch.Tensor,
) -> torch.Tensor:
    """Project candidate feature displacement onto the paired good-to-bad chord."""

    if candidate.shape != good.shape or good.shape != bad.shape or good.ndim != 2:
        raise ValueError("features must share shape [B,D]")
    chord = bad.float() - good.float()
    displacement = candidate.float() - good.float()
    denominator = chord.square().sum(dim=1).clamp_min(1e-12)
    return (displacement * chord).sum(dim=1) / denominator


def random_unit_directions(dimension: int, count: int, seed: int) -> torch.Tensor:
    if dimension < 1 or count < 1:
        raise ValueError("dimension and count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn((int(dimension), int(count)), generator=generator)
    return values / values.square().sum(dim=0, keepdim=True).sqrt().clamp_min(1e-12)


def standardized_sliced_wasserstein(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    directions: torch.Tensor,
) -> float:
    """Equal-sample SWD after standardizing with reference moments."""

    if reference.ndim != 2 or candidate.shape != reference.shape:
        raise ValueError("reference and candidate must share shape [N,D]")
    if directions.shape[0] != reference.shape[1]:
        raise ValueError("directions must have shape [D,K]")
    reference = reference.float()
    candidate = candidate.float()
    mean = reference.mean(dim=0, keepdim=True)
    scale = reference.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    ref_projection = ((reference - mean) / scale) @ directions.to(reference)
    candidate_projection = ((candidate - mean) / scale) @ directions.to(candidate)
    ref_projection = ref_projection.sort(dim=0).values
    candidate_projection = candidate_projection.sort(dim=0).values
    return float((ref_projection - candidate_projection).abs().mean())


def projected_frechet_distance(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    projection: torch.Tensor,
) -> float:
    """Compute a low-dimensional, small-sample feature Frechet proxy."""

    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("reference and candidate must be matrices")
    if reference.shape[1] != candidate.shape[1] or projection.shape[0] != reference.shape[1]:
        raise ValueError("feature and projection dimensions disagree")
    first = (reference.float() @ projection.to(reference)).double()
    second = (candidate.float() @ projection.to(candidate)).double()
    mean_delta = first.mean(dim=0) - second.mean(dim=0)

    def covariance(value: torch.Tensor) -> torch.Tensor:
        centered = value - value.mean(dim=0, keepdim=True)
        return centered.T @ centered / max(len(value) - 1, 1)

    covariance_first = covariance(first)
    covariance_second = covariance(second)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_first)
    first_root = (eigenvectors * eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.T
    middle = first_root @ covariance_second @ first_root
    middle = 0.5 * (middle + middle.T)
    root_trace = torch.linalg.eigvalsh(middle).clamp_min(0).sqrt().sum()
    value = (
        mean_delta.square().sum()
        + torch.trace(covariance_first)
        + torch.trace(covariance_second)
        - 2.0 * root_trace
    )
    return float(value.clamp_min(0))


@dataclass(frozen=True)
class PathDifferenceThresholds:
    endpoint_proxy_spearman: float = 0.80
    dose_spearman: float = 0.80
    required_pair_wins: int = 3
    correction_recovery: float = 0.20
    correction_control_margin: float = 0.10
    feature_progress_margin: float = 0.10


def path_difference_gate(
    distribution: pd.DataFrame,
    samples: pd.DataFrame,
    thresholds: PathDifferenceThresholds = PathDifferenceThresholds(),
) -> dict[str, object]:
    """Apply the preregistered gate for authorizing a fresh-seed confirmation."""

    required_distribution = {
        "pair",
        "condition",
        "alpha",
        "projected_frechet",
        "swd",
    }
    required_samples = {"pair", "condition", "feature_progress"}
    if missing := required_distribution.difference(distribution.columns):
        raise KeyError(f"distribution is missing {sorted(missing)}")
    if missing := required_samples.difference(samples.columns):
        raise KeyError(f"samples are missing {sorted(missing)}")

    endpoint_rows = []
    for source, fid in KNOWN_5K_FID.items():
        candidates = []
        for good, bad in PATH_PAIRS:
            pair = f"{good}_to_{bad}"
            if source == good:
                selected = distribution[
                    (distribution.pair == pair) & (distribution.condition == "own_a0")
                ]
            elif source == bad:
                selected = distribution[
                    (distribution.pair == pair) & (distribution.condition == "own_a100")
                ]
            else:
                continue
            if len(selected) == 1:
                candidates.append(selected.iloc[0])
        if not candidates:
            raise ValueError(f"no endpoint metric found for {source}")
        endpoint_rows.append(
            {
                "source": source,
                "known_fid": float(fid),
                "projected_frechet": float(pd.Series([r.projected_frechet for r in candidates]).median()),
                "swd": float(pd.Series([r.swd for r in candidates]).median()),
            }
        )
    endpoint_table = pd.DataFrame(endpoint_rows)
    proxy_spearman = {
        metric: float(endpoint_table.known_fid.corr(endpoint_table[metric], method="spearman"))
        for metric in ("projected_frechet", "swd")
    }
    primary_valid = proxy_spearman["projected_frechet"] >= thresholds.endpoint_proxy_spearman

    pair_rows = []
    for good, bad in PATH_PAIRS:
        pair = f"{good}_to_{bad}"
        values = distribution[distribution.pair == pair].set_index("condition")
        own = distribution[
            (distribution.pair == pair) & distribution.condition.str.startswith("own_a")
        ].sort_values("alpha")
        dose = float(own.alpha.corr(own.projected_frechet, method="spearman"))
        good_metric = float(values.loc["own_a0", "projected_frechet"])
        bad_metric = float(values.loc["own_a100", "projected_frechet"])
        corrected = float(values.loc["own_a75", "projected_frechet"])
        gap = bad_metric - good_metric
        recovery = (bad_metric - corrected) / max(gap, 1e-12)
        control_recoveries = []
        for condition in ("bad_shuffled", "bad_random"):
            metric = float(values.loc[condition, "projected_frechet"])
            control_recoveries.append((bad_metric - metric) / max(gap, 1e-12))
        correction_margin = recovery - max(control_recoveries)

        sample_values = samples[samples.pair == pair]
        progress = sample_values.groupby("condition").feature_progress.median()
        own_progress = float(progress["own_a25"])
        best_control_progress = max(float(progress["good_shuffled"]), float(progress["good_random"]))
        progress_margin = own_progress - best_control_progress
        pair_rows.append(
            {
                "pair": pair,
                "endpoint_gap": gap,
                "dose_spearman": dose,
                "correction_recovery": recovery,
                "correction_control_margin": correction_margin,
                "feature_progress_margin": progress_margin,
                "dose_win": bool(gap > 0 and dose >= thresholds.dose_spearman),
                "correction_win": bool(
                    gap > 0
                    and recovery >= thresholds.correction_recovery
                    and correction_margin >= thresholds.correction_control_margin
                ),
                "specificity_win": bool(progress_margin >= thresholds.feature_progress_margin),
            }
        )
    pair_table = pd.DataFrame(pair_rows)
    counts = {
        "dose_wins": int(pair_table.dose_win.sum()),
        "correction_wins": int(pair_table.correction_win.sum()),
        "specificity_wins": int(pair_table.specificity_win.sum()),
    }
    checks = {
        "endpoint_proxy_valid": bool(primary_valid),
        "dose_response": counts["dose_wins"] >= thresholds.required_pair_wins,
        "bad_endpoint_correction": counts["correction_wins"] >= thresholds.required_pair_wins,
        "sample_specificity": counts["specificity_wins"] >= thresholds.required_pair_wins,
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "proxy_spearman": proxy_spearman,
        "counts": counts,
        "endpoint_table": endpoint_table.to_dict(orient="records"),
        "pair_table": pair_table.to_dict(orient="records"),
    }
