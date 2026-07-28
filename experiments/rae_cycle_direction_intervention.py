"""Core controls and gates for the RAE cycle-direction intervention study."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch


GENERATED_SOURCES = ("static", "annealed", "reverse", "random")
INTERVENTION_CONDITIONS = ("own", "shuffled", "random", "opposite")


def sample_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Return one RMS value per sample."""

    if value.ndim < 2:
        raise ValueError("value must include batch and feature dimensions")
    result = value.float().square().flatten(1).mean(dim=1).sqrt()
    return result.clamp_min(float(eps)) if eps > 0 else result


def _expand(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return value.to(reference).reshape(-1, *([1] * (reference.ndim - 1)))


def match_sample_rms(direction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale each direction to the corresponding target RMS."""

    if direction.ndim < 2 or target.ndim != 1 or len(direction) != len(target):
        raise ValueError("direction and target must have compatible batch dimensions")
    return direction.float() * _expand(target / sample_rms(direction, 1e-12), direction)


def cyclic_derangement(count: int, seed: int) -> torch.Tensor:
    """Return a deterministic permutation with no fixed points."""

    if int(count) < 2:
        raise ValueError("a derangement requires at least two samples")
    offset = 1 + int(seed) % (int(count) - 1)
    return (torch.arange(int(count), dtype=torch.long) + offset) % int(count)


def matched_intervention_directions(
    residuals: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build own, shuffled, Gaussian, and opposite equal-RMS directions."""

    if residuals.ndim != 4 or len(residuals) < 2:
        raise ValueError("residuals must have shape [B,C,H,W] with B >= 2")
    residuals = residuals.float()
    target = sample_rms(residuals)
    permutation = cyclic_derangement(len(residuals), seed)
    shuffled = match_sample_rms(residuals[permutation], target)
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + 104_729)
    random = torch.randn(residuals.shape, generator=generator, dtype=torch.float32)
    random = match_sample_rms(random, target)
    return {
        "own": residuals,
        "shuffled": shuffled,
        "random": random,
        "opposite": -residuals,
    }


def interpolate_direction(
    latent: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if latent.shape != direction.shape:
        raise ValueError("latent and direction must have equal shape")
    return latent.float() + float(alpha) * direction.float()


def select_global_alpha(
    calibration: pd.DataFrame,
    *,
    feature_cosine_floor: float = 0.98,
) -> tuple[float, pd.DataFrame]:
    """Choose one alpha without consulting held-out intervention controls.

    The eligible alpha with the lowest mean of per-path median cycle ratios is
    selected. Every generated path must independently pass the feature-cosine
    guardrail. Ties prefer the smaller intervention.
    """

    required = {
        "source",
        "condition",
        "alpha",
        "cycle_ratio_to_base",
        "inception_cosine_to_base",
    }
    if missing := required.difference(calibration.columns):
        raise KeyError(f"calibration is missing {sorted(missing)}")
    own = calibration[
        (calibration["condition"] == "own")
        & (calibration["source"].isin(GENERATED_SOURCES))
        & (calibration["alpha"] > 0)
    ]
    if own.empty:
        raise ValueError("calibration has no generated own-direction rows")
    per_path = (
        own.groupby(["alpha", "source"], as_index=False)
        .agg(
            cycle_ratio_median=("cycle_ratio_to_base", "median"),
            feature_cosine_median=("inception_cosine_to_base", "median"),
            sample_count=("sample_index", "count"),
        )
        .sort_values(["alpha", "source"])
    )
    rows = []
    expected_sources = set(GENERATED_SOURCES)
    for alpha, values in per_path.groupby("alpha"):
        sources = set(values["source"])
        if sources != expected_sources:
            raise ValueError(
                f"alpha {alpha} has sources {sorted(sources)}, expected {sorted(expected_sources)}"
            )
        rows.append(
            {
                "alpha": float(alpha),
                "mean_path_median_cycle_ratio": float(values.cycle_ratio_median.mean()),
                "minimum_path_feature_cosine": float(values.feature_cosine_median.min()),
                "eligible": bool(
                    (values.feature_cosine_median >= float(feature_cosine_floor)).all()
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    eligible = summary[summary["eligible"]].sort_values(
        ["mean_path_median_cycle_ratio", "alpha"]
    )
    if eligible.empty:
        selected = float(summary.alpha.min())
        summary["fallback_to_smallest_alpha"] = True
    else:
        selected = float(eligible.iloc[0].alpha)
        summary["fallback_to_smallest_alpha"] = False
    summary["selected"] = summary.alpha.eq(selected)
    return selected, summary


@dataclass(frozen=True)
class CycleDirectionThresholds:
    primary_hidden_reduction: float = 0.50
    reverse_cycle_excess_reduction: float = 0.25
    path_cycle_wins: int = 3
    control_margin: float = 0.05
    feature_cosine_floor: float = 0.98


def cycle_direction_gate(
    test_samples: pd.DataFrame,
    thresholds: CycleDirectionThresholds = CycleDirectionThresholds(),
) -> dict[str, object]:
    """Apply the preregistered causal and semantic guardrails."""

    required = {
        "source",
        "condition",
        "cycle_relative_rms",
        "cycle_ratio_to_base",
        "inception_cosine_to_base",
        "primary_hidden_rms_z",
    }
    if missing := required.difference(test_samples.columns):
        raise KeyError(f"test samples are missing {sorted(missing)}")
    summary = (
        test_samples.groupby(["source", "condition"], as_index=False)
        .agg(
            cycle_median=("cycle_relative_rms", "median"),
            cycle_ratio_median=("cycle_ratio_to_base", "median"),
            feature_cosine_median=("inception_cosine_to_base", "median"),
            primary_hidden_z_median=("primary_hidden_rms_z", "median"),
        )
    )

    def row(source: str, condition: str) -> pd.Series:
        selected = summary[
            (summary.source == source) & (summary.condition == condition)
        ]
        if len(selected) != 1:
            raise ValueError(f"expected one summary row for {source}/{condition}")
        return selected.iloc[0]

    clean_base = row("clean_test", "baseline")
    clean_own = row("clean_test", "own")
    reverse_base = row("reverse", "baseline")
    reverse_own = row("reverse", "own")
    reverse_opposite = row("reverse", "opposite")

    base_hidden = abs(float(reverse_base.primary_hidden_z_median))
    own_hidden = abs(float(reverse_own.primary_hidden_z_median))
    hidden_reduction = 1.0 - own_hidden / max(base_hidden, 1e-12)
    clean_cycle = float(clean_base.cycle_median)
    base_excess = float(reverse_base.cycle_median) - clean_cycle
    own_excess = float(reverse_own.cycle_median) - clean_cycle
    cycle_excess_reduction = 1.0 - own_excess / max(base_excess, 1e-12)

    path_rows = []
    for source in GENERATED_SOURCES:
        base = row(source, "baseline")
        own = row(source, "own")
        shuffled = row(source, "shuffled")
        random = row(source, "random")
        path_rows.append(
            {
                "source": source,
                "own_cycle_ratio": float(own.cycle_ratio_median),
                "shuffled_cycle_ratio": float(shuffled.cycle_ratio_median),
                "random_cycle_ratio": float(random.cycle_ratio_median),
                "own_feature_cosine": float(own.feature_cosine_median),
                "cycle_win": bool(float(own.cycle_median) < float(base.cycle_median)),
            }
        )
    path_table = pd.DataFrame(path_rows)
    path_cycle_wins = int(path_table.cycle_win.sum())
    own_mean = float(path_table.own_cycle_ratio.mean())
    best_control_mean = min(
        float(path_table.shuffled_cycle_ratio.mean()),
        float(path_table.random_cycle_ratio.mean()),
    )
    control_advantage = best_control_mean - own_mean
    generated_feature_floor = float(path_table.own_feature_cosine.min())
    opposite_worsens = bool(
        float(reverse_opposite.cycle_median) > float(reverse_base.cycle_median)
        or abs(float(reverse_opposite.primary_hidden_z_median)) > base_hidden
    )

    checks = {
        "primary_hidden_reduction": bool(
            hidden_reduction >= thresholds.primary_hidden_reduction
        ),
        "reverse_cycle_excess_reduction": bool(
            cycle_excess_reduction >= thresholds.reverse_cycle_excess_reduction
        ),
        "path_cycle_wins": bool(path_cycle_wins >= thresholds.path_cycle_wins),
        "own_beats_controls": bool(control_advantage >= thresholds.control_margin),
        "opposite_worsens": opposite_worsens,
        "generated_feature_guardrail": bool(
            generated_feature_floor >= thresholds.feature_cosine_floor
        ),
        "clean_feature_guardrail": bool(
            float(clean_own.feature_cosine_median) >= thresholds.feature_cosine_floor
        ),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "primary_hidden_reduction": float(hidden_reduction),
        "reverse_cycle_excess_reduction": float(cycle_excess_reduction),
        "path_cycle_wins": path_cycle_wins,
        "control_advantage": float(control_advantage),
        "generated_feature_floor": generated_feature_floor,
        "clean_feature_cosine": float(clean_own.feature_cosine_median),
        "opposite_worsens": opposite_worsens,
        "per_path": path_rows,
        "summary": summary.to_dict(orient="records"),
    }


__all__ = [
    "CycleDirectionThresholds",
    "GENERATED_SOURCES",
    "INTERVENTION_CONDITIONS",
    "cycle_direction_gate",
    "cyclic_derangement",
    "interpolate_direction",
    "match_sample_rms",
    "matched_intervention_directions",
    "sample_rms",
    "select_global_alpha",
]
