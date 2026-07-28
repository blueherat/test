"""Offline screening utilities for well-conditioned RAE data-path schedules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Schedule:
    family: str
    floor: float
    shape: float

    @property
    def name(self) -> str:
        parameter = "p" if self.family == "floor_power" else "alpha"
        return f"{self.family}_floor{self.floor:g}_{parameter}{self.shape:g}"


def _validate_time(time: np.ndarray) -> np.ndarray:
    value = np.asarray(time, dtype=np.float64)
    if np.any(~np.isfinite(value)) or np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("time must be finite and lie in [0, 1]")
    return value


def _validate_floor(floor: float) -> float:
    value = float(floor)
    if not 0.0 <= value < 1.0:
        raise ValueError("floor must lie in [0, 1)")
    return value


def floor_power_coefficient(
    time: np.ndarray, *, floor: float, power: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return c(t) and c'(t) for a floored power schedule."""

    time = _validate_time(time)
    floor = _validate_floor(floor)
    power = float(power)
    if power < 1.0:
        raise ValueError("power must be at least 1")
    remaining = 1.0 - time
    coefficient = floor + (1.0 - floor) * remaining**power
    derivative = -(1.0 - floor) * power * remaining ** (power - 1.0)
    return coefficient, derivative


def floor_rational_coefficient(
    time: np.ndarray, *, floor: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return c(t) and c'(t) for a floored rational schedule."""

    time = _validate_time(time)
    floor = _validate_floor(floor)
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    denominator = 1.0 + alpha * time
    ratio = (1.0 - time) / denominator
    coefficient = floor + (1.0 - floor) * ratio
    derivative = -(1.0 - floor) * (1.0 + alpha) / denominator**2
    return coefficient, derivative


def schedule_coefficient(
    time: np.ndarray, schedule: Schedule
) -> tuple[np.ndarray, np.ndarray]:
    if schedule.family == "floor_power":
        return floor_power_coefficient(
            time, floor=schedule.floor, power=schedule.shape
        )
    if schedule.family == "floor_rational":
        return floor_rational_coefficient(
            time, floor=schedule.floor, alpha=schedule.shape
        )
    raise ValueError(f"unknown schedule family: {schedule.family}")


def endpoint_observation_factor(
    time: np.ndarray, schedule: Schedule
) -> np.ndarray:
    time = _validate_time(time)
    coefficient, derivative = schedule_coefficient(time, schedule)
    return coefficient - time * (1.0 - time) * derivative


@lru_cache(maxsize=None)
def delay_retention(schedule: Schedule) -> float:
    """Uniform-time delay area relative to the original (1-t)^2 schedule."""

    if schedule.family == "floor_power":
        candidate_delay = (1.0 - schedule.floor) * schedule.shape / (
            schedule.shape + 1.0
        )
    elif schedule.family == "floor_rational":
        alpha = schedule.shape
        coefficient_integral = -1.0 / alpha + (
            (1.0 + alpha) / alpha**2
        ) * np.log1p(alpha)
        candidate_delay = (1.0 - schedule.floor) * (
            1.0 - coefficient_integral
        )
    else:
        raise ValueError(f"unknown schedule family: {schedule.family}")
    return float(candidate_delay / (2.0 / 3.0))


@lru_cache(maxsize=None)
def minimum_observation_factor(schedule: Schedule) -> float:
    """Numerically verify and return the minimum k(t) on [0, 1]."""

    dense_time = np.linspace(0.0, 1.0, 10_001, dtype=np.float64)
    factor = endpoint_observation_factor(dense_time, schedule)
    if np.any(np.diff(factor) > 1e-10):
        raise ValueError(f"observation factor is not monotone for {schedule.name}")
    return float(factor.min())


def infer_raw_component_error(
    corrected_error: np.ndarray, observation_factor: np.ndarray
) -> np.ndarray:
    corrected = np.asarray(corrected_error, dtype=np.float64)
    factor = np.asarray(observation_factor, dtype=np.float64)
    if corrected.shape != factor.shape:
        raise ValueError("corrected_error and observation_factor must share shape")
    if np.any(factor <= 0.0):
        raise ValueError("observation_factor must be positive")
    return corrected * factor


def counterfactual_component_error(
    raw_error: np.ndarray, observation_factor: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw_error, dtype=np.float64)
    factor = np.asarray(observation_factor, dtype=np.float64)
    if raw.shape != factor.shape:
        raise ValueError("raw_error and observation_factor must share shape")
    if np.any(factor <= 0.0):
        raise ValueError("observation_factor must be positive")
    return raw / factor


def screen_schedules(
    annealed_metrics: pd.DataFrame,
    schedules: list[Schedule],
    *,
    semantic_weight: float,
    basis_weight: float,
    minimum_factor: float = 0.05,
    minimum_delay_retention: float = 0.70,
    maximum_excess_risk_ratio: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen schedules under a fixed-raw-error counterfactual assumption."""

    required = {
        "sample_index",
        "step_index",
        "time",
        "semantic_relative_error",
        "basis_relative_error",
        "basis_factor_abs",
    }
    if missing := required.difference(annealed_metrics.columns):
        raise KeyError(f"annealed_metrics is missing {sorted(missing)}")
    if semantic_weight <= 0.0 or basis_weight <= 0.0:
        raise ValueError("decoder weights must be positive")

    metrics = annealed_metrics.copy()
    if {
        "semantic_decoder_weight",
        "basis_decoder_weight",
    }.issubset(metrics.columns):
        semantic_weights = metrics["semantic_decoder_weight"].to_numpy(
            dtype=np.float64
        )
        basis_weights = metrics["basis_decoder_weight"].to_numpy(dtype=np.float64)
        if np.any(semantic_weights <= 0.0) or np.any(basis_weights <= 0.0):
            raise ValueError("per-row decoder weights must be positive")
    else:
        semantic_weights = np.full(len(metrics), float(semantic_weight))
        basis_weights = np.full(len(metrics), float(basis_weight))
    old_factor = metrics["basis_factor_abs"].to_numpy(dtype=np.float64)
    raw_error = infer_raw_component_error(
        metrics["basis_relative_error"].to_numpy(dtype=np.float64), old_factor
    )
    semantic_error = metrics["semantic_relative_error"].to_numpy(dtype=np.float64)
    baseline_risk = semantic_weights * semantic_error**2 + basis_weights * raw_error**2
    old_risk = semantic_weights * semantic_error**2 + basis_weights * (
        raw_error / old_factor
    ) ** 2
    old_excess = np.mean(old_risk - baseline_risk)
    if old_excess <= 0.0:
        raise ValueError("current schedule must have positive path-induced excess risk")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[pd.DataFrame] = []
    time = metrics["time"].to_numpy(dtype=np.float64)
    for schedule in schedules:
        factor = endpoint_observation_factor(time, schedule)
        candidate_error = counterfactual_component_error(raw_error, factor)
        candidate_risk = semantic_weights * semantic_error**2 + basis_weights * (
            candidate_error**2
        )
        candidate_excess = np.mean(candidate_risk - baseline_risk)
        risk_frame = pd.DataFrame(
            {
                "step_index": metrics["step_index"].to_numpy(),
                "candidate_excess": candidate_risk - baseline_risk,
                "old_excess": old_risk - baseline_risk,
            }
        )
        step_risk = risk_frame.groupby("step_index").mean()
        step_ratios = step_risk.candidate_excess / step_risk.old_excess
        worst_step_ratio = float(step_ratios.max())
        retention = delay_retention(schedule)
        min_factor = minimum_observation_factor(schedule)
        excess_ratio = float(candidate_excess / old_excess)
        total_ratio = float(candidate_risk.mean() / old_risk.mean())
        passes = bool(
            min_factor >= minimum_factor - 1e-12
            and retention >= minimum_delay_retention
            and excess_ratio <= maximum_excess_risk_ratio
            and worst_step_ratio <= maximum_excess_risk_ratio
        )
        summary_rows.append(
            {
                "schedule": schedule.name,
                "family": schedule.family,
                "floor": schedule.floor,
                "shape": schedule.shape,
                "min_observation_factor": min_factor,
                "max_inverse_amplification": 1.0 / min_factor,
                "delay_retention": retention,
                "total_risk_ratio": total_ratio,
                "path_excess_risk_ratio": excess_ratio,
                "worst_step_excess_risk_ratio": worst_step_ratio,
                "median_basis_error": float(np.median(candidate_error)),
                "p95_basis_error": float(np.quantile(candidate_error, 0.95)),
                "passes_gate": passes,
            }
        )
        details = metrics[
            ["sample_index", "step_index", "time", "semantic_relative_error"]
        ].copy()
        details["schedule"] = schedule.name
        details["old_factor"] = old_factor
        details["candidate_factor"] = factor
        details["raw_basis_error"] = raw_error
        details["candidate_basis_error"] = candidate_error
        details["baseline_weighted_risk"] = baseline_risk
        details["old_weighted_risk"] = old_risk
        details["candidate_weighted_risk"] = candidate_risk
        detail_rows.append(details)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["passes_gate", "path_excess_risk_ratio", "delay_retention"],
        ascending=[False, True, False],
    )
    return summary.reset_index(drop=True), pd.concat(detail_rows, ignore_index=True)


def random_control_table(
    *,
    channels: int,
    guided_rank: int,
    guided_explained_fraction: float,
) -> pd.DataFrame:
    """Expose the unavoidable rank-versus-energy tradeoff for random controls."""

    if not 0 < guided_rank <= channels:
        raise ValueError("guided_rank must lie in (0, channels]")
    if not 0.0 < guided_explained_fraction <= 1.0:
        raise ValueError("guided_explained_fraction must lie in (0, 1]")
    old_scale = np.sqrt(
        guided_explained_fraction / (float(guided_rank) / float(channels))
    )
    energy_rank = int(np.clip(round(guided_explained_fraction * channels), 1, channels))
    rows = [
        {
            "control": "old_scaled_rank_matched",
            "rank": guided_rank,
            "latent_scale": old_scale,
            "expected_energy_fraction": old_scale**2 * guided_rank / channels,
            "rank_matched": True,
            "energy_matched": True,
            "clean_path_geometry": bool(old_scale <= 1.0),
        },
        {
            "control": "same_rank_unscaled",
            "rank": guided_rank,
            "latent_scale": 1.0,
            "expected_energy_fraction": guided_rank / channels,
            "rank_matched": True,
            "energy_matched": False,
            "clean_path_geometry": True,
        },
        {
            "control": "energy_rank_unscaled",
            "rank": energy_rank,
            "latent_scale": 1.0,
            "expected_energy_fraction": energy_rank / channels,
            "rank_matched": energy_rank == guided_rank,
            "energy_matched": bool(
                abs(energy_rank / channels - guided_explained_fraction)
                <= 0.5 / channels + 1e-12
            ),
            "clean_path_geometry": True,
        },
    ]
    return pd.DataFrame(rows)
