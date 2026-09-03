#!/usr/bin/env python3
"""Paired ImageNet-100 study of path-aware Internal Guidance.

The production IG controller is the strongest paired FID-1K depth-4 setting:

    [0.00, 0.25): S + 0.6 * (S - W4)
    [0.25, 0.50): S + 0.7 * (S - W4)
    [0.50, 1.00]: S

For an interval [a, b], this script integrates both the strong field and the
production IG field from the same state and forms

    E_rho(z_a) = Phi_S(z_a) + rho * (Phi_IG(z_a) - Phi_S(z_a)).

``rho=1`` is exactly the ordinary IG endpoint for that interval.  The screen
compares whole-half and two-segment path extrapolation against directly
scaling the pointwise IG gamma.  All conditions use paired noise and labels.

The successful condition removes the local-gap term duplicated by one Euler
Foresight round trip and retains only the future weak-field residual:

    h = min(H, 0.5 - t)
    G = S + gamma(t) * (S - W4)
    z_future = z + h * G
    G_fmd = G + eta * (W4(z, t) - W4(z_future, t + h)).

For small H this is ``G - kappa * D_G W4 + O(kappa H)``, where
``kappa = eta * H``.  The future query evaluates only the depth-4 prefix; its
output is bitwise identical to evaluating the same head while running the
otherwise-unused backbone suffix.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.internal_guidance_path_extrapolation import (  # noqa: E402
    PathEndpointPair,
    affine_counterfactual_ratio_velocity,
    align_linear_path_state_to_endpoint_coordinate,
    calibration_split_foresight_velocity,
    counterfactual_telescoping_velocity,
    decompose_cross_time_velocity_change,
    decompose_endpoint_posterior_change,
    decompose_material_change,
    decompose_future_weak_drift,
    decompose_euler_foresight_roundtrip,
    extrapolate_path_endpoints,
    factorized_scale_space_guidance_velocity,
    finite_lie_bracket_change,
    forecast_weak_reference,
    foresight_weak_guidance,
    match_sample_rms,
    mix_characteristic_velocity,
    mix_material_curvature,
    project_per_sample,
    project_to_forward_ray,
    relax_future_weak_reference,
    richardson_forward_change,
    sample_rms,
    split_internal_guidance,
    telescoping_scale_space_guidance_velocity,
    transported_internal_gap_velocity,
)
from experiments.run_imagenet100_sit_internal_early_two_segment_gamma_sweep import (  # noqa: E402
    atomic_json,
    detect_adm_python,
    detect_data,
    detect_repo,
    load_repo_modules,
    parse_gpus,
    read_json,
    runtime_paths,
)
from experiments.semigroup_consistent_guidance import (  # noqa: E402
    local_jensen_velocity_correction,
)


EXPECTED_NOISE = "ab8419c7fdfd5b15dacbf4d37a3d567158e4332f25fd94580d3df73bac87e2c2"
EXPECTED_LABEL = "7c3ae6894e7ebab5c9b6524606f03b6a56b38dccbe472ff40edde26e48654fe6"
HISTORICAL_BEST_FID1K = 64.85428230760442
CONDITION_FORMAT = "eqvae_path_extrapolated_internal_guidance_condition_v6"
LEGACY_CONDITION_FORMAT = "eqvae_path_extrapolated_internal_guidance_condition_v5"
LEGACY_FORESIGHT_RESIDUAL_FORMULA = (
    "best_IG + rho*(RMS_match(g_future,g_current)-g_current)"
)
BASE_GAMMA_FIRST = 0.6
BASE_GAMMA_SECOND = 0.7
FIRST_END = 0.25
GUIDANCE_END = 0.5


def _float_tag(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"
    return text.replace("-", "m").replace(".", "p")


def _parse_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) or item < 0.0 for item in result):
        raise argparse.ArgumentTypeError("values must be finite and non-negative")
    return tuple(dict.fromkeys(result))


def _parse_signed_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from error
    if not result or any(not math.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("values must be finite")
    return tuple(dict.fromkeys(result))


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"whole", "segmented"}
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= allowed:
        raise argparse.ArgumentTypeError("path modes must be unique whole/segmented values")
    return modes


def _parse_future_components(value: str) -> tuple[str, ...]:
    components = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"full", "orthogonal"}
    if (
        not components
        or len(set(components)) != len(components)
        or not set(components) <= allowed
    ):
        raise argparse.ArgumentTypeError(
            "future components must be unique full/orthogonal values"
        )
    return components


def _parse_fmd_decomposition_components(value: str) -> tuple[str, ...]:
    components = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {
        "combined",
        "gap_change",
        "strong_curvature",
        "gap_change_matched",
        "strong_curvature_matched",
        "lookahead_weak",
        "picard_weak",
        "picard_weak_poly",
        "weak_drift_gap_parallel",
        "weak_drift_gap_orthogonal",
        "weak_reference_angle",
        "weak_reference_magnitude",
        "richardson_weak",
        "weak_drift_anisotropic",
        "weak_curvature_mix",
        "weak_drift_velocity_parallel",
        "weak_drift_velocity_orthogonal",
        "weak_velocity_angle",
        "weak_velocity_magnitude",
        "weak_lie_bracket",
        "weak_lie_bracket_matched",
        "weak_drift_midpoint",
        "weak_material_anisotropic",
        "weak_drift_segmented",
        "weak_drift_path_mix",
        "weak_drift_oblique",
        "weak_drift_oblique_curvature",
        "weak_drift_oblique_picard",
        "weak_clean_endpoint_drift",
        "weak_clean_endpoint_drift_matched",
        "weak_noise_endpoint_drift",
        "weak_noise_endpoint_drift_matched",
        "weak_endpoint_contrast",
        "weak_endpoint_contrast_segmented",
        "weak_reference_forecast",
        "weak_drift_extended",
        "weak_calibration_split",
        "weak_calibration_time_only",
        "weak_calibration_characteristic",
        "weak_calibration_weak_characteristic",
        "weak_calibration_strong_characteristic",
        "weak_calibration_projected",
        "weak_calibration_projected_coupled",
        "weak_calibration_reference_geomean",
        "weak_calibration_horizon_geomean",
        "weak_calibration_depth8_response",
        "weak_calibration_telescoping_depth8",
        "weak_gap_transport_time_only",
        "weak_gap_transport_projected",
        "strong_gap_transport_projected",
        "weak_gap_antitransport_projected",
        "score_noisier_aligned",
        "score_noisier_same_state",
        "score_cleaner_aligned",
        "velocity_noisier_aligned",
        "marginal_score_weak_noisier",
        "marginal_score_strong_noisier",
        "marginal_score_weak_cleaner",
        "velocity_parameterization_transport",
        "velocity_score_evolution",
        "velocity_change_recomposed",
        "weak_calibration_innovation",
        "weak_calibration_innovation_strong_axis",
        "weak_calibration_innovation_guided_axis",
        "weak_guidance_innovation",
    }
    if (
        not components
        or len(set(components)) != len(components)
        or not set(components) <= allowed
    ):
        raise argparse.ArgumentTypeError(
            "FMD components must be unique combined/gap_change/"
            "strong_curvature values, optionally suffixed by _matched"
        )
    return components


def _parse_fsg_bases(value: str) -> tuple[str, ...]:
    bases = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"strong", "ig"}
    if not bases or len(set(bases)) != len(bases) or not set(bases) <= allowed:
        raise argparse.ArgumentTypeError("FSG bases must be unique strong/ig values")
    return bases


def _parse_pairs(value: str) -> tuple[tuple[float, float], ...]:
    if not value.strip():
        return ()
    pairs: list[tuple[float, float]] = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 2:
            raise argparse.ArgumentTypeError("rho pairs require first:second entries")
        try:
            pair = (float(fields[0]), float(fields[1]))
        except ValueError as error:
            raise argparse.ArgumentTypeError("rho pairs must be numeric") from error
        if any(not math.isfinite(number) or number < 0.0 for number in pair):
            raise argparse.ArgumentTypeError("rho pairs must be finite and non-negative")
        pairs.append(pair)
    return tuple(dict.fromkeys(pairs))


def _parse_triples(value: str) -> tuple[tuple[float, float, float], ...]:
    if not value.strip():
        return ()
    triples: list[tuple[float, float, float]] = []
    for item in value.split(","):
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                "FSG budget specs require base:first:second entries"
            )
        try:
            triple = tuple(float(field) for field in fields)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "FSG budget specs must be numeric"
            ) from error
        if any(not math.isfinite(number) or number < 0.0 for number in triple):
            raise argparse.ArgumentTypeError(
                "FSG budget specs must be finite and non-negative"
            )
        if triple[0] > 1.0:
            raise argparse.ArgumentTypeError("FSG base fraction must lie in [0, 1]")
        triples.append(triple)
    return tuple(dict.fromkeys(triples))


def _parse_fsg_schedules(value: str) -> tuple[str, ...]:
    schedules = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"official", "balanced"}
    if (
        not schedules
        or len(set(schedules)) != len(schedules)
        or not set(schedules) <= allowed
    ):
        raise argparse.ArgumentTypeError(
            "FSG schedules must be unique official/balanced values"
        )
    return schedules


def _parse_positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("iteration counts must be positive")
    return tuple(dict.fromkeys(result))


@dataclass(frozen=True)
class Condition:
    kind: str
    rho_first: float = 1.0
    rho_second: float = 1.0
    probe_multiplier: float = 1.0
    local_multiplier: float = 1.0
    roundtrip_iterations: int = 1
    roundtrip_multiplier: float = 1.0
    residual_strength: float = 0.0
    partition_count: int = 2
    fsg_base_fraction: float = 0.0

    def validate(self) -> None:
        if self.kind not in {
            "global",
            "path_whole",
            "path_segmented",
            "roundtrip_whole",
            "roundtrip_segmented",
            "roundtrip_partitioned",
            "roundtrip_residual_whole",
            "roundtrip_residual_segmented",
            "future_gap_full_partitioned",
            "future_gap_orthogonal_partitioned",
            "fsg_strong_base",
            "fsg_ig_base",
            "fsg_hybrid_base",
            "fsg_budgeted_official",
            "fsg_budgeted_balanced",
            "fsg_early_official",
            "fsg_early_balanced",
            "fsg_residual_continuous",
            "fmd_combined_continuous",
            "fmd_gap_change_continuous",
            "fmd_strong_curvature_continuous",
            "fmd_gap_change_matched_continuous",
            "fmd_strong_curvature_matched_continuous",
            "fmd_lookahead_weak_continuous",
            "fmd_picard_weak_continuous",
            "fmd_picard_weak_poly_continuous",
            "fmd_weak_drift_gap_parallel_continuous",
            "fmd_weak_drift_gap_orthogonal_continuous",
            "fmd_weak_reference_angle_continuous",
            "fmd_weak_reference_magnitude_continuous",
            "fmd_richardson_weak_continuous",
            "fmd_weak_drift_anisotropic_continuous",
            "fmd_weak_curvature_mix_continuous",
            "fmd_weak_drift_velocity_parallel_continuous",
            "fmd_weak_drift_velocity_orthogonal_continuous",
            "fmd_weak_velocity_angle_continuous",
            "fmd_weak_velocity_magnitude_continuous",
            "fmd_weak_lie_bracket_continuous",
            "fmd_weak_lie_bracket_matched_continuous",
            "fmd_weak_drift_midpoint_continuous",
            "fmd_weak_material_anisotropic_continuous",
            "fmd_weak_drift_segmented_continuous",
            "fmd_weak_drift_path_mix_continuous",
            "fmd_weak_drift_oblique_continuous",
            "fmd_weak_drift_oblique_curvature_continuous",
            "fmd_weak_drift_oblique_picard_continuous",
            "fmd_weak_clean_endpoint_drift_continuous",
            "fmd_weak_clean_endpoint_drift_matched_continuous",
            "fmd_weak_noise_endpoint_drift_continuous",
            "fmd_weak_noise_endpoint_drift_matched_continuous",
            "fmd_weak_endpoint_contrast_continuous",
            "fmd_weak_endpoint_contrast_segmented_continuous",
            "fmd_weak_reference_forecast_continuous",
            "fmd_weak_drift_extended_continuous",
            "fmd_weak_calibration_split_continuous",
            "fmd_weak_calibration_time_only_continuous",
            "fmd_weak_calibration_characteristic_continuous",
            "fmd_weak_calibration_weak_characteristic_continuous",
            "fmd_weak_calibration_strong_characteristic_continuous",
            "fmd_weak_calibration_projected_continuous",
            "fmd_weak_calibration_projected_coupled_continuous",
            "fmd_weak_calibration_reference_geomean_continuous",
            "fmd_weak_calibration_horizon_geomean_continuous",
            "fmd_weak_calibration_depth8_response_continuous",
            "fmd_weak_calibration_telescoping_depth8_continuous",
            "fmd_weak_gap_transport_time_only_continuous",
            "fmd_weak_gap_transport_projected_continuous",
            "fmd_strong_gap_transport_projected_continuous",
            "fmd_weak_gap_antitransport_projected_continuous",
            "fmd_score_noisier_aligned_continuous",
            "fmd_score_noisier_same_state_continuous",
            "fmd_score_cleaner_aligned_continuous",
            "fmd_velocity_noisier_aligned_continuous",
            "fmd_marginal_score_weak_noisier_continuous",
            "fmd_marginal_score_strong_noisier_continuous",
            "fmd_marginal_score_weak_cleaner_continuous",
            "fmd_velocity_parameterization_transport_continuous",
            "fmd_velocity_score_evolution_continuous",
            "fmd_velocity_change_recomposed_continuous",
            "fmd_weak_calibration_innovation_continuous",
            "fmd_weak_calibration_innovation_strong_axis_continuous",
            "fmd_weak_calibration_innovation_guided_axis_continuous",
            "fmd_weak_guidance_innovation_continuous",
            "semigroup_local_jensen",
            "local_scaled",
            "local_scaled_whole",
            "local_scaled_segmented",
        }:
            raise ValueError(f"unsupported condition kind: {self.kind}")
        values = (
            self.rho_first,
            self.rho_second,
            self.probe_multiplier,
            self.local_multiplier,
            self.roundtrip_multiplier,
        )
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
            raise ValueError("condition coefficients must be finite and non-negative")
        if self.kind == "path_whole" and self.rho_first != self.rho_second:
            raise ValueError("whole-interval path extrapolation uses one shared rho")
        if self.kind.startswith("path_") and self.probe_multiplier <= 0.0:
            raise ValueError("path probe multiplier must be positive")
        if self.roundtrip_iterations <= 0:
            raise ValueError("roundtrip iterations must be positive")
        if not math.isfinite(float(self.residual_strength)):
            raise ValueError("residual strength must be finite")
        if self.partition_count <= 0 or self.partition_count % 2:
            raise ValueError("partition count must be a positive even integer")
        if not 0.0 <= float(self.fsg_base_fraction) <= 1.0:
            raise ValueError("FSG base fraction must lie in [0, 1]")
        if (
            self.kind == "fsg_residual_continuous"
            or self.kind.startswith("fmd_")
        ) and self.rho_first <= 0.0:
            raise ValueError("continuous foresight horizon must be positive")
        if self.kind in {
            "fmd_lookahead_weak_continuous",
            "fmd_picard_weak_continuous",
            "fmd_picard_weak_poly_continuous",
        } and self.residual_strength < 0:
            raise ValueError("lookahead weak coefficient must be non-negative")
        if self.kind == "fmd_weak_drift_extended_continuous" and not (
            GUIDANCE_END < self.rho_second <= 1.0
        ):
            raise ValueError("extended FMD correction end must lie in (0.5, 1]")

    @property
    def name(self) -> str:
        self.validate()
        if self.kind == "global":
            return "ig_depth4_best_global"
        if self.kind == "semigroup_local_jensen":
            return "ig_depth4_semigroup_local_jensen"
        if self.kind == "path_whole":
            probe = (
                ""
                if self.probe_multiplier == 1.0
                else f"_alpha{_float_tag(self.probe_multiplier)}"
            )
            return f"path_whole{probe}_rho{_float_tag(self.rho_first)}"
        if self.kind == "path_segmented":
            probe = (
                ""
                if self.probe_multiplier == 1.0
                else f"_alpha{_float_tag(self.probe_multiplier)}"
            )
            return (
                f"path_segmented{probe}_rho1{_float_tag(self.rho_first)}"
                f"_rho2{_float_tag(self.rho_second)}"
            )
        if self.kind.startswith("roundtrip_"):
            if self.kind == "roundtrip_partitioned":
                return (
                    f"roundtrip_partitioned_n{self.partition_count}"
                    f"_m{_float_tag(self.roundtrip_multiplier)}"
                )
            if self.kind.startswith("roundtrip_residual_"):
                mode = self.kind.removeprefix("roundtrip_residual_")
                return (
                    f"roundtrip_residual_{mode}"
                    f"_eta{_float_tag(self.residual_strength)}"
                )
            mode = self.kind.removeprefix("roundtrip_")
            return (
                f"roundtrip_{mode}_k{self.roundtrip_iterations}"
                f"_m{_float_tag(self.roundtrip_multiplier)}"
            )
        if self.kind.startswith("future_gap_"):
            component = self.kind.removeprefix("future_gap_").removesuffix(
                "_partitioned"
            )
            return (
                f"future_gap_{component}_n{self.partition_count}"
                f"_rho{_float_tag(self.residual_strength)}"
            )
        if self.kind.startswith("fsg_budgeted_"):
            schedule = self.kind.removeprefix("fsg_budgeted_")
            return (
                f"fsg_budgeted_{schedule}"
                f"_lam{_float_tag(self.fsg_base_fraction)}"
                f"_rho1{_float_tag(self.rho_first)}"
                f"_rho2{_float_tag(self.rho_second)}"
            )
        if self.kind.startswith("fsg_early_"):
            schedule = self.kind.removeprefix("fsg_early_")
            return f"fsg_early_{schedule}_rho{_float_tag(self.residual_strength)}"
        if self.kind == "fsg_residual_continuous":
            return (
                f"fsg_residual_continuous_h{_float_tag(self.rho_first)}"
                f"_eta{_float_tag(self.residual_strength)}"
            )
        if self.kind.startswith("fmd_"):
            component = self.kind.removeprefix("fmd_").removesuffix("_continuous")
            coefficient = "alpha" if component == "lookahead_weak" else "eta"
            gamma_first_multiplier = self.local_multiplier * self.probe_multiplier
            gamma_second_multiplier = self.local_multiplier
            if gamma_first_multiplier == gamma_second_multiplier == 1.0:
                gamma_suffix = ""
            elif gamma_first_multiplier == gamma_second_multiplier:
                gamma_suffix = f"_gm{_float_tag(gamma_first_multiplier)}"
            else:
                gamma_suffix = (
                    f"_gm1{_float_tag(gamma_first_multiplier)}"
                    f"_gm2{_float_tag(gamma_second_multiplier)}"
                )
            iterations = (
                f"_k{self.roundtrip_iterations}"
                if component == "picard_weak"
                else ""
            )
            if component == "picard_weak_poly":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta1{_float_tag(self.residual_strength)}"
                    f"_eta2{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_drift_anisotropic":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_orth{_float_tag(self.residual_strength)}"
                    f"_parallel{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_curvature_mix":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta{_float_tag(self.residual_strength)}"
                    f"_lambda{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_material_anisotropic":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_time{_float_tag(self.residual_strength)}"
                    f"_state{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_drift_segmented":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta1{_float_tag(self.residual_strength)}"
                    f"_eta2{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_drift_path_mix":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta{_float_tag(self.residual_strength)}"
                    f"_rho{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component in {
                "weak_drift_oblique",
                "weak_drift_oblique_picard",
            }:
                iterations = (
                    f"_k{self.roundtrip_iterations}"
                    if component == "weak_drift_oblique_picard"
                    else ""
                )
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta{_float_tag(self.residual_strength)}"
                    f"_alpha{_float_tag(self.roundtrip_multiplier)}"
                    f"{iterations}"
                    f"{gamma_suffix}"
                )
            if component == "weak_drift_oblique_curvature":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta{_float_tag(self.residual_strength)}"
                    f"_alpha{_float_tag(self.roundtrip_multiplier)}"
                    f"_lambda{_float_tag(self.rho_second)}"
                    f"{gamma_suffix}"
                )
            if component in {
                "weak_clean_endpoint_drift",
                "weak_clean_endpoint_drift_matched",
                "weak_noise_endpoint_drift",
                "weak_noise_endpoint_drift_matched",
            }:
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta{_float_tag(self.residual_strength)}"
                    f"_alpha{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_endpoint_contrast":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta{_float_tag(self.residual_strength)}"
                    f"_alpha{_float_tag(self.roundtrip_multiplier)}"
                    f"_lambda{_float_tag(self.rho_second)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_endpoint_contrast_segmented":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_eta1{_float_tag(self.residual_strength)}"
                    f"_eta2{_float_tag(self.rho_second)}"
                    f"_alpha{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_reference_forecast":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_r{_float_tag(self.residual_strength)}"
                    f"_alpha{_float_tag(self.roundtrip_multiplier)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_drift_extended":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"_early{_float_tag(self.residual_strength)}"
                    f"_late{_float_tag(self.roundtrip_multiplier)}"
                    f"_end{_float_tag(self.rho_second)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_calibration_split":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"{gamma_suffix}"
                )
            if component in {
                "weak_calibration_time_only",
                "weak_calibration_characteristic",
                "weak_calibration_weak_characteristic",
                "weak_calibration_strong_characteristic",
                "weak_calibration_projected",
                "weak_calibration_projected_coupled",
                "weak_calibration_reference_geomean",
                "weak_calibration_horizon_geomean",
                "weak_calibration_depth8_response",
                "weak_calibration_telescoping_depth8",
                "weak_gap_transport_time_only",
                "weak_gap_transport_projected",
                "strong_gap_transport_projected",
                "weak_gap_antitransport_projected",
                "score_noisier_aligned",
                "score_noisier_same_state",
                "score_cleaner_aligned",
                "velocity_noisier_aligned",
                "marginal_score_weak_noisier",
                "marginal_score_strong_noisier",
                "marginal_score_weak_cleaner",
                "velocity_parameterization_transport",
                "velocity_score_evolution",
                "velocity_change_recomposed",
            }:
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"{gamma_suffix}"
                )
            if component == "weak_calibration_innovation":
                return (
                    f"fmd_decomposition_{component}"
                    f"_h{_float_tag(self.rho_first)}"
                    f"{gamma_suffix}"
                )
            return (
                f"fmd_decomposition_{component}"
                f"_h{_float_tag(self.rho_first)}"
                f"_{coefficient}{_float_tag(self.residual_strength)}"
                f"{iterations}"
                f"{gamma_suffix}"
            )
        if self.kind.startswith("fsg_"):
            if self.kind == "fsg_hybrid_base":
                return (
                    f"fsg_hybrid_lam{_float_tag(self.fsg_base_fraction)}"
                    f"_rho{_float_tag(self.residual_strength)}"
                )
            base = self.kind.removeprefix("fsg_").removesuffix("_base")
            return f"fsg_{base}_base_rho{_float_tag(self.residual_strength)}"
        prefix = {
            "local_scaled": "local_global_gamma_multiplier",
            "local_scaled_whole": "local_whole_gamma_multiplier",
            "local_scaled_segmented": "local_segmented_gamma_multiplier",
        }[self.kind]
        return f"{prefix}{_float_tag(self.local_multiplier)}"

    def payload(self) -> dict[str, Any]:
        self.validate()
        if self.kind == "semigroup_local_jensen":
            foresight_formula = (
                "G=S+gamma*(S-W4); beta=1+gamma; "
                "G_scg=G+beta*(beta-1)*t*(1-t)*J_(S-W4)^T*(S-W4)"
            )
            material_derivative = {
                "endpoint_target": "p_strong^beta * p_weak^(1-beta)",
                "missing_potential": (
                    "log E_q[r^beta|z_t] - beta*log E_q[r|z_t]"
                ),
                "approximation": "first local heat-time term",
                "extra_tunable_coefficients": 0,
            }
        elif self.kind == "fsg_residual_continuous":
            foresight_formula = (
                "G(z,t) + eta * [W4(z,t) - "
                "W4(z + h*G(z,t), t+h)], h=min(H,0.5-t)"
            )
            material_derivative = {
                "lookahead_horizon": float(self.rho_first),
                "finite_difference_strength": float(self.residual_strength),
                "peak_material_derivative_gain": float(
                    self.rho_first * self.residual_strength
                ),
                "small_h_limit": "G - kappa*(partial_t W4 + J_W4 G)",
                "boundary_taper": "kappa(t)=eta*min(H,0.5-t)",
            }
        elif self.kind.startswith("fmd_"):
            component = self.kind.removeprefix("fmd_").removesuffix("_continuous")
            foresight_formula = (
                "G(z,t) + eta*C(z,t), z+=z+h*G, "
                "C_weak=W-W+=(g+-g)+(S-S+)"
            )
            material_derivative = {
                "lookahead_horizon": float(self.rho_first),
                "finite_difference_strength": float(self.residual_strength),
                "selected_component": component,
                "exact_identity": "W-W_future=(g_future-g)+(S-S_future)",
                "matched_reference": (
                    "per-sample RMS of W-W_future"
                    if component.endswith("_matched")
                    else None
                ),
                "boundary_taper": "h=min(H,0.5-t)",
            }
            gamma_first_multiplier = self.local_multiplier * self.probe_multiplier
            gamma_second_multiplier = self.local_multiplier
            if gamma_first_multiplier != 1.0 or gamma_second_multiplier != 1.0:
                material_derivative["gamma_multipliers"] = {
                    "first_interval": float(gamma_first_multiplier),
                    "second_interval": float(gamma_second_multiplier),
                }
            if component == "lookahead_weak":
                foresight_formula = (
                    "z+=z+h*G; W_alpha=(1-alpha)*W(z,t)+alpha*W(z+,t+h); "
                    "G_look=S+gamma*(S-W_alpha)"
                )
                material_derivative["coefficient_semantics"] = (
                    "alpha=0 ordinary IG; alpha=1 strong-minus-future-weak"
                )
            elif component == "picard_weak":
                foresight_formula = (
                    "V0=S+gamma*(S-W0); repeat K times: "
                    "W+=W(z+h*V,t+h); rho=eta/gamma; "
                    "V<-(1-rho)*V+rho*[S+gamma*(S-W+)]"
                )
                material_derivative["picard_iterations"] = int(
                    self.roundtrip_iterations
                )
                material_derivative["coefficient_semantics"] = (
                    "eta is the historical FMD coefficient; rho=eta/gamma(t)"
                )
            elif component == "picard_weak_poly":
                foresight_formula = (
                    "V0=S+gamma*(S-W0); perform two relaxed reference-Picard "
                    "updates with eta1 then eta2"
                )
                material_derivative["picard_iterations"] = 2
                material_derivative["first_strength"] = float(
                    self.residual_strength
                )
                material_derivative["second_strength"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["coefficient_semantics"] = (
                    "eta1 controls the first weak-Jacobian term; eta2 controls "
                    "the second fixed-point residual independently"
                )
            elif component == "weak_drift_anisotropic":
                foresight_formula = (
                    "C=W-W+; split C parallel/orthogonal to gamma*(S-W); "
                    "G_new=G+eta_parallel*C_parallel+eta_orthogonal*C_orthogonal"
                )
                material_derivative["orthogonal_strength"] = float(
                    self.residual_strength
                )
                material_derivative["parallel_strength"] = float(
                    self.roundtrip_multiplier
                )
            elif component == "weak_curvature_mix":
                foresight_formula = (
                    "C_first=4*(W-W_half)-(W-W_future); "
                    "C_lambda=C_first+lambda*((W-W_future)-C_first); "
                    "G_new=G+eta*C_lambda"
                )
                material_derivative["curvature_weight"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["coefficient_semantics"] = (
                    "lambda=0 Richardson first-material term; "
                    "lambda=1 historical finite-horizon correction"
                )
            elif component == "weak_material_anisotropic":
                foresight_formula = (
                    "C_time=W(z,t)-W(z,t+h); "
                    "C_state=W(z,t+h)-W(z+h*G,t+h); "
                    "G_new=G+eta_time*C_time+eta_state*C_state"
                )
                material_derivative["temporal_strength"] = float(
                    self.residual_strength
                )
                material_derivative["advective_strength"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["exact_identity"] = (
                    "C_time+C_state=W(z,t)-W(z+h*G,t+h)"
                )
            elif component == "weak_drift_segmented":
                foresight_formula = (
                    "G_new=G+eta(t)*(W-W_future), with independent eta on "
                    "[0,.25) and [.25,.5)"
                )
                material_derivative["segment_strengths"] = {
                    "first_interval": float(self.residual_strength),
                    "second_interval": float(self.roundtrip_multiplier),
                }
            elif component == "weak_drift_path_mix":
                foresight_formula = (
                    "P_rho=(1-rho)*W+rho*G; "
                    "G_new=G+eta*(W(z,t)-W(z+h*P_rho,t+h))"
                )
                material_derivative["characteristic_mix"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["coefficient_semantics"] = (
                    "rho=0 weak characteristic; rho=1 historical guided "
                    "characteristic; rho=1/(1+gamma) strong characteristic"
                )
            elif component == "weak_drift_oblique":
                foresight_formula = (
                    "G_new=G+eta*(W(z,t)-W(z+alpha*h*G,t+h)); "
                    "alpha=0 is pure temporal change and alpha=1 is the "
                    "historical guided-characteristic correction"
                )
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["small_h_limit"] = (
                    "G_new=G-eta*h*(partial_t W+alpha*J_W*G)+O(h^2)"
                )
            elif component == "weak_drift_oblique_curvature":
                foresight_formula = (
                    "C_h=W-W(z+alpha*h*G,t+h); "
                    "C_R=4*(W-W(z+alpha*h*G/2,t+h/2))-C_h; "
                    "G_new=G+eta*(C_R+lambda*(C_h-C_R))"
                )
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["curvature_weight"] = float(self.rho_second)
                material_derivative["coefficient_semantics"] = (
                    "lambda=0 removes the second-order finite-horizon term; "
                    "lambda=1 is exactly the historical oblique correction"
                )
            elif component in {
                "weak_clean_endpoint_drift",
                "weak_clean_endpoint_drift_matched",
                "weak_noise_endpoint_drift",
                "weak_noise_endpoint_drift_matched",
            }:
                endpoint = "clean" if "clean" in component else "negative_noise"
                foresight_formula = (
                    "xhat_W=z+(1-t)*W; epshat_W=z-t*W; "
                    "C_x=xhat_W(z,t)-xhat_W(q,t+h); "
                    "C_eps=epshat_W(q,t+h)-epshat_W(z,t); "
                    "C_x+C_eps=W(z,t)-W(q,t+h)"
                )
                material_derivative["selected_endpoint_component"] = endpoint
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["rms_matched_to_velocity_change"] = bool(
                    component.endswith("_matched")
                )
                material_derivative["exact_identity"] = (
                    "clean_change+negative_noise_change=weak_velocity_change"
                )
            elif component == "weak_endpoint_contrast":
                foresight_formula = (
                    "C_lambda=C_x+lambda*C_negative_eps="
                    "C_velocity+(lambda-1)*C_negative_eps; "
                    "lambda=1 uniquely cancels the common query displacement"
                )
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["negative_noise_weight"] = float(
                    self.rho_second
                )
                material_derivative["coordinate_invariant_weight"] = 1.0
            elif component == "weak_endpoint_contrast_segmented":
                foresight_formula = (
                    "G_new=G+eta(t)*(W(z,t)-W(z+alpha*h*G,t+h)); "
                    "eta1 applies on [0,.25), eta2 on [.25,.5)"
                )
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["segment_strengths"] = {
                    "first_interval": float(self.residual_strength),
                    "second_interval": float(self.rho_second),
                }
                material_derivative["endpoint_contrast"] = (
                    "clean_change+negative_noise_change=weak_velocity_change"
                )
            elif component == "weak_drift_oblique_picard":
                foresight_formula = (
                    "V0=G; V{k+1}=G+eta*(W(z,t)-"
                    "W(z+alpha*h*Vk,t+h)); repeat K times"
                )
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["picard_iterations"] = int(
                    self.roundtrip_iterations
                )
                material_derivative["fixed_point"] = (
                    "V=G+eta*(W(z,t)-W(z+alpha*h*V,t+h))"
                )
            elif component == "weak_reference_forecast":
                foresight_formula = (
                    "W_eff=(1-r)*W(z,t)+r*W(z+alpha*h*G,t+h); "
                    "G_new=S+gamma*(S-W_eff)"
                )
                material_derivative["forecast_factor"] = float(
                    self.residual_strength
                )
                material_derivative["spatial_transport_fraction"] = float(
                    self.roundtrip_multiplier
                )
                material_derivative["coefficient_semantics"] = (
                    "r=1 replaces the current weak reference by its future "
                    "query; r=2 linearly extrapolates the weak reference one "
                    "equal query displacement beyond that future query"
                )
            elif component == "weak_drift_extended":
                foresight_formula = (
                    "G_new=G+eta(t)*(W(z,t)-W(z+h*G,t+h)); ordinary IG "
                    "ends at .5 while the finite weak innovation may continue"
                )
                material_derivative["segment_strengths"] = {
                    "ig_interval": float(self.residual_strength),
                    "post_ig_interval": float(self.roundtrip_multiplier),
                }
                material_derivative["correction_end"] = float(self.rho_second)
                material_derivative["boundary_taper"] = (
                    "h=min(H, correction_end-t)"
                )
            elif component == "weak_calibration_split":
                foresight_formula = (
                    "beta=1+gamma; C=beta*(S-W); q=(z+h*C,t+h); "
                    "G_new=W+beta*(S-W(q))"
                )
                material_derivative["coefficient_semantics"] = (
                    "both query displacement and correction strength are "
                    "fixed by the exact IG split G=W+(1+gamma)*(S-W)"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_calibration_time_only":
                foresight_formula = (
                    "beta=1+gamma; q=(z,t+h); "
                    "G_new=W+beta*(S-W(q))"
                )
                material_derivative["query_geometry"] = (
                    "pure information-time intervention with fixed latent"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_calibration_characteristic":
                foresight_formula = (
                    "beta=1+gamma; q=(z+h*G,t+h); "
                    "G_new=W+beta*(S-W(q))"
                )
                material_derivative["query_geometry"] = (
                    "one frozen-Euler step along the deployed guided field"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "weak_calibration_weak_characteristic",
                "weak_calibration_strong_characteristic",
            }:
                probe = "W" if "weak_characteristic" in component else "S"
                foresight_formula = (
                    f"beta=1+gamma; q=(z+h*{probe},t+h); "
                    "G_new=W+beta*(S-W(q))"
                )
                material_derivative["query_geometry"] = (
                    f"one frozen-Euler step along the {probe} field"
                )
                material_derivative["posterior_pressure_probe"] = (
                    probe == "W"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_calibration_projected":
                foresight_formula = (
                    "beta=1+gamma; C=beta*(S-W); "
                    "a*=argmin_{a>=0}||a*G-C||^2; "
                    "q=(z+h*a*G,t+h); G_new=W+beta*(S-W(q))"
                )
                material_derivative["query_geometry"] = (
                    "information time advances by h while the spatial probe "
                    "is the Euclidean projection of calibration onto the "
                    "forward guided ray"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_calibration_projected_coupled":
                foresight_formula = (
                    "beta=1+gamma; C=beta*(S-W); "
                    "a*=argmin_{a>=0}||a*G-C||^2; "
                    "q=(z+h*a*G,t+h*a); G_new=W+beta*(S-W(q))"
                )
                material_derivative["query_geometry"] = (
                    "space and information time share the same projected "
                    "forward-ray coefficient"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "weak_gap_transport_time_only",
                "weak_gap_transport_projected",
                "strong_gap_transport_projected",
                "weak_gap_antitransport_projected",
            }:
                time_only = component == "weak_gap_transport_time_only"
                anchor = "strong" if component.startswith("strong_") else "weak"
                sign = -1.0 if "antitransport" in component else 1.0
                query = "q=(z,t+h)" if time_only else (
                    "C=beta*(S-W); a*=argmin_{a>=0}||a*G-C||^2; "
                    "q=(z+h*a*G,t+h)"
                )
                coefficient = "gamma" if anchor == "strong" else "beta"
                base = "S" if anchor == "strong" else "W"
                foresight_formula = (
                    f"beta=1+gamma; {query}; gp=S-W; gq=S(q)-W(q); "
                    f"G_new={base}+{coefficient}*(gp{'+(gq-gp)' if sign > 0 else '-(gq-gp)'})"
                )
                material_derivative["query_geometry"] = (
                    "pure information-time intervention with fixed latent"
                    if time_only
                    else "projected deep-refinement intervention"
                )
                material_derivative["factorial_interaction"] = (
                    "Omega=(S(q)-W(q))-(S-W)"
                )
                material_derivative["common_mode_invariant"] = True
                material_derivative["diagonal_consistent"] = True
                material_derivative["anchor"] = anchor
                material_derivative["interaction_sign"] = sign
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "score_noisier_aligned",
                "score_noisier_same_state",
                "score_cleaner_aligned",
                "velocity_noisier_aligned",
            }:
                noisier = "cleaner" not in component
                aligned = "same_state" not in component
                score_space = component.startswith("score_")
                direction = "t-H" if noisier else "t+H"
                state_query = (
                    "z_ref=(t_ref/t)*z"
                    if aligned
                    else "z_ref=z"
                )
                if score_space:
                    foresight_formula = (
                        f"t_ref={direction}; {state_query}; "
                        "r=t*(t*v-z)/(1-t); "
                        "r_new=r_S+gamma*(r_S-r_W_ref); "
                        "convert r_new back to current-time velocity"
                    )
                else:
                    foresight_formula = (
                        f"t_ref={direction}; {state_query}; "
                        "G_new=S+gamma*(S-W_ref) in raw velocity space"
                    )
                material_derivative["density_ratio_target"] = (
                    "q_S,current * (q_S,current/q_W,reference)^gamma"
                    if score_space
                    else None
                )
                material_derivative["reference_noise"] = (
                    "noisier" if noisier else "cleaner"
                )
                material_derivative["endpoint_coordinate_aligned"] = aligned
                material_derivative["parameterization"] = (
                    "endpoint_normalized_score" if score_space else "raw_velocity"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "marginal_score_weak_noisier",
                "marginal_score_strong_noisier",
                "marginal_score_weak_cleaner",
            }:
                temporal_branch = (
                    "strong" if "strong" in component else "weak"
                )
                reference_noise = (
                    "cleaner" if "cleaner" in component else "noisier"
                )
                direction = "t+H" if reference_noise == "cleaner" else "t-H"
                foresight_formula = (
                    f"t_ref={direction}; z_ref=z; "
                    "s=(t*v-z)/(1-t); a(t)=min(1,t/H); "
                    "s_new=s_IG+gamma*a(t)*(s_branch(t)-s_branch(t_ref)); "
                    "convert s_new back to current-time velocity"
                )
                material_derivative["density_ratio_target"] = (
                    "p_IG,current * "
                    "(p_branch,current/p_branch,reference)^(gamma*a(t))"
                )
                material_derivative["temporal_branch"] = temporal_branch
                material_derivative["reference_noise"] = reference_noise
                material_derivative["coordinate_system"] = "path_marginal_z"
                material_derivative["prior_boundary_taper"] = "min(1,t/H)"
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "weak_calibration_reference_geomean",
                "weak_calibration_horizon_geomean",
            }:
                foresight_formula = (
                    "beta=1+gamma; C=beta*(S-W); "
                    "a*=argmin_{a>=0}||a*G-C||^2; "
                    "Qbar=sum_j pi_j*W(q_j); "
                    "G_new=W+beta*(S-Qbar); sum_j pi_j=1"
                )
                material_derivative["density_ratio_target"] = (
                    "p_W * (p_S / prod_j p_Qj^pi_j)^beta"
                )
                material_derivative["reference_ensemble"] = (
                    "equal time-only/projected references"
                    if component == "weak_calibration_reference_geomean"
                    else "equal half/full-horizon projected references"
                )
                material_derivative["affine_coefficient_sum"] = 1.0
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "velocity_parameterization_transport",
                "velocity_score_evolution",
                "velocity_change_recomposed",
            }:
                selected = component.removeprefix("velocity_")
                foresight_formula = (
                    "t_ref=t+H; s_t=(t*W_t-z)/(1-t); "
                    "W_hold=V_(t_ref)(s_t); "
                    "P=W_t-W_hold; E=W_hold-W_ref; "
                    f"G_new=G+(1+gamma)*{selected}"
                )
                material_derivative["exact_decomposition"] = (
                    "W_t-W_ref=parameterization_transport+score_evolution"
                )
                material_derivative["selected_component"] = selected
                material_derivative["reference_noise"] = "cleaner"
                material_derivative["coordinate_system"] = "path_marginal_z"
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_calibration_innovation":
                foresight_formula = (
                    "beta=1+gamma; C=beta*(S-W); "
                    "a*=argmin_{a>=0}||a*G-C||^2; q=(z+h*a*G,t+h); "
                    "R=beta*(W-W(q)); G_new=G+Proj_perp_(S-W)(R)"
                )
                material_derivative["query_geometry"] = (
                    "closest forward-characteristic encoding of calibration"
                )
                material_derivative["revision_geometry"] = (
                    "orthogonal innovation relative to the already calibrated "
                    "static strong-minus-weak axis"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "weak_calibration_depth8_response",
                "weak_calibration_telescoping_depth8",
            }:
                if component == "weak_calibration_depth8_response":
                    foresight_formula = (
                        "G_IG=W4+beta*(S-W4); q=(z+h*Proj_ray(beta*(S-W4)),t+h); "
                        "G_new=G_IG+beta*(W8-W8(q))"
                    )
                    levels = "depth8 response only"
                else:
                    foresight_formula = (
                        "q=(z+h*Proj_ray(beta*(S-W4)),t+h); "
                        "G_new=W4+beta*[(W8-W4(q))+(S-W8(q))]"
                    )
                    levels = "depth4 -> depth8 -> full"
                material_derivative["hierarchy_levels"] = levels
                material_derivative["ordinary_ig_anchor"] = "exact when q=p"
                material_derivative["affine_coefficient_sum"] = 1.0
                material_derivative["extra_tuned_coefficients"] = 0
            elif component in {
                "weak_calibration_innovation_strong_axis",
                "weak_calibration_innovation_guided_axis",
            }:
                axis = (
                    "S"
                    if component == "weak_calibration_innovation_strong_axis"
                    else "G"
                )
                foresight_formula = (
                    "beta=1+gamma; C=beta*(S-W); "
                    "a*=argmin_{a>=0}||a*G-C||^2; q=(z+h*a*G,t+h); "
                    f"R=beta*(W-W(q)); G_new=G+Proj_perp_({axis})(R)"
                )
                material_derivative["query_geometry"] = (
                    "closest forward-characteristic encoding of calibration"
                )
                material_derivative["revision_geometry"] = (
                    f"orthogonal control using {axis} rather than the "
                    "strong-minus-weak calibration axis"
                )
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_guidance_innovation":
                foresight_formula = (
                    "C=gamma*(S-W); a*=argmin_{a>=0}||a*G-C||^2; "
                    "q=(z+h*a*G,t+h); R=gamma*(W-W(q)); "
                    "G_new=G+Proj_perp_(S-W)(R)"
                )
                material_derivative["anchor"] = "strong"
                material_derivative["zero_guidance_limit"] = "exactly S"
                material_derivative["extra_tuned_coefficients"] = 0
            elif component == "weak_drift_midpoint":
                foresight_formula = (
                    "G_mid=G(z+.5*h*G,t+.5*h); "
                    "z_plus=z+h*G_mid; G_new=G+eta*(W(z,t)-W(z_plus,t+h))"
                )
                material_derivative["characteristic_integrator"] = (
                    "explicit midpoint RK2"
                )
            elif component in {"weak_lie_bracket", "weak_lie_bracket_matched"}:
                foresight_formula = (
                    "C=(W-W(z+h*G,t+h))-(G-G(z+h*W,t+h)); "
                    "G_new=G+eta*C"
                )
                material_derivative["small_h_limit"] = (
                    "C=-h*[G,W]+O(h^2), using time-augmented fields"
                )
                material_derivative["matched_reference"] = (
                    "per-sample RMS of W-W_future"
                    if component.endswith("_matched")
                    else None
                )
        else:
            foresight_formula = LEGACY_FORESIGHT_RESIDUAL_FORMULA
            material_derivative = None
        return {
            "format": CONDITION_FORMAT,
            "name": self.name,
            "kind": self.kind,
            "rho_first": float(self.rho_first),
            "rho_second": float(self.rho_second),
            "probe_multiplier": float(self.probe_multiplier),
            "local_multiplier": float(self.local_multiplier),
            "roundtrip_iterations": int(self.roundtrip_iterations),
            "roundtrip_multiplier": float(self.roundtrip_multiplier),
            "residual_strength": float(self.residual_strength),
            "partition_count": int(self.partition_count),
            "fsg_base_fraction": float(self.fsg_base_fraction),
            "strong_field": "S",
            "weak_field": "W4",
            "production_ig": {
                "first_interval": [0.0, FIRST_END],
                "first_gamma": BASE_GAMMA_FIRST,
                "second_interval": [FIRST_END, GUIDANCE_END],
                "second_gamma": BASE_GAMMA_SECOND,
                "late_interval": [GUIDANCE_END, 1.0],
                "late_gamma": 0.0,
            },
            "path_formula": "Phi_S + rho*(Phi_IG-Phi_S)",
            "roundtrip_formula": "relax(Phi_W4^{-1}(Phi_S(z))-z), then transport with S",
            "foresight_residual_formula": foresight_formula,
            "foresight_material_derivative": material_derivative,
            "fsg_formula": "iterate Phi_W4^{-1} o Phi_IG, then base transport",
        }

    def legacy_payload(self) -> dict[str, Any]:
        payload = self.payload()
        payload["format"] = LEGACY_CONDITION_FORMAT
        payload["foresight_residual_formula"] = LEGACY_FORESIGHT_RESIDUAL_FORMULA
        payload.pop("foresight_material_derivative")
        return payload


def condition_from_payload(payload: dict[str, Any]) -> Condition:
    payload_format = payload.get("format")
    if payload_format not in {CONDITION_FORMAT, LEGACY_CONDITION_FORMAT}:
        raise ValueError("invalid path-IG condition format")
    condition = Condition(
        kind=str(payload["kind"]),
        rho_first=float(payload["rho_first"]),
        rho_second=float(payload["rho_second"]),
        probe_multiplier=float(payload["probe_multiplier"]),
        local_multiplier=float(payload["local_multiplier"]),
        roundtrip_iterations=int(payload["roundtrip_iterations"]),
        roundtrip_multiplier=float(payload["roundtrip_multiplier"]),
        residual_strength=float(payload["residual_strength"]),
        partition_count=int(payload["partition_count"]),
        fsg_base_fraction=float(payload["fsg_base_fraction"]),
    )
    expected = (
        condition.payload()
        if payload_format == CONDITION_FORMAT
        else condition.legacy_payload()
    )
    if expected != payload:
        raise ValueError("non-canonical path-IG condition payload")
    return condition


def conditions_from_args(args: argparse.Namespace) -> tuple[Condition, ...]:
    conditions: list[Condition] = []
    if args.include_global_anchor:
        conditions.append(Condition("global"))
    if getattr(args, "include_semigroup_local_jensen", False):
        conditions.append(Condition("semigroup_local_jensen"))
    for mode in args.path_modes:
        kind = f"path_{mode}"
        for rho in args.path_rhos:
            conditions.append(Condition(kind, rho_first=rho, rho_second=rho))
    for first, second in args.path_rho_pairs:
        conditions.append(
            Condition("path_segmented", rho_first=first, rho_second=second)
        )
    for mode in args.path_modes:
        kind = f"path_{mode}"
        for alpha in args.matched_probe_alphas:
            rho = 1.0 / float(alpha)
            conditions.append(
                Condition(
                    kind,
                    rho_first=rho,
                    rho_second=rho,
                    probe_multiplier=alpha,
                )
            )
    for mode in args.local_modes:
        kind = "local_scaled_whole" if mode == "whole" else "local_scaled_segmented"
        for multiplier in args.local_multipliers:
            conditions.append(Condition(kind, local_multiplier=multiplier))
    for mode in args.roundtrip_modes:
        kind = f"roundtrip_{mode}"
        for iterations in args.roundtrip_iterations:
            for multiplier in args.roundtrip_multipliers:
                conditions.append(
                    Condition(
                        kind,
                        roundtrip_iterations=iterations,
                        roundtrip_multiplier=multiplier,
                    )
                )
    for mode in getattr(args, "roundtrip_residual_modes", ()):
        for strength in getattr(args, "roundtrip_residual_strengths", ()):
            conditions.append(
                Condition(
                    f"roundtrip_residual_{mode}", residual_strength=strength
                )
            )
    for partitions in getattr(args, "partition_counts", ()):
        for multiplier in getattr(args, "partitioned_roundtrip_multipliers", ()):
            conditions.append(
                Condition(
                    "roundtrip_partitioned",
                    roundtrip_multiplier=multiplier,
                    partition_count=partitions,
                )
            )
        for strength in getattr(args, "future_gap_strengths", ()):
            for component in getattr(args, "future_gap_components", ()):
                conditions.append(
                    Condition(
                        f"future_gap_{component}_partitioned",
                        residual_strength=strength,
                        partition_count=partitions,
                    )
                )
    for base in getattr(args, "fsg_bases", ()):
        for relaxation in getattr(args, "fsg_relaxations", ()):
            conditions.append(
                Condition(f"fsg_{base}_base", residual_strength=relaxation)
            )
    for base_fraction, relaxation in getattr(args, "fsg_hybrid_pairs", ()):
        conditions.append(
            Condition(
                "fsg_hybrid_base",
                residual_strength=relaxation,
                fsg_base_fraction=base_fraction,
            )
        )
    for schedule in getattr(args, "fsg_budgeted_schedules", ()):
        for base_fraction, first, second in getattr(args, "fsg_budgeted_specs", ()):
            conditions.append(
                Condition(
                    f"fsg_budgeted_{schedule}",
                    rho_first=first,
                    rho_second=second,
                    fsg_base_fraction=base_fraction,
                )
            )
    for schedule in getattr(args, "fsg_early_schedules", ()):
        for relaxation in getattr(args, "fsg_early_relaxations", ()):
            conditions.append(
                Condition(
                    f"fsg_early_{schedule}",
                    residual_strength=relaxation,
                )
            )
    for horizon in getattr(args, "fsg_residual_horizons", ()):
        for strength in getattr(args, "fsg_residual_strengths", ()):
            conditions.append(
                Condition(
                    "fsg_residual_continuous",
                    rho_first=horizon,
                    residual_strength=strength,
                )
            )
    for horizon in getattr(args, "fmd_decomposition_horizons", ()):
        for strength in getattr(args, "fmd_decomposition_strengths", ()):
            for component in getattr(args, "fmd_decomposition_components", ()):
                if component in {
                    "weak_drift_anisotropic",
                    "weak_curvature_mix",
                    "weak_material_anisotropic",
                    "weak_drift_segmented",
                    "weak_drift_path_mix",
                    "weak_drift_oblique",
                    "weak_drift_oblique_curvature",
                    "weak_drift_oblique_picard",
                    "weak_clean_endpoint_drift",
                    "weak_clean_endpoint_drift_matched",
                    "weak_noise_endpoint_drift",
                    "weak_noise_endpoint_drift_matched",
                    "weak_endpoint_contrast",
                    "weak_endpoint_contrast_segmented",
                    "weak_reference_forecast",
                    "weak_drift_extended",
                }:
                    continue
                gamma_specs = [
                    (multiplier, multiplier)
                    for multiplier in getattr(args, "fmd_gamma_multipliers", (1.0,))
                ]
                gamma_specs.extend(
                    getattr(args, "fmd_gamma_segment_pairs", ())
                )
                for gamma_first_multiplier, gamma_second_multiplier in dict.fromkeys(
                    gamma_specs
                ):
                    zero_gamma_control = (
                        gamma_first_multiplier == gamma_second_multiplier == 0.0
                        and component.startswith("weak_calibration_")
                    )
                    if gamma_second_multiplier <= 0.0 and not zero_gamma_control:
                        raise ValueError(
                            "the second FMD gamma multiplier must be positive"
                        )
                    global_gamma_multiplier = gamma_second_multiplier
                    first_gamma_ratio = (
                        1.0
                        if zero_gamma_control
                        else gamma_first_multiplier / gamma_second_multiplier
                    )
                    iterations = (
                        getattr(args, "fmd_picard_iterations", (1,))
                        if component == "picard_weak"
                        else (1,)
                    )
                    if component == "picard_weak_poly":
                        for second_strength in getattr(
                            args, "fmd_picard_second_strengths", ()
                        ):
                            conditions.append(
                                Condition(
                                    "fmd_picard_weak_poly_continuous",
                                    rho_first=horizon,
                                    residual_strength=strength,
                                    roundtrip_iterations=2,
                                    roundtrip_multiplier=second_strength,
                                    local_multiplier=global_gamma_multiplier,
                                    probe_multiplier=first_gamma_ratio,
                                )
                            )
                        continue
                    for iteration_count in iterations:
                        conditions.append(
                            Condition(
                                f"fmd_{component}_continuous",
                                rho_first=horizon,
                                residual_strength=strength,
                                roundtrip_iterations=iteration_count,
                                local_multiplier=global_gamma_multiplier,
                                probe_multiplier=first_gamma_ratio,
                            )
                        )
    special_components = set(
        getattr(args, "fmd_decomposition_components", ())
    )
    for horizon in getattr(args, "fmd_decomposition_horizons", ()):
        if "weak_drift_anisotropic" in special_components:
            for orthogonal_strength, parallel_strength in getattr(
                args, "fmd_anisotropic_pairs", ()
            ):
                conditions.append(
                    Condition(
                        "fmd_weak_drift_anisotropic_continuous",
                        rho_first=horizon,
                        residual_strength=orthogonal_strength,
                        roundtrip_multiplier=parallel_strength,
                    )
                )
        if "weak_curvature_mix" in special_components:
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for curvature_weight in getattr(
                    args, "fmd_curvature_weights", ()
                ):
                    conditions.append(
                        Condition(
                            "fmd_weak_curvature_mix_continuous",
                            rho_first=horizon,
                            residual_strength=strength,
                            roundtrip_multiplier=curvature_weight,
                        )
                    )
        if "weak_material_anisotropic" in special_components:
            for temporal_strength, advective_strength in getattr(
                args, "fmd_material_pairs", ()
            ):
                conditions.append(
                    Condition(
                        "fmd_weak_material_anisotropic_continuous",
                        rho_first=horizon,
                        residual_strength=temporal_strength,
                        roundtrip_multiplier=advective_strength,
                    )
                )
        if "weak_drift_segmented" in special_components:
            for first_strength, second_strength in getattr(
                args, "fmd_strength_segment_pairs", ()
            ):
                conditions.append(
                    Condition(
                        "fmd_weak_drift_segmented_continuous",
                        rho_first=horizon,
                        residual_strength=first_strength,
                        roundtrip_multiplier=second_strength,
                    )
                )
        if "weak_drift_path_mix" in special_components:
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for path_rho in getattr(args, "fmd_characteristic_rhos", ()):
                    conditions.append(
                        Condition(
                            "fmd_weak_drift_path_mix_continuous",
                            rho_first=horizon,
                            residual_strength=strength,
                            roundtrip_multiplier=path_rho,
                        )
                    )
        if "weak_drift_oblique" in special_components:
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    conditions.append(
                        Condition(
                            "fmd_weak_drift_oblique_continuous",
                            rho_first=horizon,
                            residual_strength=strength,
                            roundtrip_multiplier=spatial_fraction,
                        )
                    )
        if "weak_drift_oblique_curvature" in special_components:
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    for curvature_weight in getattr(
                        args, "fmd_curvature_weights", ()
                    ):
                        conditions.append(
                            Condition(
                                "fmd_weak_drift_oblique_curvature_continuous",
                                rho_first=horizon,
                                rho_second=curvature_weight,
                                residual_strength=strength,
                                roundtrip_multiplier=spatial_fraction,
                            )
                        )
        if "weak_drift_oblique_picard" in special_components:
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    for iteration_count in getattr(
                        args, "fmd_picard_iterations", (1,)
                    ):
                        conditions.append(
                            Condition(
                                "fmd_weak_drift_oblique_picard_continuous",
                                rho_first=horizon,
                                residual_strength=strength,
                                roundtrip_multiplier=spatial_fraction,
                                roundtrip_iterations=iteration_count,
                            )
                        )
        for endpoint_component in (
            "weak_clean_endpoint_drift",
            "weak_clean_endpoint_drift_matched",
            "weak_noise_endpoint_drift",
            "weak_noise_endpoint_drift_matched",
        ):
            if endpoint_component not in special_components:
                continue
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    conditions.append(
                        Condition(
                            f"fmd_{endpoint_component}_continuous",
                            rho_first=horizon,
                            residual_strength=strength,
                            roundtrip_multiplier=spatial_fraction,
                        )
                    )
        if "weak_endpoint_contrast" in special_components:
            for strength in getattr(args, "fmd_decomposition_strengths", ()):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    for noise_weight in getattr(
                        args, "fmd_endpoint_noise_weights", ()
                    ):
                        conditions.append(
                            Condition(
                                "fmd_weak_endpoint_contrast_continuous",
                                rho_first=horizon,
                                rho_second=noise_weight,
                                residual_strength=strength,
                                roundtrip_multiplier=spatial_fraction,
                            )
                        )
        if "weak_endpoint_contrast_segmented" in special_components:
            for first_strength, second_strength in getattr(
                args, "fmd_strength_segment_pairs", ()
            ):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    gamma_specs = [
                        (multiplier, multiplier)
                        for multiplier in getattr(
                            args, "fmd_gamma_multipliers", (1.0,)
                        )
                    ]
                    gamma_specs.extend(
                        getattr(args, "fmd_gamma_segment_pairs", ())
                    )
                    for gamma_first, gamma_second in dict.fromkeys(gamma_specs):
                        if gamma_second <= 0.0:
                            raise ValueError(
                                "the second FMD gamma multiplier must be positive"
                            )
                        conditions.append(
                            Condition(
                                "fmd_weak_endpoint_contrast_segmented_continuous",
                                rho_first=horizon,
                                rho_second=second_strength,
                                residual_strength=first_strength,
                                roundtrip_multiplier=spatial_fraction,
                                local_multiplier=gamma_second,
                                probe_multiplier=gamma_first / gamma_second,
                            )
                        )
        if "weak_reference_forecast" in special_components:
            for forecast_factor in getattr(args, "fmd_forecast_factors", ()):
                for spatial_fraction in getattr(args, "fmd_oblique_alphas", ()):
                    conditions.append(
                        Condition(
                            "fmd_weak_reference_forecast_continuous",
                            rho_first=horizon,
                            residual_strength=forecast_factor,
                            roundtrip_multiplier=spatial_fraction,
                        )
                    )
        if "weak_drift_extended" in special_components:
            for early_strength, late_strength, correction_end in getattr(
                args, "fmd_extended_specs", ()
            ):
                conditions.append(
                    Condition(
                        "fmd_weak_drift_extended_continuous",
                        rho_first=horizon,
                        rho_second=correction_end,
                        residual_strength=early_strength,
                        roundtrip_multiplier=late_strength,
                    )
                )
    if "weak_drift_oblique" in special_components:
        for horizon, strength, spatial_fraction in getattr(
            args, "fmd_oblique_specs", ()
        ):
            conditions.append(
                Condition(
                    "fmd_weak_drift_oblique_continuous",
                    rho_first=horizon,
                    residual_strength=strength,
                    roundtrip_multiplier=spatial_fraction,
                )
            )
    unique = {condition.name: condition for condition in conditions}
    pattern = re.compile(args.condition_regex)
    selected = tuple(
        condition for name, condition in unique.items() if pattern.search(name)
    )
    if not selected:
        raise ValueError("condition regex selected no conditions")
    return selected


def _gamma_at(time_value: float, *, multiplier: float = 1.0) -> float:
    if time_value < FIRST_END:
        return BASE_GAMMA_FIRST * multiplier
    if time_value < GUIDANCE_END:
        return BASE_GAMMA_SECOND * multiplier
    return 0.0


def _fsg_events(condition: Condition) -> tuple[tuple[float, float, int, float, str], ...]:
    if condition.kind == "fsg_early_official":
        return (
            (0.0, 0.125, 2, condition.residual_strength, "event0"),
            (0.125, 0.25, 2, condition.residual_strength, "event1"),
        )
    if condition.kind == "fsg_early_balanced":
        return (
            (0.0, 0.125, 1, condition.residual_strength, "event0"),
            (0.125, 0.25, 1, condition.residual_strength, "event1"),
        )
    if condition.kind == "fsg_budgeted_official":
        return (
            (0.0, 0.125, 2, condition.rho_first, "event0"),
            (0.125, 0.25, 2, condition.rho_first, "event1"),
            (0.375, 0.5, 1, condition.rho_second, "event2"),
        )
    if condition.kind == "fsg_budgeted_balanced":
        return (
            (0.0, 0.125, 1, condition.rho_first, "event0"),
            (0.125, 0.25, 1, condition.rho_first, "event1"),
            (0.25, 0.375, 1, condition.rho_second, "event2"),
            (0.375, 0.5, 1, condition.rho_second, "event3"),
        )
    return (
        (0.0, 0.125, 2, condition.residual_strength, "event0"),
        (0.125, 0.25, 2, condition.residual_strength, "event1"),
        (0.375, 0.5, 1, condition.residual_strength, "event2"),
    )


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
    }


def _integrate(odeint, field, state, start: float, end: float, args, device):
    return odeint(
        field,
        state,
        torch.tensor([start, end], device=device),
        method="dopri5",
        atol=args.atol,
        rtol=args.rtol,
    )[-1]


def reusable(path: Path, condition: Condition, args: argparse.Namespace) -> bool:
    if not path.is_file():
        return False
    try:
        result = read_json(path)
        manifest = result["sampling_manifest"]
        sampling = manifest["sampling"]
        metrics = result.get("metrics")
        return (
            condition_from_payload(result["condition"]) == condition
            and str(sampling["integrator"]) == "dopri5"
            and int(sampling["num_samples"]) == args.num_samples
            and int(sampling["batch_size"]) == args.batch_size
            and int(sampling["seed"]) == args.seed
            and float(sampling["atol"]) == float(args.atol)
            and float(sampling["rtol"]) == float(args.rtol)
            and bool(manifest["noise_sha256"])
            and bool(manifest["label_sha256"])
            and (
                not args.keep_samples
                or (
                    bool(result.get("sample_retained"))
                    and Path(manifest["samples"]).is_file()
                )
            )
            and (
                args.skip_fid
                or (
                    isinstance(metrics, dict)
                    and all(
                        isinstance(metrics.get(key), (int, float))
                        and math.isfinite(float(metrics[key]))
                        for key in ("fid", "sfid", "inception_score")
                    )
                )
            )
        )
    except Exception:
        return False


def worker(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    from diffusers.models import AutoencoderKL
    from torchdiffeq import odeint
    from torchvision.utils import save_image

    from experiments.imagenet100_sit_multiscale_models import (
        evaluate_internal_head_only,
    )

    repo = Path(args.repo).resolve()
    data = Path(args.data).resolve()
    paths = runtime_paths(repo, data, Path(args.adm_python))
    condition = condition_from_payload(read_json(Path(args.condition_json)))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "condition_result.json"
    if reusable(result_path, condition, args):
        print(json.dumps({"event": "reuse", "condition": condition.name}), flush=True)
        return

    modules = load_repo_modules(repo)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    allocator = modules["configure_cuda_allocator"](
        device, limit_gib=args.cuda_allocator_limit_gib
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    sit_module, source_metadata = modules["load_official_sit_module"](
        Path(modules["DEFAULT_OFFICIAL_SIT_REPO"]).expanduser().resolve(),
        verify_source=True,
    )
    strong, semantics, strong_metadata = modules["load_sit_field_model"](
        checkpoint_path=paths["strong"],
        weights="ema",
        sit_module=sit_module,
        source_metadata=source_metadata,
        device=device,
    )
    if semantics.prediction_target != "velocity":
        raise ValueError("path-IG requires the native-velocity v800 source")
    head = modules["load_internal_head_for_source"](
        checkpoint_path=paths["depth4"],
        name="depth4_v",
        head_weights="ema",
        model=strong,
        sit_module=sit_module,
        source_checkpoint_path=paths["strong"],
        source_metadata=source_metadata,
        device=device,
    )
    heads = {"depth4_v": head}
    depth8_head = None
    if "depth8" in condition.kind:
        depth8_path = (
            data
            / "runs/sit-s-2_v800-ema_frozen-internal-v-depth8_seed0/"
            "checkpoints/step_00050000.pt"
        )
        if not depth8_path.is_file():
            raise FileNotFoundError(depth8_path)
        depth8_head = modules["load_internal_head_for_source"](
            checkpoint_path=depth8_path,
            name="depth8_v",
            head_weights="ema",
            model=strong,
            sit_module=sit_module,
            source_checkpoint_path=paths["strong"],
            source_metadata=source_metadata,
            device=device,
        )
        heads["depth8_v"] = depth8_head
    vae = (
        AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", local_files_only=True
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )

    def evaluate_hierarchy(time: Any, latent: Any, labels: Any):
        times = time.expand(len(latent))
        full, trained, _ = modules["evaluate_source_with_heads"](
            strong, latent, times, labels, heads=heads
        )
        return full, trained["depth4_v"], trained.get("depth8_v")

    def evaluate_pair(time: Any, latent: Any, labels: Any):
        full, weak, _ = evaluate_hierarchy(time, latent, labels)
        return full, weak

    def evaluate_weak_only(time: Any, latent: Any, labels: Any):
        times = time.expand(len(latent))
        return evaluate_internal_head_only(
            strong,
            latent,
            times,
            labels,
            spec=head,
        )

    def evaluate_counterfactual_hierarchy(time: Any, latent: Any, labels: Any):
        if depth8_head is None:
            raise RuntimeError("depth-8 counterfactual hierarchy was not loaded")
        _, weak4, weak8 = evaluate_hierarchy(time, latent, labels)
        assert weak8 is not None
        return weak4, weak8

    class StrongField:
        def __init__(self, labels: Any):
            self.labels = labels
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            full, _ = evaluate_pair(time, latent, self.labels)
            return full

    class GuidedField:
        def __init__(self, labels: Any, *, multiplier: float):
            self.labels = labels
            self.multiplier = float(multiplier)
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            full, weak = evaluate_pair(time, latent, self.labels)
            gamma = _gamma_at(
                float(time.detach().float().item()), multiplier=self.multiplier
            )
            if gamma == 0.0:
                return full
            return full + gamma * (full - weak)

    class SemigroupLocalJensenField:
        """Zero-extra-scale local correction implied by endpoint power guidance."""

        def __init__(self, labels: Any):
            self.labels = labels
            self.nfe = 0
            self.diagnostics: dict[str, list[float]] = {}

        def _record_rms(self, name: str, value: Any) -> None:
            self.diagnostics.setdefault(name, []).extend(
                sample_rms(value).detach().cpu().tolist()
            )

        def _record_cosine(self, name: str, left: Any, right: Any) -> None:
            cosine = torch.nn.functional.cosine_similarity(
                left.float().flatten(1),
                right.float().flatten(1),
                dim=1,
                eps=1e-12,
            )
            self.diagnostics.setdefault(name, []).extend(
                cosine.detach().cpu().tolist()
            )

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            time_value = float(time.detach().float().item())
            gamma = _gamma_at(time_value)
            if gamma == 0.0:
                full, _ = evaluate_pair(time, latent, self.labels)
                return full

            # The outer sampler runs in inference mode.  Recreate only the
            # current state as a normal autograd tensor for one input VJP.
            with torch.inference_mode(False), torch.enable_grad():
                differentiable_state = latent.detach().clone().requires_grad_(True)
                differentiable_time = time.detach().clone()
                full, weak = evaluate_pair(
                    differentiable_time,
                    differentiable_state,
                    self.labels,
                )
                gap = full - weak
                guided = full + gamma * gap
                correction = local_jensen_velocity_correction(
                    gap,
                    state=differentiable_state,
                    time_value=differentiable_time,
                    beta=1.0 + gamma,
                )
                result = guided + correction
            self._record_rms("semigroup_gap_rms", gap)
            self._record_rms("semigroup_correction_rms", correction)
            self._record_rms("semigroup_guided_rms", guided)
            self._record_cosine("semigroup_correction_gap_cosine", correction, gap)
            self._record_cosine(
                "semigroup_correction_guided_cosine", correction, guided
            )
            return result.detach()

    class ContinuousForesightResidualField:
        def __init__(self, labels: Any, *, horizon: float, strength: float):
            self.labels = labels
            self.horizon = float(horizon)
            self.strength = float(strength)
            self.nfe = 0
            self.future_nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            full, weak = evaluate_pair(time, latent, self.labels)
            time_value = float(time.detach().float().item())
            gamma_multiplier = condition.local_multiplier
            if time_value < FIRST_END:
                gamma_multiplier *= condition.probe_multiplier
            gamma = _gamma_at(time_value, multiplier=gamma_multiplier)
            guided = full if gamma == 0.0 else full + gamma * (full - weak)
            if self.strength == 0.0 or time_value >= GUIDANCE_END:
                return guided
            horizon = min(self.horizon, GUIDANCE_END - time_value)
            if horizon <= 0.0:
                return guided
            future_time = time + time.new_tensor(horizon)

            future_state = latent + horizon * guided
            weak_future = evaluate_weak_only(
                future_time,
                future_state,
                self.labels,
            )
            self.future_nfe += 1
            decomposition = decompose_euler_foresight_roundtrip(
                guided,
                weak,
                weak_future,
                horizon=horizon,
            )
            future_velocity = decomposition.future_displacement / horizon
            return guided + self.strength * future_velocity

    class ContinuousForesightDecompositionField:
        def __init__(
            self,
            labels: Any,
            *,
            horizon: float,
            strength: float,
            component: str,
        ):
            self.labels = labels
            self.horizon = float(horizon)
            self.strength = float(strength)
            self.component = component
            self.nfe = 0
            self.future_nfe = 0
            self.diagnostics: dict[str, list[float]] = {}

        def _record_rms(self, name: str, value: Any) -> None:
            self.diagnostics.setdefault(name, []).extend(
                sample_rms(value).detach().cpu().tolist()
            )

        def _record_cosine(self, name: str, left: Any, right: Any) -> None:
            left_flat = left.float().flatten(1)
            right_flat = right.float().flatten(1)
            cosine = torch.nn.functional.cosine_similarity(
                left_flat, right_flat, dim=1, eps=1e-12
            )
            self.diagnostics.setdefault(name, []).extend(
                cosine.detach().cpu().tolist()
            )

        def _record_projection(self, name: str, value: Any, reference: Any) -> None:
            value_flat = value.float().flatten(1)
            reference_flat = reference.float().flatten(1)
            coefficient = (
                (value_flat * reference_flat).sum(dim=1)
                / reference_flat.square().sum(dim=1).clamp_min(1e-12)
            )
            self.diagnostics.setdefault(name, []).extend(
                coefficient.detach().cpu().tolist()
            )

        @staticmethod
        def _scheduled_gamma(time_value: float) -> float:
            multiplier = condition.local_multiplier
            if time_value < FIRST_END:
                multiplier *= condition.probe_multiplier
            return _gamma_at(time_value, multiplier=multiplier)

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            middle = None
            if self.component in {
                "weak_calibration_depth8_response",
                "weak_calibration_telescoping_depth8",
            }:
                full, weak, middle = evaluate_hierarchy(time, latent, self.labels)
                if middle is None:
                    raise RuntimeError("depth-8 hierarchy output is missing")
            else:
                full, weak = evaluate_pair(time, latent, self.labels)
            time_value = float(time.detach().float().item())
            gamma_multiplier = condition.local_multiplier
            if time_value < FIRST_END:
                gamma_multiplier *= condition.probe_multiplier
            gamma = _gamma_at(time_value, multiplier=gamma_multiplier)
            guided = full if gamma == 0.0 else full + gamma * (full - weak)
            correction_end = (
                condition.rho_second
                if self.component == "weak_drift_extended"
                else GUIDANCE_END
            )
            active_strength = (
                condition.roundtrip_multiplier
                if self.component == "weak_drift_extended"
                and time_value >= GUIDANCE_END
                else self.strength
            )
            if self.component == "weak_endpoint_contrast_segmented":
                active_strength = (
                    self.strength
                    if time_value < FIRST_END
                    else condition.rho_second
                )
            if (
                self.component not in {
                    "weak_calibration_split",
                    "weak_calibration_time_only",
                    "weak_calibration_characteristic",
                    "weak_calibration_weak_characteristic",
                    "weak_calibration_strong_characteristic",
                    "weak_calibration_projected",
                    "weak_calibration_projected_coupled",
                    "weak_calibration_reference_geomean",
                    "weak_calibration_horizon_geomean",
                    "weak_calibration_depth8_response",
                    "weak_calibration_telescoping_depth8",
                    "weak_gap_transport_time_only",
                    "weak_gap_transport_projected",
                    "strong_gap_transport_projected",
                    "weak_gap_antitransport_projected",
                    "score_noisier_aligned",
                    "score_noisier_same_state",
                    "score_cleaner_aligned",
                    "velocity_noisier_aligned",
                    "marginal_score_weak_noisier",
                    "marginal_score_strong_noisier",
                    "marginal_score_weak_cleaner",
                    "velocity_parameterization_transport",
                    "velocity_score_evolution",
                    "velocity_change_recomposed",
                    "weak_calibration_innovation",
                    "weak_calibration_innovation_strong_axis",
                    "weak_calibration_innovation_guided_axis",
                    "weak_guidance_innovation",
                }
                and active_strength == 0.0
            ) or time_value >= correction_end:
                return guided
            segment_end = (
                GUIDANCE_END
                if self.component == "weak_drift_extended"
                and time_value < GUIDANCE_END
                else correction_end
            )
            horizon = min(self.horizon, segment_end - time_value)
            if horizon <= 0.0:
                return guided

            future_time = time + time.new_tensor(horizon)
            if self.component == "weak_calibration_split":
                weak_base, calibration = split_internal_guidance(
                    full,
                    weak,
                    gamma=gamma,
                )
                future_state = latent + horizon * calibration
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                result = calibration_split_foresight_velocity(
                    full,
                    weak_base,
                    weak_future,
                    gamma=gamma,
                )
                self._record_rms("calibration_split_rms", calibration)
                self._record_rms("calibration_split_guided_rms", guided)
                self._record_cosine(
                    "calibration_split_guided_cosine", calibration, guided
                )
                self._record_projection(
                    "calibration_split_guided_projection", calibration, guided
                )
                self._record_rms(
                    "calibration_split_revision_rms", result - guided
                )
                return result

            if self.component in {
                "weak_calibration_time_only",
                "weak_calibration_characteristic",
                "weak_calibration_weak_characteristic",
                "weak_calibration_strong_characteristic",
                "weak_calibration_projected",
                "weak_calibration_projected_coupled",
                "weak_calibration_reference_geomean",
                "weak_calibration_horizon_geomean",
                "weak_calibration_depth8_response",
                "weak_calibration_telescoping_depth8",
            }:
                weak_base, calibration = split_internal_guidance(
                    full,
                    weak,
                    gamma=gamma,
                )
                ray_projection = None
                if self.component == "weak_calibration_time_only":
                    future_state = latent
                elif self.component == "weak_calibration_characteristic":
                    future_state = latent + horizon * guided
                elif self.component == "weak_calibration_weak_characteristic":
                    future_state = latent + horizon * weak
                elif self.component == "weak_calibration_strong_characteristic":
                    future_state = latent + horizon * full
                else:
                    ray_projection = project_to_forward_ray(calibration, guided)
                    future_state = latent + horizon * ray_projection.parallel
                    if self.component == "weak_calibration_projected_coupled":
                        coefficient = ray_projection.coefficient.reshape(
                            len(latent), *([1] * (latent.ndim - 1))
                        ).to(dtype=time.dtype, device=time.device)
                        future_time = time + horizon * coefficient.flatten()
                if self.component == "weak_calibration_reference_geomean":
                    weak_time_only = evaluate_weak_only(
                        future_time,
                        latent,
                        self.labels,
                    )
                    weak_projected = evaluate_weak_only(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 2
                    weak_references = (weak_time_only, weak_projected)
                    self._record_cosine(
                        "counterfactual_reference_cosine",
                        weak_time_only,
                        weak_projected,
                    )
                elif self.component == "weak_calibration_horizon_geomean":
                    half_horizon = 0.5 * horizon
                    half_time = time + time.new_tensor(half_horizon)
                    half_state = latent + half_horizon * ray_projection.parallel
                    weak_half = evaluate_weak_only(
                        half_time,
                        half_state,
                        self.labels,
                    )
                    weak_full = evaluate_weak_only(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 2
                    weak_references = (weak_half, weak_full)
                    self._record_cosine(
                        "counterfactual_reference_cosine",
                        weak_half,
                        weak_full,
                    )
                elif self.component in {
                    "weak_calibration_depth8_response",
                    "weak_calibration_telescoping_depth8",
                }:
                    if middle is None:
                        raise RuntimeError("depth-8 current readout is missing")
                    weak_future, middle_future = evaluate_counterfactual_hierarchy(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 1
                    beta = 1.0 + gamma
                    if self.component == "weak_calibration_depth8_response":
                        result = guided + beta * (middle - middle_future)
                    else:
                        result = counterfactual_telescoping_velocity(
                            (weak_base, middle, full),
                            (weak_future, middle_future),
                            gamma=gamma,
                        )
                    self._record_rms(
                        "calibration_depth4_response_rms",
                        weak_base - weak_future,
                    )
                    self._record_rms(
                        "calibration_depth8_response_rms",
                        middle - middle_future,
                    )
                    self._record_cosine(
                        "calibration_depth_responses_cosine",
                        weak_base - weak_future,
                        middle - middle_future,
                    )
                    self._record_rms(
                        f"calibration_{self.component}_revision_rms",
                        result - guided,
                    )
                    weak_references = ()
                else:
                    weak_future = evaluate_weak_only(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 1
                    weak_references = (weak_future,)
                if not weak_references:
                    if ray_projection is not None:
                        self.diagnostics.setdefault(
                            f"calibration_{self.component}_alpha", []
                        ).extend(ray_projection.coefficient.detach().cpu().tolist())
                    return result
                result = affine_counterfactual_ratio_velocity(
                    full,
                    weak_base,
                    weak_references,
                    (0.5, 0.5) if len(weak_references) == 2 else (1.0,),
                    gamma=gamma,
                )
                if ray_projection is not None:
                    self.diagnostics.setdefault(
                        f"calibration_{self.component}_alpha", []
                    ).extend(ray_projection.coefficient.detach().cpu().tolist())
                    self._record_rms(
                        f"calibration_{self.component}_query_rms",
                        ray_projection.parallel,
                    )
                    self._record_rms(
                        f"calibration_{self.component}_residual_rms",
                        ray_projection.orthogonal,
                    )
                self._record_rms(
                    f"calibration_{self.component}_revision_rms", result - guided
                )
                return result

            if self.component in {
                "weak_gap_transport_time_only",
                "weak_gap_transport_projected",
                "strong_gap_transport_projected",
                "weak_gap_antitransport_projected",
            }:
                weak_base, calibration = split_internal_guidance(
                    full,
                    weak,
                    gamma=gamma,
                )
                ray_projection = None
                if self.component == "weak_gap_transport_time_only":
                    query_state = latent
                else:
                    ray_projection = project_to_forward_ray(calibration, guided)
                    query_state = latent + horizon * ray_projection.parallel
                full_query, weak_query = evaluate_pair(
                    future_time,
                    query_state,
                    self.labels,
                )
                self.future_nfe += 1
                anchor = (
                    "strong"
                    if self.component == "strong_gap_transport_projected"
                    else "weak"
                )
                interaction_sign = (
                    -1.0
                    if self.component == "weak_gap_antitransport_projected"
                    else 1.0
                )
                result = transported_internal_gap_velocity(
                    full,
                    weak_base,
                    full_query,
                    weak_query,
                    gamma=gamma,
                    anchor=anchor,
                    interaction_sign=interaction_sign,
                )
                gap_now = full - weak_base
                gap_query = full_query - weak_query
                interaction = gap_query - gap_now
                self._record_rms("gap_transport_gap_now_rms", gap_now)
                self._record_rms("gap_transport_gap_query_rms", gap_query)
                self._record_rms("gap_transport_interaction_rms", interaction)
                self._record_cosine(
                    "gap_transport_gap_cosine",
                    gap_now,
                    gap_query,
                )
                self._record_cosine(
                    "gap_transport_interaction_gap_cosine",
                    interaction,
                    gap_now,
                )
                if ray_projection is not None:
                    self.diagnostics.setdefault(
                        "gap_transport_query_alpha", []
                    ).extend(ray_projection.coefficient.detach().cpu().tolist())
                return result

            if self.component in {
                "score_noisier_aligned",
                "score_noisier_same_state",
                "score_cleaner_aligned",
                "velocity_noisier_aligned",
            }:
                # Endpoint-normalized score coordinates are singular at t=0.
                # Returning ordinary IG through the first complete reference
                # interval avoids manufacturing a clamped pseudo-density.
                reference_horizon = self.horizon
                if time_value <= reference_horizon:
                    return guided
                cleaner_reference = self.component == "score_cleaner_aligned"
                reference_time = (
                    time + time.new_tensor(reference_horizon)
                    if cleaner_reference
                    else time - time.new_tensor(reference_horizon)
                )
                same_state = self.component == "score_noisier_same_state"
                reference_state = (
                    latent
                    if same_state
                    else align_linear_path_state_to_endpoint_coordinate(
                        latent,
                        time,
                        reference_time,
                    )
                )
                weak_reference = evaluate_weak_only(
                    reference_time,
                    reference_state,
                    self.labels,
                )
                self.future_nfe += 1
                if self.component == "velocity_noisier_aligned":
                    result = full + gamma * (full - weak_reference)
                else:
                    result = telescoping_scale_space_guidance_velocity(
                        full,
                        weak_reference,
                        latent,
                        reference_state,
                        time,
                        reference_time,
                        gamma=gamma,
                    )
                self._record_rms(
                    "scale_space_reference_state_change_rms",
                    reference_state - latent,
                )
                self._record_rms(
                    "scale_space_raw_velocity_gap_rms",
                    full - weak_reference,
                )
                self._record_rms(
                    "scale_space_revision_rms",
                    result - guided,
                )
                self._record_cosine(
                    "scale_space_revision_ig_cosine",
                    result - guided,
                    guided,
                )
                return result

            if self.component in {
                "marginal_score_weak_noisier",
                "marginal_score_strong_noisier",
                "marginal_score_weak_cleaner",
            }:
                if time_value <= 1e-8:
                    return guided
                cleaner_reference = "cleaner" in self.component
                reference_value = (
                    time_value + self.horizon
                    if cleaner_reference
                    else max(0.0, time_value - self.horizon)
                )
                reference_time = time.new_tensor(reference_value)
                strong_temporal = "strong" in self.component
                if strong_temporal:
                    full_reference, _ = evaluate_pair(
                        reference_time,
                        latent,
                        self.labels,
                    )
                    temporal_now = full
                    temporal_reference = full_reference
                else:
                    temporal_now = weak
                    temporal_reference = evaluate_weak_only(
                        reference_time,
                        latent,
                        self.labels,
                    )
                self.future_nfe += 1
                boundary_weight = min(1.0, time_value / self.horizon)
                result = factorized_scale_space_guidance_velocity(
                    full,
                    weak,
                    temporal_now,
                    temporal_reference,
                    latent,
                    latent,
                    time,
                    reference_time,
                    gamma=gamma,
                    temporal_weight=boundary_weight,
                )
                self._record_rms(
                    "marginal_scale_space_raw_velocity_gap_rms",
                    temporal_now - temporal_reference,
                )
                self._record_rms(
                    "marginal_scale_space_revision_rms",
                    result - guided,
                )
                self._record_cosine(
                    "marginal_scale_space_revision_ig_cosine",
                    result - guided,
                    guided,
                )
                self.diagnostics.setdefault(
                    "marginal_scale_space_boundary_weight", []
                ).append(float(boundary_weight))
                return result

            if self.component in {
                "velocity_parameterization_transport",
                "velocity_score_evolution",
                "velocity_change_recomposed",
            }:
                weak_reference = evaluate_weak_only(
                    future_time,
                    latent,
                    self.labels,
                )
                self.future_nfe += 1
                parts = decompose_cross_time_velocity_change(
                    weak,
                    weak_reference,
                    latent,
                    time,
                    future_time,
                )
                if self.component == "velocity_parameterization_transport":
                    selected = parts.parameterization_transport
                elif self.component == "velocity_score_evolution":
                    selected = parts.score_evolution
                else:
                    selected = parts.total
                result = guided + (1.0 + gamma) * selected
                self._record_rms(
                    "velocity_time_parameterization_transport_rms",
                    parts.parameterization_transport,
                )
                self._record_rms(
                    "velocity_time_score_evolution_rms",
                    parts.score_evolution,
                )
                self._record_cosine(
                    "velocity_time_component_cosine",
                    parts.parameterization_transport,
                    parts.score_evolution,
                )
                self._record_rms(
                    "velocity_time_selected_revision_rms",
                    result - guided,
                )
                return result

            if self.component in {
                "weak_calibration_innovation",
                "weak_calibration_innovation_strong_axis",
                "weak_calibration_innovation_guided_axis",
            }:
                weak_base, calibration = split_internal_guidance(
                    full,
                    weak,
                    gamma=gamma,
                )
                ray_projection = project_to_forward_ray(calibration, guided)
                future_state = latent + horizon * ray_projection.parallel
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                beta = 1.0 + gamma
                revision = beta * (weak_base - weak_future)
                projection_axis = full - weak_base
                if self.component == "weak_calibration_innovation_strong_axis":
                    projection_axis = full
                elif self.component == "weak_calibration_innovation_guided_axis":
                    projection_axis = guided
                innovation = project_per_sample(revision, projection_axis).orthogonal
                self.diagnostics.setdefault(
                    "calibration_innovation_alpha", []
                ).extend(ray_projection.coefficient.detach().cpu().tolist())
                self._record_rms("calibration_revision_rms", revision)
                self._record_rms("calibration_innovation_rms", innovation)
                self._record_cosine(
                    "calibration_revision_gap_cosine",
                    revision,
                    full - weak_base,
                )
                return guided + innovation

            if self.component == "weak_guidance_innovation":
                gap = full - weak
                calibration = gamma * gap
                ray_projection = project_to_forward_ray(calibration, guided)
                future_state = latent + horizon * ray_projection.parallel
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                revision = gamma * (weak - weak_future)
                innovation = project_per_sample(revision, gap).orthogonal
                self.diagnostics.setdefault(
                    "guidance_innovation_alpha", []
                ).extend(ray_projection.coefficient.detach().cpu().tolist())
                self._record_rms("guidance_revision_rms", revision)
                self._record_rms("guidance_innovation_rms", innovation)
                return guided + innovation

            if self.component == "weak_drift_path_mix":
                path_velocity = mix_characteristic_velocity(
                    weak,
                    guided,
                    rho=condition.roundtrip_multiplier,
                )
                future_state = latent + horizon * path_velocity
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                weak_drift = horizon * (weak - weak_future) / horizon
                self._record_rms("path_mix_weak_drift_rms", weak_drift)
                return guided + self.strength * weak_drift

            if self.component == "weak_drift_oblique":
                spatial_fraction = condition.roundtrip_multiplier
                if spatial_fraction == 0.0:
                    future_state = latent
                elif spatial_fraction == 1.0:
                    future_state = latent + horizon * guided
                else:
                    future_state = latent + spatial_fraction * horizon * guided
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                weak_drift = horizon * (weak - weak_future) / horizon
                self._record_rms("oblique_weak_drift_rms", weak_drift)
                return guided + self.strength * weak_drift

            if self.component in {
                "weak_clean_endpoint_drift",
                "weak_clean_endpoint_drift_matched",
                "weak_noise_endpoint_drift",
                "weak_noise_endpoint_drift_matched",
                "weak_endpoint_contrast",
                "weak_endpoint_contrast_segmented",
            }:
                spatial_fraction = condition.roundtrip_multiplier
                future_state = latent + spatial_fraction * horizon * guided
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                parts = decompose_endpoint_posterior_change(
                    latent,
                    time,
                    weak,
                    future_state,
                    future_time,
                    weak_future,
                )
                if self.component == "weak_endpoint_contrast_segmented":
                    selected = horizon * parts.velocity / horizon
                    self._record_rms("endpoint_contrast_rms", selected)
                elif self.component == "weak_endpoint_contrast":
                    noise_weight = condition.rho_second
                    selected = (
                        horizon * parts.velocity / horizon
                        if noise_weight == 1.0
                        else parts.velocity
                        + (noise_weight - 1.0) * parts.negative_noise
                    )
                    self._record_rms("endpoint_contrast_rms", selected)
                else:
                    selected = (
                        parts.clean
                        if "clean" in self.component
                        else parts.negative_noise
                    )
                    if self.component.endswith("_matched"):
                        selected = match_sample_rms(selected, parts.velocity)
                self._record_rms("endpoint_clean_change_rms", parts.clean)
                self._record_rms(
                    "endpoint_negative_noise_change_rms",
                    parts.negative_noise,
                )
                self._record_cosine(
                    "endpoint_component_cosine",
                    parts.clean,
                    parts.negative_noise,
                )
                self._record_rms(
                    "endpoint_identity_residual_rms",
                    parts.clean + parts.negative_noise - parts.velocity,
                )
                return guided + active_strength * selected

            if self.component == "weak_drift_oblique_curvature":
                spatial_fraction = condition.roundtrip_multiplier
                future_state = latent + spatial_fraction * horizon * guided
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                half_horizon = 0.5 * horizon
                weak_half = evaluate_weak_only(
                    time + time.new_tensor(half_horizon),
                    latent + spatial_fraction * half_horizon * guided,
                    self.labels,
                )
                self.future_nfe += 2
                weak_drift = horizon * (weak - weak_future) / horizon
                curvature_weight = condition.rho_second
                if curvature_weight == 1.0:
                    selected = weak_drift
                else:
                    first_oblique_change = richardson_forward_change(
                        weak,
                        weak_half,
                        weak_future,
                    )
                    selected = mix_material_curvature(
                        first_oblique_change,
                        weak_drift,
                        curvature_weight=curvature_weight,
                    )
                    self._record_rms(
                        "oblique_curvature_rms",
                        weak_drift - first_oblique_change,
                    )
                self._record_rms("oblique_curvature_mix_rms", selected)
                return guided + self.strength * selected

            if self.component == "weak_drift_oblique_picard":
                spatial_fraction = condition.roundtrip_multiplier
                velocity = guided
                for iteration in range(condition.roundtrip_iterations):
                    future_state = (
                        latent + spatial_fraction * horizon * velocity
                    )
                    weak_future = evaluate_weak_only(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 1
                    # K=1 intentionally preserves the oblique finite-difference
                    # operation order as a bitwise regression anchor.
                    weak_drift = horizon * (weak - weak_future) / horizon
                    self._record_rms(
                        f"oblique_picard_update_{iteration + 1}_rms",
                        weak_drift,
                    )
                    velocity = guided + self.strength * weak_drift
                return velocity

            if self.component == "weak_reference_forecast":
                spatial_fraction = condition.roundtrip_multiplier
                if spatial_fraction == 0.0:
                    future_state = latent
                elif spatial_fraction == 1.0:
                    future_state = latent + horizon * guided
                else:
                    future_state = latent + spatial_fraction * horizon * guided
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                forecast_factor = self.strength
                weak_effective = forecast_weak_reference(
                    weak,
                    weak_future,
                    factor=forecast_factor,
                )
                self._record_rms(
                    "forecast_weak_displacement_rms", weak_effective - weak
                )
                return full + gamma * (full - weak_effective)

            if self.component == "weak_drift_extended":
                future_state = latent + horizon * guided
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                weak_drift = horizon * (weak - weak_future) / horizon
                self._record_rms("extended_weak_drift_rms", weak_drift)
                return guided + active_strength * weak_drift

            if self.component == "weak_drift_midpoint":
                half_horizon = 0.5 * horizon
                half_time = time + time.new_tensor(half_horizon)
                half_state = latent + half_horizon * guided
                full_half, weak_half = evaluate_pair(
                    half_time,
                    half_state,
                    self.labels,
                )
                self.future_nfe += 1
                half_gamma = self._scheduled_gamma(
                    time_value + half_horizon
                )
                guided_half = (
                    full_half
                    if half_gamma == 0.0
                    else full_half + half_gamma * (full_half - weak_half)
                )
                future_state = latent + horizon * guided_half
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                weak_drift = weak - weak_future
                self._record_rms("midpoint_weak_drift_rms", weak_drift)
                return guided + self.strength * weak_drift

            if self.component == "weak_material_anisotropic":
                future_state = latent + horizon * guided
                if self.strength == condition.roundtrip_multiplier:
                    weak_future_along_path = evaluate_weak_only(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 1
                    # Keep the historical operation sequence as an exact
                    # anchor; an unused extra CUDA query can otherwise nudge
                    # an adaptive closed-loop solve onto another trajectory.
                    weak_drift = horizon * (
                        weak - weak_future_along_path
                    ) / horizon
                    return guided + self.strength * weak_drift
                weak_future_same_state = evaluate_weak_only(
                    future_time,
                    latent,
                    self.labels,
                )
                weak_future_along_path = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 2
                material = decompose_material_change(
                    weak,
                    weak_future_same_state,
                    weak_future_along_path,
                )
                self._record_rms("material_temporal_rms", material.temporal)
                self._record_rms("material_advective_rms", material.advective)
                self._record_cosine(
                    "material_time_vs_state_cosine",
                    material.temporal,
                    material.advective,
                )
                return (
                    guided
                    + self.strength * material.temporal
                    + condition.roundtrip_multiplier * material.advective
                )

            if self.component in {"weak_lie_bracket", "weak_lie_bracket_matched"}:
                guided_future_state = latent + horizon * guided
                weak_future_along_guided = evaluate_weak_only(
                    future_time,
                    guided_future_state,
                    self.labels,
                )
                weak_future_state = latent + horizon * weak
                full_future_along_weak, weak_future_along_weak = evaluate_pair(
                    future_time,
                    weak_future_state,
                    self.labels,
                )
                self.future_nfe += 2
                # Freeze the current piecewise-constant gamma while comparing
                # the two local flows. This avoids attributing a schedule jump
                # at t=.25 to their spatial non-commutativity.
                guided_future_along_weak = (
                    full_future_along_weak
                    if gamma == 0.0
                    else full_future_along_weak
                    + gamma * (full_future_along_weak - weak_future_along_weak)
                )
                weak_drift = weak - weak_future_along_guided
                bracket_change = finite_lie_bracket_change(
                    weak,
                    weak_future_along_guided,
                    guided,
                    guided_future_along_weak,
                )
                if self.component.endswith("_matched"):
                    bracket_change = match_sample_rms(
                        bracket_change,
                        weak_drift,
                    )
                self._record_rms("lie_bracket_change_rms", bracket_change)
                self._record_cosine(
                    "lie_bracket_vs_weak_drift_cosine",
                    bracket_change,
                    weak_drift,
                )
                return guided + self.strength * bracket_change

            if self.component in {
                "weak_drift_gap_parallel",
                "weak_drift_gap_orthogonal",
                "weak_reference_angle",
                "weak_reference_magnitude",
                "richardson_weak",
                "weak_drift_anisotropic",
                "weak_curvature_mix",
                "weak_drift_velocity_parallel",
                "weak_drift_velocity_orthogonal",
                "weak_velocity_angle",
                "weak_velocity_magnitude",
                "weak_drift_segmented",
            }:
                future_state = latent + horizon * guided
                weak_future = evaluate_weak_only(
                    future_time,
                    future_state,
                    self.labels,
                )
                self.future_nfe += 1
                weak_drift = horizon * (weak - weak_future) / horizon
                if self.component in {"richardson_weak", "weak_curvature_mix"}:
                    half_horizon = 0.5 * horizon
                    weak_half = evaluate_weak_only(
                        time + time.new_tensor(half_horizon),
                        latent + half_horizon * guided,
                        self.labels,
                    )
                    self.future_nfe += 1
                    first_material_change = richardson_forward_change(
                        weak,
                        weak_half,
                        weak_future,
                    )
                    selected = first_material_change
                    if self.component == "weak_curvature_mix":
                        selected = mix_material_curvature(
                            first_material_change,
                            weak_drift,
                            curvature_weight=condition.roundtrip_multiplier,
                        )
                        self._record_rms(
                            "material_curvature_rms",
                            weak_drift - first_material_change,
                        )
                    self._record_rms("richardson_correction_rms", selected)
                    return guided + self.strength * selected

                guidance_correction = guided - full
                projection = project_per_sample(weak_drift, guidance_correction)
                self._record_rms("weak_drift_rms", weak_drift)
                self._record_rms("weak_drift_parallel_rms", projection.parallel)
                self._record_rms(
                    "weak_drift_orthogonal_rms", projection.orthogonal
                )
                self._record_cosine(
                    "weak_drift_vs_guidance_cosine",
                    weak_drift,
                    guidance_correction,
                )
                if self.component == "weak_drift_gap_parallel":
                    return guided + self.strength * projection.parallel
                if self.component == "weak_drift_gap_orthogonal":
                    return guided + self.strength * projection.orthogonal
                if self.component == "weak_drift_anisotropic":
                    if self.strength == condition.roundtrip_multiplier:
                        return guided + self.strength * weak_drift
                    return (
                        guided
                        + self.strength * projection.orthogonal
                        + condition.roundtrip_multiplier * projection.parallel
                    )

                if self.component == "weak_drift_segmented":
                    segment_strength = (
                        self.strength
                        if time_value < FIRST_END
                        else condition.roundtrip_multiplier
                    )
                    return guided + segment_strength * weak_drift

                velocity_projection = project_per_sample(weak_drift, guided)
                self._record_rms(
                    "weak_drift_velocity_parallel_rms",
                    velocity_projection.parallel,
                )
                self._record_rms(
                    "weak_drift_velocity_orthogonal_rms",
                    velocity_projection.orthogonal,
                )
                self._record_cosine(
                    "weak_drift_vs_velocity_cosine",
                    weak_drift,
                    guided,
                )
                if self.component == "weak_drift_velocity_parallel":
                    return guided + self.strength * velocity_projection.parallel
                if self.component == "weak_drift_velocity_orthogonal":
                    return guided + self.strength * velocity_projection.orthogonal
                if self.component in {"weak_velocity_angle", "weak_velocity_magnitude"}:
                    corrected_velocity = guided + self.strength * weak_drift
                    if self.component == "weak_velocity_angle":
                        return match_sample_rms(corrected_velocity, guided)
                    return match_sample_rms(guided, corrected_velocity)

                corrected_guidance = (
                    guidance_correction + self.strength * weak_drift
                )
                self._record_rms(
                    "corrected_guidance_rms", corrected_guidance
                )
                if self.component == "weak_reference_angle":
                    corrected_guidance = match_sample_rms(
                        corrected_guidance, guidance_correction
                    )
                else:
                    corrected_guidance = match_sample_rms(
                        guidance_correction, corrected_guidance
                    )
                return full + corrected_guidance

            if self.component in {"picard_weak", "picard_weak_poly"}:
                weak_reference = weak
                velocity = guided
                previous_update_rms = None
                iteration_strengths = (
                    (self.strength, condition.roundtrip_multiplier)
                    if self.component == "picard_weak_poly"
                    else (self.strength,) * condition.roundtrip_iterations
                )
                for iteration, iteration_strength in enumerate(iteration_strengths):
                    future_state = latent + horizon * velocity
                    weak_future = evaluate_weak_only(
                        future_time,
                        future_state,
                        self.labels,
                    )
                    self.future_nfe += 1
                    # Preserve the historical FMD operation order so K=1 is
                    # an exact regression anchor, including FP32 rounding.
                    update = horizon * (weak_reference - weak_future) / horizon
                    self._record_rms(f"picard_update_{iteration + 1}_rms", update)
                    current_update_rms = sample_rms(update)
                    if previous_update_rms is not None:
                        ratio = current_update_rms / previous_update_rms.clamp_min(1e-12)
                        self.diagnostics.setdefault(
                            f"picard_update_{iteration + 1}_ratio", []
                        ).extend(ratio.detach().cpu().tolist())
                    velocity = velocity + iteration_strength * update
                    weak_reference = relax_future_weak_reference(
                        weak_reference,
                        weak_future,
                        gamma=gamma,
                        eta=iteration_strength,
                    )
                    previous_update_rms = current_update_rms
                return velocity

            future_state = latent + horizon * guided
            full_future, weak_future = evaluate_pair(
                future_time, future_state, self.labels
            )
            self.future_nfe += 1
            decomposition = decompose_future_weak_drift(
                full,
                weak,
                full_future,
                weak_future,
            )
            # Preserve the historical FMD operation order exactly.  The old
            # implementation formed a displacement and divided by the horizon;
            # cancelling that pair algebraically changes FP32 rounding enough
            # for an adaptive closed-loop solve to follow a different path.
            combined = (
                horizon * decomposition.weak_drift_correction / horizon
            )
            gap_change = horizon * decomposition.gap_change / horizon
            strong_curvature = (
                horizon * decomposition.strong_curvature_correction / horizon
            )
            if self.component in {"combined", "lookahead_weak"}:
                selected = combined
            elif self.component in {"gap_change", "gap_change_matched"}:
                selected = gap_change
            elif self.component in {
                "strong_curvature",
                "strong_curvature_matched",
            }:
                selected = strong_curvature
            else:
                raise ValueError(f"unsupported FMD decomposition: {self.component}")
            if self.component.endswith("_matched"):
                selected = match_sample_rms(selected, combined)

            identity_error = combined - (gap_change + strong_curvature)
            self._record_rms("combined_rms", combined)
            self._record_rms("gap_change_rms", gap_change)
            self._record_rms(
                "strong_curvature_rms",
                strong_curvature,
            )
            self._record_rms("selected_rms", selected)
            self._record_rms("identity_error_rms", identity_error)
            self._record_cosine(
                "gap_vs_strong_cosine",
                gap_change,
                strong_curvature,
            )
            self._record_cosine(
                "gap_vs_combined_cosine", gap_change, combined
            )
            self._record_cosine(
                "strong_vs_combined_cosine",
                strong_curvature,
                combined,
            )
            self._record_cosine("selected_vs_combined_cosine", selected, combined)
            if self.component == "lookahead_weak":
                return foresight_weak_guidance(
                    full,
                    weak,
                    weak_future,
                    gamma=gamma,
                    alpha=self.strength,
                )
            return guided + self.strength * selected

    class WeakField:
        def __init__(self, labels: Any):
            self.labels = labels
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            _, weak = evaluate_pair(time, latent, self.labels)
            return weak

    class ForesightResidualGuidedField:
        def __init__(
            self,
            labels: Any,
            *,
            residual: Any,
            strength: float,
        ):
            self.labels = labels
            self.residual = residual
            self.strength = float(strength)
            self.nfe = 0

        def __call__(self, time: Any, latent: Any) -> Any:
            self.nfe += 1
            full, weak = evaluate_pair(time, latent, self.labels)
            gamma = _gamma_at(float(time.detach().float().item()))
            if gamma == 0.0:
                return full
            correction = full - weak
            if self.strength != 0.0:
                correction = correction + self.strength * self.residual
            return full + gamma * correction

    images = np.empty((args.num_samples, 256, 256, 3), dtype=np.uint8)
    labels_array = np.empty(args.num_samples, dtype=np.int16)
    noise_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    total_strong_nfe = 0
    total_guided_nfe = 0
    total_weak_nfe = 0
    endpoint_gap_rms: dict[str, list[float]] = {"first": [], "second": [], "whole": []}
    roundtrip_move_rms: dict[str, list[float]] = {
        "first": [],
        "second": [],
        "whole": [],
    }
    residual_rms: dict[str, list[float]] = {}
    residual_parallel_fraction: dict[str, list[float]] = {}
    future_gap_cosine: dict[str, list[float]] = {}
    drift_decomposition_diagnostics: dict[str, list[float]] = {}
    cursor = 0
    preview = None

    def integrate_path_interval(
        state, labels, start, end, rho, key, *, probe_multiplier
    ):
        nonlocal total_strong_nfe, total_guided_nfe
        strong_field = StrongField(labels)
        guided_field = GuidedField(labels, multiplier=probe_multiplier)
        strong_endpoint = _integrate(
            odeint, strong_field, state, start, end, args, device
        )
        guided_endpoint = _integrate(
            odeint, guided_field, state, start, end, args, device
        )
        total_strong_nfe += strong_field.nfe
        total_guided_nfe += guided_field.nfe
        pair = PathEndpointPair(strong=strong_endpoint, guided=guided_endpoint)
        endpoint_gap_rms[key].extend(sample_rms(pair.displacement).cpu().tolist())
        return extrapolate_path_endpoints(pair, rho=rho)

    def integrate_roundtrip_interval(
        state,
        labels,
        start,
        end,
        total_relaxation,
        iterations,
        key,
    ):
        nonlocal total_strong_nfe, total_weak_nfe

        current = state
        relaxation = float(total_relaxation) / float(iterations)
        for _ in range(iterations):
            forward = StrongField(labels)
            future = _integrate(
                odeint, forward, current, start, end, args, device
            )
            inverse = WeakField(labels)
            mapped = _integrate(
                odeint, inverse, future, end, start, args, device
            )
            total_strong_nfe += forward.nfe
            total_weak_nfe += inverse.nfe
            move = mapped - current
            roundtrip_move_rms.setdefault(key, []).extend(
                sample_rms(move).cpu().tolist()
            )
            current = current + relaxation * move
        transport = StrongField(labels)
        endpoint = _integrate(
            odeint, transport, current, start, end, args, device
        )
        total_strong_nfe += transport.nfe
        return endpoint

    def roundtrip_displacement(state, labels, start, end, key):
        nonlocal total_strong_nfe, total_weak_nfe
        forward = StrongField(labels)
        future = _integrate(odeint, forward, state, start, end, args, device)
        inverse = WeakField(labels)
        mapped = _integrate(odeint, inverse, future, end, start, args, device)
        total_strong_nfe += forward.nfe
        total_weak_nfe += inverse.nfe
        move = mapped - state
        roundtrip_move_rms.setdefault(key, []).extend(sample_rms(move).cpu().tolist())
        return move

    def integrate_roundtrip_residual_interval(
        state, labels, start, end, strength, key
    ):
        nonlocal total_guided_nfe
        move = roundtrip_displacement(state, labels, start, end, key)
        time_value = torch.tensor(start, device=device, dtype=state.dtype)
        full, weak = evaluate_pair(time_value, state, labels)
        local_gap = full - weak
        projection = project_per_sample(move, local_gap)
        residual = projection.orthogonal
        residual_rms.setdefault(key, []).extend(sample_rms(residual).cpu().tolist())
        move_energy = sample_rms(move).square()
        parallel_energy = sample_rms(projection.parallel).square()
        fraction = parallel_energy / move_energy.clamp_min(
            torch.finfo(move_energy.dtype).tiny
        )
        residual_parallel_fraction.setdefault(key, []).extend(fraction.cpu().tolist())
        corrected = state + float(strength) * residual
        field = GuidedField(labels, multiplier=1.0)
        endpoint = _integrate(odeint, field, corrected, start, end, args, device)
        total_guided_nfe += field.nfe
        return endpoint

    def apply_fsg_calibration(
        state, labels, start, end, iterations, relaxation, key
    ):
        nonlocal total_guided_nfe, total_weak_nfe
        current = state
        for _ in range(iterations):
            forward = GuidedField(labels, multiplier=1.0)
            future = _integrate(
                odeint, forward, current, start, end, args, device
            )
            inverse = WeakField(labels)
            mapped = _integrate(
                odeint, inverse, future, end, start, args, device
            )
            total_guided_nfe += forward.nfe
            total_weak_nfe += inverse.nfe
            move = mapped - current
            roundtrip_move_rms.setdefault(key, []).extend(
                sample_rms(move).cpu().tolist()
            )
            current = current + float(relaxation) * move
        return current

    def integrate_fsg_base_interval(state, labels, start, end, fraction):
        nonlocal total_strong_nfe, total_guided_nfe
        if float(fraction) == 0.0:
            field = StrongField(labels)
            endpoint = _integrate(odeint, field, state, start, end, args, device)
            total_strong_nfe += field.nfe
            return endpoint
        field = GuidedField(labels, multiplier=float(fraction))
        endpoint = _integrate(odeint, field, state, start, end, args, device)
        total_guided_nfe += field.nfe
        return endpoint

    def future_gap_residual(state, labels, start, end, component, key):
        nonlocal total_strong_nfe
        start_time = torch.tensor(start, device=device, dtype=state.dtype)
        end_time = torch.tensor(end, device=device, dtype=state.dtype)
        full_now, weak_now = evaluate_pair(start_time, state, labels)
        current_gap = full_now - weak_now
        lookahead = StrongField(labels)
        future = _integrate(odeint, lookahead, state, start, end, args, device)
        total_strong_nfe += lookahead.nfe
        full_future, weak_future = evaluate_pair(end_time, future, labels)
        future_gap = full_future - weak_future
        matched_future = match_sample_rms(future_gap, current_gap)
        delta = matched_future - current_gap
        if component == "orthogonal":
            residual = project_per_sample(delta, current_gap).orthogonal
        elif component == "full":
            residual = delta
        else:
            raise ValueError(f"unsupported future residual component: {component}")
        residual_rms.setdefault(key, []).extend(sample_rms(residual).cpu().tolist())
        numerator = (current_gap.float().flatten(1) * future_gap.float().flatten(1)).sum(1)
        denominator = (
            current_gap.float().flatten(1).norm(dim=1)
            * future_gap.float().flatten(1).norm(dim=1)
        )
        cosine = numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
        future_gap_cosine.setdefault(key, []).extend(cosine.cpu().tolist())
        return residual

    def integrate_future_residual_interval(
        state, labels, start, end, strength, component, key
    ):
        nonlocal total_guided_nfe
        residual = future_gap_residual(
            state, labels, start, end, component, key
        )
        field = ForesightResidualGuidedField(
            labels, residual=residual, strength=strength
        )
        endpoint = _integrate(odeint, field, state, start, end, args, device)
        total_guided_nfe += field.nfe
        return endpoint

    with torch.inference_mode():
        while cursor < args.num_samples:
            current_batch = min(args.batch_size, args.num_samples - cursor)
            batch_index = cursor // args.batch_size
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
            noise = torch.randn(
                current_batch,
                *modules["LATENT_SHAPE"],
                generator=generator,
                device=device,
            )
            labels = torch.randint(
                0,
                modules["NUM_CLASSES"],
                (current_batch,),
                generator=generator,
                device=device,
            )
            if condition.kind in {"global", "local_scaled"}:
                multiplier = 1.0 if condition.kind == "global" else condition.local_multiplier
                field = GuidedField(labels, multiplier=multiplier)
                endpoint = _integrate(odeint, field, noise.float(), 0.0, 1.0, args, device)
                total_guided_nfe += field.nfe
            elif condition.kind == "semigroup_local_jensen":
                field = SemigroupLocalJensenField(labels)
                state = _integrate(
                    odeint, field, noise.float(), 0.0, GUIDANCE_END, args, device
                )
                total_guided_nfe += field.nfe
                for key, values in field.diagnostics.items():
                    drift_decomposition_diagnostics.setdefault(key, []).extend(values)
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "local_scaled_whole":
                field = GuidedField(labels, multiplier=condition.local_multiplier)
                state = _integrate(
                    odeint, field, noise.float(), 0.0, GUIDANCE_END, args, device
                )
                total_guided_nfe += field.nfe
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "local_scaled_segmented":
                state = noise.float()
                for start, end in ((0.0, FIRST_END), (FIRST_END, GUIDANCE_END)):
                    field = GuidedField(labels, multiplier=condition.local_multiplier)
                    state = _integrate(odeint, field, state, start, end, args, device)
                    total_guided_nfe += field.nfe
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "path_whole":
                state = integrate_path_interval(
                    noise.float(),
                    labels,
                    0.0,
                    GUIDANCE_END,
                    condition.rho_first,
                    "whole",
                    probe_multiplier=condition.probe_multiplier,
                )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "path_segmented":
                state = integrate_path_interval(
                    noise.float(),
                    labels,
                    0.0,
                    FIRST_END,
                    condition.rho_first,
                    "first",
                    probe_multiplier=condition.probe_multiplier,
                )
                state = integrate_path_interval(
                    state,
                    labels,
                    FIRST_END,
                    GUIDANCE_END,
                    condition.rho_second,
                    "second",
                    probe_multiplier=condition.probe_multiplier,
                )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "roundtrip_whole":
                mean_gamma = 0.5 * (BASE_GAMMA_FIRST + BASE_GAMMA_SECOND)
                state = integrate_roundtrip_interval(
                    noise.float(),
                    labels,
                    0.0,
                    GUIDANCE_END,
                    mean_gamma * condition.roundtrip_multiplier,
                    condition.roundtrip_iterations,
                    "whole",
                )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "roundtrip_segmented":
                state = noise.float()
                for start, end, gamma, key in (
                    (0.0, FIRST_END, BASE_GAMMA_FIRST, "first"),
                    (FIRST_END, GUIDANCE_END, BASE_GAMMA_SECOND, "second"),
                ):
                    state = integrate_roundtrip_interval(
                        state,
                        labels,
                        start,
                        end,
                        gamma * condition.roundtrip_multiplier,
                        condition.roundtrip_iterations,
                        key,
                    )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "roundtrip_partitioned":
                state = noise.float()
                boundaries = torch.linspace(
                    0.0,
                    GUIDANCE_END,
                    condition.partition_count + 1,
                    dtype=torch.float64,
                ).tolist()
                for index, (start, end) in enumerate(
                    zip(boundaries[:-1], boundaries[1:], strict=True)
                ):
                    gamma = _gamma_at(0.5 * (start + end))
                    state = integrate_roundtrip_interval(
                        state,
                        labels,
                        start,
                        end,
                        gamma * condition.roundtrip_multiplier,
                        1,
                        f"part{index}",
                    )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "roundtrip_residual_whole":
                state = integrate_roundtrip_residual_interval(
                    noise.float(),
                    labels,
                    0.0,
                    GUIDANCE_END,
                    condition.residual_strength,
                    "whole",
                )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "roundtrip_residual_segmented":
                state = noise.float()
                for start, end, key in (
                    (0.0, FIRST_END, "first"),
                    (FIRST_END, GUIDANCE_END, "second"),
                ):
                    state = integrate_roundtrip_residual_interval(
                        state,
                        labels,
                        start,
                        end,
                        condition.residual_strength,
                        key,
                    )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind.startswith("future_gap_"):
                component = condition.kind.removeprefix("future_gap_").removesuffix(
                    "_partitioned"
                )
                state = noise.float()
                boundaries = torch.linspace(
                    0.0,
                    GUIDANCE_END,
                    condition.partition_count + 1,
                    dtype=torch.float64,
                ).tolist()
                for index, (start, end) in enumerate(
                    zip(boundaries[:-1], boundaries[1:], strict=True)
                ):
                    state = integrate_future_residual_interval(
                        state,
                        labels,
                        start,
                        end,
                        condition.residual_strength,
                        component,
                        f"part{index}",
                    )
                late = StrongField(labels)
                endpoint = _integrate(
                    odeint, late, state, GUIDANCE_END, 1.0, args, device
                )
                total_strong_nfe += late.nfe
            elif condition.kind == "fsg_residual_continuous":
                field = ContinuousForesightResidualField(
                    labels,
                    horizon=condition.rho_first,
                    strength=condition.residual_strength,
                )
                endpoint = _integrate(
                    odeint, field, noise.float(), 0.0, 1.0, args, device
                )
                total_guided_nfe += field.nfe
                total_weak_nfe += field.future_nfe
            elif condition.kind.startswith("fmd_"):
                component = condition.kind.removeprefix("fmd_").removesuffix(
                    "_continuous"
                )
                field = ContinuousForesightDecompositionField(
                    labels,
                    horizon=condition.rho_first,
                    strength=condition.residual_strength,
                    component=component,
                )
                endpoint = _integrate(
                    odeint, field, noise.float(), 0.0, 1.0, args, device
                )
                total_guided_nfe += field.nfe + field.future_nfe
                for key, values in field.diagnostics.items():
                    drift_decomposition_diagnostics.setdefault(key, []).extend(values)
            elif condition.kind in {
                "fsg_strong_base",
                "fsg_ig_base",
                "fsg_hybrid_base",
                "fsg_budgeted_official",
                "fsg_budgeted_balanced",
                "fsg_early_official",
                "fsg_early_balanced",
            }:
                if condition.residual_strength < 0.0:
                    raise ValueError("FSG relaxation must be non-negative")
                base_fraction = {
                    "fsg_strong_base": 0.0,
                    "fsg_ig_base": 1.0,
                    "fsg_hybrid_base": condition.fsg_base_fraction,
                    "fsg_budgeted_official": condition.fsg_base_fraction,
                    "fsg_budgeted_balanced": condition.fsg_base_fraction,
                    "fsg_early_official": 0.0,
                    "fsg_early_balanced": 0.0,
                }[condition.kind]
                state = noise.float()
                current_time = 0.0
                events = _fsg_events(condition)
                if (
                    not condition.kind.startswith("fsg_early_")
                    and all(relaxation == 0.0 for _, _, _, relaxation, _ in events)
                ):
                    endpoint = integrate_fsg_base_interval(
                        state, labels, 0.0, 1.0, base_fraction
                    )
                    events = ()
                    current_time = 1.0
                for start, future_time, iterations, relaxation, key in events:
                    if current_time < start:
                        state = integrate_fsg_base_interval(
                            state,
                            labels,
                            current_time,
                            start,
                            base_fraction,
                        )
                    state = apply_fsg_calibration(
                        state,
                        labels,
                        start,
                        future_time,
                        iterations,
                        relaxation,
                        key,
                    )
                    current_time = start
                if condition.kind.startswith("fsg_early_"):
                    if current_time < FIRST_END:
                        state = integrate_fsg_base_interval(
                            state, labels, current_time, FIRST_END, 0.0
                        )
                    second = GuidedField(labels, multiplier=1.0)
                    state = _integrate(
                        odeint, second, state, FIRST_END, GUIDANCE_END, args, device
                    )
                    total_guided_nfe += second.nfe
                    late = StrongField(labels)
                    endpoint = _integrate(
                        odeint, late, state, GUIDANCE_END, 1.0, args, device
                    )
                    total_strong_nfe += late.nfe
                    current_time = 1.0
                if current_time < 1.0:
                    endpoint = integrate_fsg_base_interval(
                        state, labels, current_time, 1.0, base_fraction
                    )
            else:
                raise AssertionError(f"unhandled condition kind: {condition.kind}")
            if not torch.isfinite(endpoint).all():
                raise FloatingPointError(condition.name)
            decoded = modules["decode_latents_in_chunks"](
                vae,
                endpoint,
                scaling_factor=modules["SD_VAE_SCALING_FACTOR"],
                chunk_size=args.vae_decode_batch_size,
            )
            stop = cursor + current_batch
            images[cursor:stop] = modules["official_pixel_quantization"](decoded)
            labels_array[cursor:stop] = labels.cpu().numpy().astype(np.int16, copy=False)
            noise_hash.update(noise.cpu().contiguous().numpy().tobytes())
            label_hash.update(labels.cpu().contiguous().numpy().tobytes())
            if preview is None:
                preview = decoded[: min(16, len(decoded))].cpu()
            cursor = stop
            if cursor == current_batch or cursor == args.num_samples or cursor % 256 == 0:
                print(
                    json.dumps(
                        {
                            "condition": condition.name,
                            "generated": cursor,
                            "total": args.num_samples,
                            "strong_nfe": total_strong_nfe,
                            "guided_nfe": total_guided_nfe,
                            "weak_nfe": total_weak_nfe,
                        }
                    ),
                    flush=True,
                )

    sample_path = output / f"samples_n{args.num_samples}.npz"
    label_path = output / f"labels_n{args.num_samples}.npy"
    np.savez(sample_path, arr_0=images)
    np.save(label_path, labels_array, allow_pickle=False)
    assert preview is not None
    save_image(preview, output / "preview.png", nrow=4, normalize=True, value_range=(-1, 1))

    manifest = {
        "format": "eqvae_path_extrapolated_internal_guidance_samples_v1",
        "condition": condition.payload(),
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "integrator": "dopri5",
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "strong": strong_metadata,
        "head": {
            "depth": head.depth,
            "prediction_target": head.prediction_target,
            "checkpoint": head.checkpoint,
            "checkpoint_sha256": head.checkpoint_sha256,
        },
        "auxiliary_head": (
            None
            if depth8_head is None
            else {
                "depth": depth8_head.depth,
                "prediction_target": depth8_head.prediction_target,
                "checkpoint": depth8_head.checkpoint,
                "checkpoint_sha256": depth8_head.checkpoint_sha256,
            }
        ),
        "noise_sha256": noise_hash.hexdigest(),
        "label_sha256": label_hash.hexdigest(),
        "strong_nfe": total_strong_nfe,
        "guided_nfe": total_guided_nfe,
        "weak_nfe": total_weak_nfe,
        "total_nfe": total_strong_nfe + total_guided_nfe + total_weak_nfe,
        "path_endpoint_gap_rms": {
            key: summary
            for key, values in endpoint_gap_rms.items()
            if (summary := _summary(values)) is not None
        },
        "roundtrip_move_rms": {
            key: summary
            for key, values in roundtrip_move_rms.items()
            if (summary := _summary(values)) is not None
        },
        "foresight_residual_rms": {
            key: summary
            for key, values in residual_rms.items()
            if (summary := _summary(values)) is not None
        },
        "roundtrip_parallel_energy_fraction": {
            key: summary
            for key, values in residual_parallel_fraction.items()
            if (summary := _summary(values)) is not None
        },
        "current_future_gap_cosine": {
            key: summary
            for key, values in future_gap_cosine.items()
            if (summary := _summary(values)) is not None
        },
        "foresight_drift_decomposition": {
            key: summary
            for key, values in drift_decomposition_diagnostics.items()
            if (summary := _summary(values)) is not None
        },
        "samples": str(sample_path),
        "labels": str(label_path),
        **allocator,
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    modules["atomic_json_dump"](manifest, output / "sampling_manifest.json")

    metrics = None
    if not args.skip_fid:
        del vae, strong, heads, head
        gc.collect()
        torch.cuda.empty_cache()
        metric_path = output / "adm_metrics.json"
        environment = os.environ.copy()
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        subprocess.run(
            [
                str(paths["adm_python"]),
                str(paths["compute_fid"]),
                "--reference",
                str(paths["reference"]),
                "--samples",
                str(sample_path),
                "--batch-size",
                str(args.fid_batch_size),
                "--gpu-memory-fraction",
                str(args.fid_gpu_memory_fraction),
                "--output",
                str(metric_path),
            ],
            cwd=repo,
            env=environment,
            check=True,
        )
        metrics = read_json(metric_path)
    result = {
        "format": "eqvae_path_extrapolated_internal_guidance_result_v1",
        "condition": condition.payload(),
        "sampling_manifest": manifest,
        "metrics": metrics,
        "sample_retained": bool(args.keep_samples),
    }
    modules["atomic_json_dump"](result, result_path)
    if not args.keep_samples:
        sample_path.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "event": "complete",
                "condition": condition.name,
                "fid": None if metrics is None else metrics["fid"],
            }
        ),
        flush=True,
    )


def run_one(
    *,
    condition: Condition,
    gpu: int,
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = root / condition.name
    output.mkdir(parents=True, exist_ok=True)
    condition_path = output / "condition.json"
    atomic_json(condition_path, condition.payload())
    result_path = output / "condition_result.json"
    if reusable(result_path, condition, args):
        result = read_json(result_path)
        print(f"[reuse] {condition.name}", flush=True)
        return result
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "worker",
        "--repo",
        str(repo),
        "--data",
        str(data),
        "--adm-python",
        str(adm_python),
        "--condition-json",
        str(condition_path),
        "--output-dir",
        str(output),
        "--num-samples",
        str(args.num_samples),
        "--batch-size",
        str(args.batch_size),
        "--vae-decode-batch-size",
        str(args.vae_decode_batch_size),
        "--seed",
        str(args.seed),
        "--atol",
        str(args.atol),
        "--rtol",
        str(args.rtol),
        "--cuda-allocator-limit-gib",
        str(args.cuda_allocator_limit_gib),
        "--fid-batch-size",
        str(args.fid_batch_size),
        "--fid-gpu-memory-fraction",
        str(args.fid_gpu_memory_fraction),
    ]
    if args.keep_samples:
        command.append("--keep-samples")
    if args.skip_fid:
        command.append("--skip-fid")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = output / "run.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-80:])
        raise RuntimeError(f"{condition.name} failed on GPU {gpu}\n{tail}")
    result = read_json(result_path)
    metric = result.get("metrics")
    suffix = "" if metric is None else f" FID={float(metric['fid']):.4f}"
    print(f"[GPU {gpu}] {condition.name}:{suffix}", flush=True)
    return result


def run_parallel(
    conditions: tuple[Condition, ...],
    *,
    gpus: tuple[int, ...],
    root: Path,
    repo: Path,
    data: Path,
    adm_python: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    lanes: list[list[Condition]] = [[] for _ in gpus]
    for index, condition in enumerate(conditions):
        lanes[index % len(gpus)].append(condition)

    def lane(gpu: int, items: list[Condition]) -> list[dict[str, Any]]:
        return [
            run_one(
                condition=condition,
                gpu=gpu,
                root=root,
                repo=repo,
                data=data,
                adm_python=adm_python,
                args=args,
            )
            for condition in items
        ]

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(lane, gpu, items)
            for gpu, items in zip(gpus, lanes)
            if items
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def write_summary(root: Path, results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        condition = result["condition"]
        manifest = result["sampling_manifest"]
        metrics = result.get("metrics")
        rows.append(
            {
                "condition": condition["name"],
                "kind": condition["kind"],
                "rho_first": condition["rho_first"],
                "rho_second": condition["rho_second"],
                "probe_multiplier": condition["probe_multiplier"],
                "local_multiplier": condition["local_multiplier"],
                "roundtrip_iterations": condition["roundtrip_iterations"],
                "roundtrip_multiplier": condition["roundtrip_multiplier"],
                "residual_strength": condition["residual_strength"],
                "partition_count": condition["partition_count"],
                "fsg_base_fraction": condition["fsg_base_fraction"],
                "fid": None if metrics is None else float(metrics["fid"]),
                "sfid": None if metrics is None else float(metrics["sfid"]),
                "inception_score": (
                    None if metrics is None else float(metrics["inception_score"])
                ),
                "strong_nfe": int(manifest["strong_nfe"]),
                "guided_nfe": int(manifest["guided_nfe"]),
                "weak_nfe": int(manifest["weak_nfe"]),
                "total_nfe": int(manifest["total_nfe"]),
                "noise_sha256": manifest["noise_sha256"],
                "label_sha256": manifest["label_sha256"],
            }
        )
    rows.sort(key=lambda row: str(row["condition"]))
    noise_hashes = {row["noise_sha256"] for row in rows}
    label_hashes = {row["label_sha256"] for row in rows}
    if len(noise_hashes) != 1 or len(label_hashes) != 1:
        raise RuntimeError("conditions did not use paired noise and labels")
    if args.num_samples == 1000 and args.batch_size == 8 and args.seed == 0:
        if next(iter(noise_hashes)) != EXPECTED_NOISE:
            raise RuntimeError("noise hash differs from the historical FID-1K bank")
        if next(iter(label_hashes)) != EXPECTED_LABEL:
            raise RuntimeError("label hash differs from the historical FID-1K bank")
    anchor = next((row for row in rows if row["kind"] == "global"), None)
    if anchor is not None and anchor["fid"] is not None:
        if (
            args.num_samples == 1000
            and args.batch_size == 8
            and args.seed == 0
            and args.atol == 1e-6
            and args.rtol == 1e-3
            and abs(float(anchor["fid"]) - HISTORICAL_BEST_FID1K) > 0.15
        ):
            raise RuntimeError(
                "historical depth4 anchor did not reproduce: "
                f"new={anchor['fid']}, expected={HISTORICAL_BEST_FID1K}"
            )
        for row in rows:
            row["fid_delta_vs_global"] = (
                None if row["fid"] is None else float(row["fid"]) - float(anchor["fid"])
            )
    else:
        for row in rows:
            row["fid_delta_vs_global"] = None
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "all_conditions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    measured = [row for row in rows if row["fid"] is not None]
    best = min(measured, key=lambda row: float(row["fid"])) if measured else None
    summary = {
        "format": "eqvae_path_extrapolated_internal_guidance_summary_v1",
        "scientific_question": (
            "Can path-foresight information, after removing the duplicated "
            "local IG contrast, improve the best depth4 IG controller?"
        ),
        "protocol": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "atol": args.atol,
            "rtol": args.rtol,
        },
        "pairing": {
            "verified": True,
            "noise_sha256": next(iter(noise_hashes)),
            "label_sha256": next(iter(label_hashes)),
        },
        "historical_depth4_best_fid1k": HISTORICAL_BEST_FID1K,
        "global_anchor": anchor,
        "best": best,
        "rows": str(summary_dir / "all_conditions.csv"),
    }
    atomic_json(summary_dir / "summary.json", summary)
    if best is not None:
        delta = best["fid_delta_vs_global"]
        delta_text = "n/a" if delta is None else f"{float(delta):+.4f}"
        print(
            f"best {best['condition']}: FID={float(best['fid']):.4f} "
            f"delta={delta_text}",
            flush=True,
        )


def sweep(args: argparse.Namespace) -> None:
    repo = detect_repo()
    data = detect_data()
    adm_python = detect_adm_python()
    runtime_paths(repo, data, adm_python)
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else data / "internal_guidance_path_endpoint_v1/fid1k"
    )
    root.mkdir(parents=True, exist_ok=True)
    conditions = conditions_from_args(args)
    print(
        json.dumps(
            {
                "event": "launch",
                "conditions": [condition.name for condition in conditions],
                "gpus": args.gpus,
                "output_root": str(root),
            },
            indent=2,
        ),
        flush=True,
    )
    results = run_parallel(
        conditions,
        gpus=args.gpus,
        root=root,
        repo=repo,
        data=data,
        adm_python=adm_python,
        args=args,
    )
    write_summary(root, results, args)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vae-decode-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--cuda-allocator-limit-gib", type=float, default=6.0)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--fid-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep_parser = subparsers.add_parser("sweep")
    add_shared_arguments(sweep_parser)
    sweep_parser.add_argument("--gpus", type=parse_gpus, default=(0, 1, 2, 3))
    sweep_parser.add_argument("--output-root", type=Path)
    sweep_parser.add_argument(
        "--path-rhos", type=_parse_floats, default=(1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)
    )
    sweep_parser.add_argument(
        "--path-modes", type=_parse_modes, default=("whole", "segmented")
    )
    sweep_parser.add_argument("--path-rho-pairs", type=_parse_pairs, default=())
    sweep_parser.add_argument(
        "--matched-probe-alphas", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--local-multipliers", type=_parse_floats, default=(1.2, 1.5, 2.0)
    )
    sweep_parser.add_argument(
        "--local-modes", type=_parse_modes, default=("segmented",)
    )
    sweep_parser.add_argument(
        "--roundtrip-modes", type=_parse_modes, default=()
    )
    sweep_parser.add_argument(
        "--roundtrip-iterations", type=_parse_positive_ints, default=(1,)
    )
    sweep_parser.add_argument(
        "--roundtrip-multipliers", type=_parse_floats, default=(1.0,)
    )
    sweep_parser.add_argument(
        "--roundtrip-residual-modes", type=_parse_modes, default=()
    )
    sweep_parser.add_argument(
        "--roundtrip-residual-strengths", type=_parse_signed_floats, default=()
    )
    sweep_parser.add_argument(
        "--partition-counts", type=_parse_positive_ints, default=()
    )
    sweep_parser.add_argument(
        "--partitioned-roundtrip-multipliers", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--future-gap-components", type=_parse_future_components, default=()
    )
    sweep_parser.add_argument(
        "--future-gap-strengths", type=_parse_signed_floats, default=()
    )
    sweep_parser.add_argument("--fsg-bases", type=_parse_fsg_bases, default=())
    sweep_parser.add_argument(
        "--fsg-relaxations", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fsg-hybrid-pairs", type=_parse_pairs, default=()
    )
    sweep_parser.add_argument(
        "--fsg-budgeted-schedules", type=_parse_fsg_schedules, default=()
    )
    sweep_parser.add_argument(
        "--fsg-budgeted-specs", type=_parse_triples, default=()
    )
    sweep_parser.add_argument(
        "--fsg-early-schedules", type=_parse_fsg_schedules, default=()
    )
    sweep_parser.add_argument(
        "--fsg-early-relaxations", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fsg-residual-horizons", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fsg-residual-strengths", type=_parse_signed_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-decomposition-components",
        type=_parse_fmd_decomposition_components,
        default=(),
    )
    sweep_parser.add_argument(
        "--fmd-decomposition-horizons", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-decomposition-strengths", type=_parse_signed_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-picard-iterations", type=_parse_positive_ints, default=(1,)
    )
    sweep_parser.add_argument(
        "--fmd-picard-second-strengths", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-gamma-multipliers", type=_parse_floats, default=(1.0,)
    )
    sweep_parser.add_argument(
        "--fmd-gamma-segment-pairs", type=_parse_pairs, default=()
    )
    sweep_parser.add_argument(
        "--fmd-anisotropic-pairs", type=_parse_pairs, default=()
    )
    sweep_parser.add_argument(
        "--fmd-curvature-weights", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-material-pairs", type=_parse_pairs, default=()
    )
    sweep_parser.add_argument(
        "--fmd-strength-segment-pairs", type=_parse_pairs, default=()
    )
    sweep_parser.add_argument(
        "--fmd-characteristic-rhos", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-oblique-alphas", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-endpoint-noise-weights", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-oblique-specs", type=_parse_triples, default=()
    )
    sweep_parser.add_argument(
        "--fmd-forecast-factors", type=_parse_floats, default=()
    )
    sweep_parser.add_argument(
        "--fmd-extended-specs", type=_parse_triples, default=()
    )
    sweep_parser.add_argument("--condition-regex", default=".*")
    sweep_parser.add_argument(
        "--include-global-anchor", action=argparse.BooleanOptionalAction, default=True
    )
    sweep_parser.add_argument(
        "--include-semigroup-local-jensen",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    worker_parser = subparsers.add_parser("worker")
    add_shared_arguments(worker_parser)
    worker_parser.add_argument("--repo", type=Path, required=True)
    worker_parser.add_argument("--data", type=Path, required=True)
    worker_parser.add_argument("--adm-python", type=Path, required=True)
    worker_parser.add_argument("--condition-json", type=Path, required=True)
    worker_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts and batch sizes must be positive")
    if args.command == "worker":
        worker(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
