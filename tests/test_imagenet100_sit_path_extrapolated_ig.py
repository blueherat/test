from __future__ import annotations

import json
from argparse import Namespace

import pytest

from experiments.run_imagenet100_sit_path_extrapolated_ig import (
    BASE_GAMMA_FIRST,
    BASE_GAMMA_SECOND,
    Condition,
    _fsg_events,
    _gamma_at,
    condition_from_payload,
    conditions_from_args,
    reusable,
)


def test_condition_payload_round_trip_shape() -> None:
    condition = Condition("path_segmented", rho_first=1.2, rho_second=1.4)
    payload = condition.payload()
    assert payload["name"] == "path_segmented_rho11p2_rho21p4"
    assert payload["path_formula"] == "Phi_S + rho*(Phi_IG-Phi_S)"


def test_whole_path_requires_one_rho() -> None:
    with pytest.raises(ValueError, match="one shared rho"):
        Condition("path_whole", rho_first=1.1, rho_second=1.2).validate()


def test_gamma_schedule_has_expected_boundaries() -> None:
    assert _gamma_at(0.0) == BASE_GAMMA_FIRST
    assert _gamma_at(0.249999) == BASE_GAMMA_FIRST
    assert _gamma_at(0.25) == BASE_GAMMA_SECOND
    assert _gamma_at(0.499999) == BASE_GAMMA_SECOND
    assert _gamma_at(0.5) == 0.0
    assert _gamma_at(0.1, multiplier=1.5) == pytest.approx(0.9)


def test_condition_matrix_deduplicates_uniform_and_explicit_pairs() -> None:
    args = Namespace(
        include_global_anchor=True,
        path_modes=("segmented",),
        path_rhos=(1.0, 1.2),
        path_rho_pairs=((1.0, 1.0), (1.1, 1.3)),
        matched_probe_alphas=(),
        local_multipliers=(1.2,),
        local_modes=("segmented",),
        roundtrip_modes=(),
        roundtrip_iterations=(1,),
        roundtrip_multipliers=(1.0,),
        condition_regex=".*",
    )
    conditions = conditions_from_args(args)
    names = [condition.name for condition in conditions]
    assert len(names) == len(set(names)) == 5
    assert "ig_depth4_best_global" in names
    assert "path_segmented_rho11p1_rho21p3" in names
    assert "local_segmented_gamma_multiplier1p2" in names


def test_matched_probe_holds_first_order_strength_fixed() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=("whole",),
        path_rhos=(1.0,),
        path_rho_pairs=(),
        matched_probe_alphas=(0.5, 2.0),
        local_multipliers=(1.0,),
        local_modes=("segmented",),
        roundtrip_modes=(),
        roundtrip_iterations=(1,),
        roundtrip_multipliers=(1.0,),
        condition_regex="^path_",
    )
    conditions = conditions_from_args(args)
    matched = [condition for condition in conditions if condition.probe_multiplier != 1.0]
    assert len(matched) == 2
    for condition in matched:
        assert condition.probe_multiplier * condition.rho_first == pytest.approx(1.0)


def test_roundtrip_condition_matrix_tracks_iterations_and_strength() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=("whole",),
        path_rhos=(1.0,),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(1.0,),
        local_modes=("segmented",),
        roundtrip_modes=("segmented",),
        roundtrip_iterations=(1, 2),
        roundtrip_multipliers=(0.5, 1.0),
        condition_regex="^roundtrip_",
    )
    conditions = conditions_from_args(args)
    assert {condition.name for condition in conditions} == {
        "roundtrip_segmented_k1_m0p5",
        "roundtrip_segmented_k1_m1",
        "roundtrip_segmented_k2_m0p5",
        "roundtrip_segmented_k2_m1",
    }


def test_foresight_residual_conditions_are_canonical() -> None:
    roundtrip = Condition(
        "roundtrip_residual_segmented", residual_strength=-0.5
    )
    future = Condition(
        "future_gap_orthogonal_partitioned",
        residual_strength=0.75,
        partition_count=8,
    )
    assert roundtrip.name == "roundtrip_residual_segmented_etam0p5"
    assert future.name == "future_gap_orthogonal_n8_rho0p75"
    assert future.payload()["format"].endswith("condition_v6")


def test_picard_weak_condition_tracks_horizon_eta_and_iterations() -> None:
    condition = Condition(
        "fmd_picard_weak_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_iterations=3,
    )
    assert condition.name == "fmd_decomposition_picard_weak_h0p046875_eta0p6_k3"
    payload = condition.payload()
    assert payload["foresight_material_derivative"]["picard_iterations"] == 3
    assert "rho=eta/gamma" in payload["foresight_residual_formula"]


def test_picard_weak_polynomial_condition_separates_two_updates() -> None:
    condition = Condition(
        "fmd_picard_weak_poly_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_iterations=2,
        roundtrip_multiplier=0.35,
    )
    assert condition.name == (
        "fmd_decomposition_picard_weak_poly_h0p046875_eta10p6_eta20p35"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["picard_iterations"] == 2
    assert material["first_strength"] == pytest.approx(0.6)
    assert material["second_strength"] == pytest.approx(0.35)


def test_picard_weak_condition_encodes_gamma_multiplier() -> None:
    condition = Condition(
        "fmd_picard_weak_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_iterations=2,
        local_multiplier=1.25,
    )
    assert condition.name.endswith("_k2_gm1p25")
    assert condition.payload()["foresight_material_derivative"][
        "gamma_multipliers"
    ] == {"first_interval": pytest.approx(1.25), "second_interval": pytest.approx(1.25)}


def test_picard_weak_condition_encodes_segmented_gamma_multipliers() -> None:
    condition = Condition(
        "fmd_picard_weak_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_iterations=2,
        local_multiplier=1.1,
        probe_multiplier=0.9 / 1.1,
    )
    assert condition.name.endswith("_k2_gm10p9_gm21p1")
    assert condition.payload()["foresight_material_derivative"][
        "gamma_multipliers"
    ] == {"first_interval": pytest.approx(0.9), "second_interval": pytest.approx(1.1)}


@pytest.mark.parametrize(
    "component",
    [
        "weak_drift_gap_parallel",
        "weak_drift_gap_orthogonal",
        "weak_reference_angle",
        "weak_reference_magnitude",
        "richardson_weak",
        "weak_drift_velocity_parallel",
        "weak_drift_velocity_orthogonal",
        "weak_velocity_angle",
        "weak_velocity_magnitude",
        "weak_lie_bracket",
        "weak_lie_bracket_matched",
        "weak_drift_midpoint",
    ],
)
def test_geometric_fmd_conditions_are_canonical(component: str) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p046875_eta0p6"
    assert condition_from_payload(condition.payload()) == condition


def test_anisotropic_fmd_condition_tracks_both_strengths() -> None:
    condition = Condition(
        "fmd_weak_drift_anisotropic_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_multiplier=0.3,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_anisotropic_h0p046875"
        "_orth0p6_parallel0p3"
    )
    assert condition_from_payload(condition.payload()) == condition


def test_curvature_mix_condition_tracks_curvature_weight() -> None:
    condition = Condition(
        "fmd_weak_curvature_mix_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_multiplier=1.5,
    )
    assert condition.name == (
        "fmd_decomposition_weak_curvature_mix_h0p046875"
        "_eta0p6_lambda1p5"
    )
    assert condition_from_payload(condition.payload()) == condition


def test_material_anisotropic_condition_tracks_time_and_state_strengths() -> None:
    condition = Condition(
        "fmd_weak_material_anisotropic_continuous",
        rho_first=0.046875,
        residual_strength=0.45,
        roundtrip_multiplier=0.75,
    )
    assert condition.name == (
        "fmd_decomposition_weak_material_anisotropic_h0p046875"
        "_time0p45_state0p75"
    )
    assert condition_from_payload(condition.payload()) == condition


def test_segmented_weak_drift_condition_tracks_both_intervals() -> None:
    condition = Condition(
        "fmd_weak_drift_segmented_continuous",
        rho_first=0.046875,
        residual_strength=0.45,
        roundtrip_multiplier=0.75,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_segmented_h0p046875"
        "_eta10p45_eta20p75"
    )
    assert condition_from_payload(condition.payload()) == condition


def test_path_mix_condition_tracks_characteristic_and_exact_anchor() -> None:
    condition = Condition(
        "fmd_weak_drift_path_mix_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
        roundtrip_multiplier=1.0,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_path_mix_h0p046875_eta0p6_rho1"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["characteristic_mix"] == pytest.approx(1.0)
    assert condition_from_payload(condition.payload()) == condition


def test_oblique_condition_tracks_spatial_transport_fraction() -> None:
    condition = Condition(
        "fmd_weak_drift_oblique_continuous",
        rho_first=0.046875,
        residual_strength=1.2,
        roundtrip_multiplier=0.25,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_oblique_h0p046875_eta1p2_alpha0p25"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["spatial_transport_fraction"] == pytest.approx(0.25)
    assert condition_from_payload(condition.payload()) == condition


def test_calibration_split_condition_has_no_extra_strength() -> None:
    condition = Condition(
        "fmd_weak_calibration_split_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == "fmd_decomposition_weak_calibration_split_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["extra_tuned_coefficients"] == 0
    assert "beta=1+gamma" in payload["foresight_residual_formula"]
    assert condition_from_payload(payload) == condition


def test_projected_calibration_condition_has_closed_form_query() -> None:
    condition = Condition(
        "fmd_weak_calibration_projected_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == (
        "fmd_decomposition_weak_calibration_projected_h0p03125"
    )
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["extra_tuned_coefficients"] == 0
    assert "argmin" in payload["foresight_residual_formula"]
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "geometry_fragment"),
    [
        ("weak_calibration_time_only", "information-time"),
        ("weak_calibration_characteristic", "frozen-Euler"),
        ("weak_calibration_weak_characteristic", "W field"),
        ("weak_calibration_strong_characteristic", "S field"),
        ("weak_calibration_projected_coupled", "same projected"),
    ],
)
def test_calibration_query_geometry_controls_are_explicit(
    component: str,
    geometry_fragment: str,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["extra_tuned_coefficients"] == 0
    assert geometry_fragment in material["query_geometry"]
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "anchor", "sign"),
    [
        ("weak_gap_transport_time_only", "weak", 1.0),
        ("weak_gap_transport_projected", "weak", 1.0),
        ("strong_gap_transport_projected", "strong", 1.0),
        ("weak_gap_antitransport_projected", "weak", -1.0),
    ],
)
def test_gap_transport_conditions_register_factorial_interaction(
    component: str,
    anchor: str,
    sign: float,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["factorial_interaction"] == "Omega=(S(q)-W(q))-(S-W)"
    assert material["common_mode_invariant"] is True
    assert material["diagonal_consistent"] is True
    assert material["anchor"] == anchor
    assert material["interaction_sign"] == pytest.approx(sign)
    assert material["extra_tuned_coefficients"] == 0
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "reference_noise", "aligned", "parameterization"),
    [
        ("score_noisier_aligned", "noisier", True, "endpoint_normalized_score"),
        (
            "score_noisier_same_state",
            "noisier",
            False,
            "endpoint_normalized_score",
        ),
        ("score_cleaner_aligned", "cleaner", True, "endpoint_normalized_score"),
        ("velocity_noisier_aligned", "noisier", True, "raw_velocity"),
    ],
)
def test_scale_space_conditions_register_density_semantics(
    component: str,
    reference_noise: str,
    aligned: bool,
    parameterization: str,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["reference_noise"] == reference_noise
    assert material["endpoint_coordinate_aligned"] is aligned
    assert material["parameterization"] == parameterization
    if parameterization == "endpoint_normalized_score":
        assert material["density_ratio_target"].startswith("q_S,current")
    else:
        assert material["density_ratio_target"] is None
    assert material["extra_tuned_coefficients"] == 0
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "branch", "reference_noise"),
    [
        ("marginal_score_weak_noisier", "weak", "noisier"),
        ("marginal_score_strong_noisier", "strong", "noisier"),
        ("marginal_score_weak_cleaner", "weak", "cleaner"),
    ],
)
def test_marginal_scale_space_conditions_are_prior_compatible(
    component: str,
    branch: str,
    reference_noise: str,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["temporal_branch"] == branch
    assert material["reference_noise"] == reference_noise
    assert material["coordinate_system"] == "path_marginal_z"
    assert material["prior_boundary_taper"] == "min(1,t/H)"
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    "component",
    [
        "velocity_parameterization_transport",
        "velocity_score_evolution",
        "velocity_change_recomposed",
    ],
)
def test_cross_time_velocity_components_register_exact_split(component: str) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["exact_decomposition"].startswith("W_t-W_ref=")
    assert material["reference_noise"] == "cleaner"
    assert material["extra_tuned_coefficients"] == 0
    assert condition_from_payload(payload) == condition


def test_calibration_innovation_condition_residualizes_static_gap() -> None:
    condition = Condition(
        "fmd_weak_calibration_innovation_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == (
        "fmd_decomposition_weak_calibration_innovation_h0p03125"
    )
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["extra_tuned_coefficients"] == 0
    assert "Proj_perp" in payload["foresight_residual_formula"]
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "levels"),
    [
        ("weak_calibration_depth8_response", "depth8 response only"),
        ("weak_calibration_telescoping_depth8", "depth4 -> depth8 -> full"),
    ],
)
def test_counterfactual_depth_hierarchy_conditions_are_affine_and_anchored(
    component: str,
    levels: str,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    assert condition.name == f"fmd_decomposition_{component}_h0p03125"
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["hierarchy_levels"] == levels
    assert material["ordinary_ig_anchor"] == "exact when q=p"
    assert material["affine_coefficient_sum"] == 1.0
    assert material["extra_tuned_coefficients"] == 0
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "ensemble"),
    [
        (
            "weak_calibration_reference_geomean",
            "equal time-only/projected references",
        ),
        (
            "weak_calibration_horizon_geomean",
            "equal half/full-horizon projected references",
        ),
    ],
)
def test_counterfactual_geomean_conditions_preserve_affine_closure(
    component: str,
    ensemble: str,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["reference_ensemble"] == ensemble
    assert material["affine_coefficient_sum"] == 1.0
    assert material["extra_tuned_coefficients"] == 0
    assert condition_from_payload(payload) == condition


@pytest.mark.parametrize(
    ("component", "axis"),
    [
        ("weak_calibration_innovation_strong_axis", "S"),
        ("weak_calibration_innovation_guided_axis", "G"),
    ],
)
def test_calibration_innovation_axis_controls_are_explicit(
    component: str,
    axis: str,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["extra_tuned_coefficients"] == 0
    assert f"Proj_perp_({axis})" in payload["foresight_residual_formula"]
    assert condition_from_payload(payload) == condition


def test_strong_anchored_guidance_innovation_has_zero_guidance_limit() -> None:
    condition = Condition(
        "fmd_weak_guidance_innovation_continuous",
        rho_first=0.03125,
        residual_strength=0.0,
    )
    payload = condition.payload()
    material = payload["foresight_material_derivative"]
    assert material["anchor"] == "strong"
    assert material["zero_guidance_limit"] == "exactly S"
    assert "gamma*(W-W(q))" in payload["foresight_residual_formula"]
    assert condition_from_payload(payload) == condition


def test_oblique_picard_condition_tracks_self_consistent_iterations() -> None:
    condition = Condition(
        "fmd_weak_drift_oblique_picard_continuous",
        rho_first=0.02734375,
        residual_strength=1.65,
        roundtrip_multiplier=0.25,
        roundtrip_iterations=2,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_oblique_picard"
        "_h0p027344_eta1p65_alpha0p25_k2"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["spatial_transport_fraction"] == pytest.approx(0.25)
    assert material["picard_iterations"] == 2
    assert material["fixed_point"].startswith("V=G+eta")
    assert condition_from_payload(condition.payload()) == condition


def test_oblique_curvature_condition_tracks_finite_horizon_weight() -> None:
    condition = Condition(
        "fmd_weak_drift_oblique_curvature_continuous",
        rho_first=0.02734375,
        rho_second=1.25,
        residual_strength=1.65,
        roundtrip_multiplier=0.25,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_oblique_curvature"
        "_h0p027344_eta1p65_alpha0p25_lambda1p25"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["spatial_transport_fraction"] == pytest.approx(0.25)
    assert material["curvature_weight"] == pytest.approx(1.25)
    assert "lambda=1 is exactly" in material["coefficient_semantics"]
    assert condition_from_payload(condition.payload()) == condition


@pytest.mark.parametrize(
    "component,selected,matched",
    [
        ("weak_clean_endpoint_drift", "clean", False),
        ("weak_clean_endpoint_drift_matched", "clean", True),
        ("weak_noise_endpoint_drift", "negative_noise", False),
        ("weak_noise_endpoint_drift_matched", "negative_noise", True),
    ],
)
def test_endpoint_drift_conditions_record_exact_component_semantics(
    component: str,
    selected: str,
    matched: bool,
) -> None:
    condition = Condition(
        f"fmd_{component}_continuous",
        rho_first=0.02734375,
        residual_strength=1.65,
        roundtrip_multiplier=0.25,
    )
    assert condition.name == (
        f"fmd_decomposition_{component}"
        "_h0p027344_eta1p65_alpha0p25"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["selected_endpoint_component"] == selected
    assert material["rms_matched_to_velocity_change"] is matched
    assert condition_from_payload(condition.payload()) == condition


def test_endpoint_contrast_condition_records_coordinate_invariant_weight() -> None:
    condition = Condition(
        "fmd_weak_endpoint_contrast_continuous",
        rho_first=0.02734375,
        rho_second=1.25,
        residual_strength=1.65,
        roundtrip_multiplier=0.25,
    )
    assert condition.name == (
        "fmd_decomposition_weak_endpoint_contrast"
        "_h0p027344_eta1p65_alpha0p25_lambda1p25"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["negative_noise_weight"] == pytest.approx(1.25)
    assert material["coordinate_invariant_weight"] == pytest.approx(1.0)
    assert condition_from_payload(condition.payload()) == condition


def test_segmented_endpoint_contrast_records_time_dependent_strengths() -> None:
    condition = Condition(
        "fmd_weak_endpoint_contrast_segmented_continuous",
        rho_first=0.02734375,
        rho_second=1.8,
        residual_strength=1.5,
        roundtrip_multiplier=0.25,
    )
    assert condition.name == (
        "fmd_decomposition_weak_endpoint_contrast_segmented"
        "_h0p027344_eta11p5_eta21p8_alpha0p25"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["segment_strengths"] == {
        "first_interval": pytest.approx(1.5),
        "second_interval": pytest.approx(1.8),
    }
    assert condition_from_payload(condition.payload()) == condition


def test_weak_reference_forecast_condition_has_dimensionless_factor() -> None:
    condition = Condition(
        "fmd_weak_reference_forecast_continuous",
        rho_first=0.046875,
        residual_strength=2.0,
        roundtrip_multiplier=0.25,
    )
    assert condition.name == (
        "fmd_decomposition_weak_reference_forecast_h0p046875_r2_alpha0p25"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["forecast_factor"] == pytest.approx(2.0)
    assert material["spatial_transport_fraction"] == pytest.approx(0.25)
    assert condition_from_payload(condition.payload()) == condition


def test_extended_weak_drift_condition_tracks_late_window() -> None:
    condition = Condition(
        "fmd_weak_drift_extended_continuous",
        rho_first=0.046875,
        rho_second=0.75,
        residual_strength=0.6,
        roundtrip_multiplier=0.2,
    )
    assert condition.name == (
        "fmd_decomposition_weak_drift_extended_h0p046875"
        "_early0p6_late0p2_end0p75"
    )
    material = condition.payload()["foresight_material_derivative"]
    assert material["segment_strengths"] == {
        "ig_interval": pytest.approx(0.6),
        "post_ig_interval": pytest.approx(0.2),
    }
    assert condition_from_payload(condition.payload()) == condition
    with pytest.raises(ValueError, match="correction end"):
        Condition(
            "fmd_weak_drift_extended_continuous",
            rho_first=0.046875,
            rho_second=0.5,
            residual_strength=0.6,
            roundtrip_multiplier=0.2,
        ).validate()


def test_characteristic_oblique_and_extended_condition_grids_are_explicit() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=(
            "weak_drift_path_mix",
            "weak_drift_oblique",
            "weak_reference_forecast",
            "weak_drift_extended",
        ),
        fmd_decomposition_horizons=(0.046875,),
        fmd_decomposition_strengths=(0.6,),
        fmd_characteristic_rhos=(0.6, 1.0),
        fmd_oblique_alphas=(0.25, 1.0),
        fmd_oblique_specs=(),
        fmd_forecast_factors=(2.0,),
        fmd_extended_specs=((0.6, 0.2, 0.75),),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_drift_path_mix_h0p046875_eta0p6_rho0p6",
        "fmd_decomposition_weak_drift_path_mix_h0p046875_eta0p6_rho1",
        "fmd_decomposition_weak_drift_oblique_h0p046875_eta0p6_alpha0p25",
        "fmd_decomposition_weak_drift_oblique_h0p046875_eta0p6_alpha1",
        "fmd_decomposition_weak_reference_forecast_h0p046875_r2_alpha0p25",
        "fmd_decomposition_weak_reference_forecast_h0p046875_r2_alpha1",
        "fmd_decomposition_weak_drift_extended_h0p046875_early0p6_late0p2_end0p75",
    }


def test_oblique_specs_do_not_form_an_unintended_cartesian_product() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=("weak_drift_oblique",),
        fmd_decomposition_horizons=(),
        fmd_decomposition_strengths=(),
        fmd_oblique_alphas=(),
        fmd_oblique_specs=((0.03125, 1.8, 0.25), (0.0625, 0.9, 0.25)),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_drift_oblique_h0p03125_eta1p8_alpha0p25",
        "fmd_decomposition_weak_drift_oblique_h0p0625_eta0p9_alpha0p25",
    }


def test_oblique_picard_grid_only_varies_requested_iterations() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=("weak_drift_oblique_picard",),
        fmd_decomposition_horizons=(0.02734375,),
        fmd_decomposition_strengths=(1.65,),
        fmd_oblique_alphas=(0.25,),
        fmd_picard_iterations=(1, 2, 3),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_drift_oblique_picard"
        "_h0p027344_eta1p65_alpha0p25_k1",
        "fmd_decomposition_weak_drift_oblique_picard"
        "_h0p027344_eta1p65_alpha0p25_k2",
        "fmd_decomposition_weak_drift_oblique_picard"
        "_h0p027344_eta1p65_alpha0p25_k3",
    }


def test_oblique_curvature_grid_only_varies_requested_weights() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=("weak_drift_oblique_curvature",),
        fmd_decomposition_horizons=(0.02734375,),
        fmd_decomposition_strengths=(1.65,),
        fmd_oblique_alphas=(0.25,),
        fmd_curvature_weights=(0.75, 1.0, 1.25),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_drift_oblique_curvature"
        "_h0p027344_eta1p65_alpha0p25_lambda0p75",
        "fmd_decomposition_weak_drift_oblique_curvature"
        "_h0p027344_eta1p65_alpha0p25_lambda1",
        "fmd_decomposition_weak_drift_oblique_curvature"
        "_h0p027344_eta1p65_alpha0p25_lambda1p25",
    }


def test_endpoint_drift_grid_selects_only_requested_components() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=(
            "weak_clean_endpoint_drift",
            "weak_noise_endpoint_drift_matched",
        ),
        fmd_decomposition_horizons=(0.02734375,),
        fmd_decomposition_strengths=(1.65,),
        fmd_oblique_alphas=(0.25,),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_clean_endpoint_drift"
        "_h0p027344_eta1p65_alpha0p25",
        "fmd_decomposition_weak_noise_endpoint_drift_matched"
        "_h0p027344_eta1p65_alpha0p25",
    }


def test_endpoint_contrast_grid_only_varies_noise_weight() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=("weak_endpoint_contrast",),
        fmd_decomposition_horizons=(0.02734375,),
        fmd_decomposition_strengths=(1.65,),
        fmd_oblique_alphas=(0.25,),
        fmd_endpoint_noise_weights=(0.75, 1.0, 1.25),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_endpoint_contrast"
        "_h0p027344_eta1p65_alpha0p25_lambda0p75",
        "fmd_decomposition_weak_endpoint_contrast"
        "_h0p027344_eta1p65_alpha0p25_lambda1",
        "fmd_decomposition_weak_endpoint_contrast"
        "_h0p027344_eta1p65_alpha0p25_lambda1p25",
    }


def test_segmented_endpoint_contrast_grid_uses_requested_pairs() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=("weak_endpoint_contrast_segmented",),
        fmd_decomposition_horizons=(0.02734375,),
        fmd_strength_segment_pairs=((0.0, 1.65), (1.65, 0.0), (1.5, 1.8)),
        fmd_oblique_alphas=(0.25,),
        fmd_gamma_multipliers=(1.0,),
        fmd_gamma_segment_pairs=(),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_endpoint_contrast_segmented"
        "_h0p027344_eta10_eta21p65_alpha0p25",
        "fmd_decomposition_weak_endpoint_contrast_segmented"
        "_h0p027344_eta11p65_eta20_alpha0p25",
        "fmd_decomposition_weak_endpoint_contrast_segmented"
        "_h0p027344_eta11p5_eta21p8_alpha0p25",
    }


def test_segmented_endpoint_contrast_grid_varies_ig_gamma_schedule() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=(),
        path_rhos=(),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(),
        local_modes=(),
        roundtrip_modes=(),
        roundtrip_iterations=(),
        roundtrip_multipliers=(),
        fmd_decomposition_components=("weak_endpoint_contrast_segmented",),
        fmd_decomposition_horizons=(0.02734375,),
        fmd_strength_segment_pairs=((1.65, 1.65),),
        fmd_oblique_alphas=(0.25,),
        fmd_gamma_multipliers=(1.0,),
        fmd_gamma_segment_pairs=((0.9, 1.1),),
        condition_regex="^fmd_decomposition_",
    )
    names = {condition.name for condition in conditions_from_args(args)}
    assert names == {
        "fmd_decomposition_weak_endpoint_contrast_segmented"
        "_h0p027344_eta11p65_eta21p65_alpha0p25",
        "fmd_decomposition_weak_endpoint_contrast_segmented"
        "_h0p027344_eta11p65_eta21p65_alpha0p25_gm10p9_gm21p1",
    }


def test_condition_payload_accepts_canonical_legacy_v5() -> None:
    condition = Condition(
        "fsg_residual_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
    )
    assert condition_from_payload(condition.legacy_payload()) == condition
    payload = condition.payload()
    assert payload["foresight_material_derivative"] == {
        "lookahead_horizon": 0.046875,
        "finite_difference_strength": 0.6,
        "peak_material_derivative_gain": pytest.approx(0.028125),
        "small_h_limit": "G - kappa*(partial_t W4 + J_W4 G)",
        "boundary_taper": "kappa(t)=eta*min(H,0.5-t)",
    }


def test_foresight_partition_count_must_respect_gamma_boundary() -> None:
    with pytest.raises(ValueError, match="positive even"):
        Condition(
            "future_gap_full_partitioned",
            residual_strength=1.0,
            partition_count=3,
        ).validate()


def test_fsg_condition_names_encode_base_and_relaxation() -> None:
    assert (
        Condition("fsg_strong_base", residual_strength=0.25).name
        == "fsg_strong_base_rho0p25"
    )
    assert (
        Condition("fsg_ig_base", residual_strength=1.0).name
        == "fsg_ig_base_rho1"
    )
    assert (
        Condition(
            "fsg_hybrid_base",
            residual_strength=0.16,
            fsg_base_fraction=0.5,
        ).name
        == "fsg_hybrid_lam0p5_rho0p16"
    )


def test_budgeted_fsg_events_use_segment_specific_relaxations() -> None:
    official = Condition(
        "fsg_budgeted_official",
        rho_first=0.1875,
        rho_second=0.823529,
    )
    balanced = Condition(
        "fsg_budgeted_balanced",
        rho_first=0.375,
        rho_second=0.411765,
    )

    assert [event[3] for event in _fsg_events(official)] == [
        0.1875,
        0.1875,
        0.823529,
    ]
    assert [event[:4] for event in _fsg_events(balanced)] == [
        (0.0, 0.125, 1, 0.375),
        (0.125, 0.25, 1, 0.375),
        (0.25, 0.375, 1, 0.411765),
        (0.375, 0.5, 1, 0.411765),
    ]
    assert official.name == (
        "fsg_budgeted_official_lam0_rho10p1875_rho20p823529"
    )


def test_budgeted_fsg_condition_matrix_is_explicit() -> None:
    args = Namespace(
        include_global_anchor=False,
        path_modes=("whole",),
        path_rhos=(1.0,),
        path_rho_pairs=(),
        matched_probe_alphas=(),
        local_multipliers=(1.0,),
        local_modes=("segmented",),
        roundtrip_modes=(),
        roundtrip_iterations=(1,),
        roundtrip_multipliers=(1.0,),
        fsg_budgeted_schedules=("official", "balanced"),
        fsg_budgeted_specs=((0.5, 0.1, 0.2),),
        condition_regex="^fsg_budgeted_",
    )
    conditions = conditions_from_args(args)
    assert {condition.name for condition in conditions} == {
        "fsg_budgeted_official_lam0p5_rho10p1_rho20p2",
        "fsg_budgeted_balanced_lam0p5_rho10p1_rho20p2",
    }


def test_early_fsg_only_replaces_first_ig_segment() -> None:
    official = Condition("fsg_early_official", residual_strength=0.1875)
    balanced = Condition("fsg_early_balanced", residual_strength=0.375)

    assert [event[2:] for event in _fsg_events(official)] == [
        (2, 0.1875, "event0"),
        (2, 0.1875, "event1"),
    ]
    assert [event[2:] for event in _fsg_events(balanced)] == [
        (1, 0.375, "event0"),
        (1, 0.375, "event1"),
    ]
    assert official.name == "fsg_early_official_rho0p1875"
    assert balanced.name == "fsg_early_balanced_rho0p375"


def test_continuous_fsg_residual_names_horizon_and_signed_strength() -> None:
    condition = Condition(
        "fsg_residual_continuous",
        rho_first=0.125,
        residual_strength=-0.5,
    )
    assert condition.name == "fsg_residual_continuous_h0p125_etam0p5"
    with pytest.raises(ValueError, match="horizon must be positive"):
        Condition(
            "fsg_residual_continuous",
            rho_first=0.0,
            residual_strength=1.0,
        ).validate()


def test_reusable_requires_matching_solver_and_retained_sample_protocol(
    tmp_path,
) -> None:
    condition = Condition(
        "fsg_residual_continuous",
        rho_first=0.046875,
        residual_strength=0.6,
    )
    sample_path = tmp_path / "samples.npz"
    result_path = tmp_path / "condition_result.json"
    payload = {
        "condition": condition.payload(),
        "sample_retained": False,
        "sampling_manifest": {
            "sampling": {
                "integrator": "dopri5",
                "num_samples": 1000,
                "batch_size": 8,
                "seed": 0,
                "atol": 1e-6,
                "rtol": 1e-3,
            },
            "noise_sha256": "noise",
            "label_sha256": "labels",
            "samples": str(sample_path),
        },
        "metrics": {"fid": 1.0, "sfid": 2.0, "inception_score": 3.0},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    args = Namespace(
        num_samples=1000,
        batch_size=8,
        seed=0,
        atol=1e-6,
        rtol=1e-3,
        keep_samples=False,
        skip_fid=False,
    )

    assert reusable(result_path, condition, args)
    args.rtol = 3e-4
    assert not reusable(result_path, condition, args)
    args.rtol = 1e-3
    args.keep_samples = True
    assert not reusable(result_path, condition, args)
    sample_path.touch()
    payload["sample_retained"] = True
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    assert reusable(result_path, condition, args)
