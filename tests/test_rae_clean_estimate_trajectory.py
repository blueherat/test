from __future__ import annotations

import pandas as pd
import torch

from experiments.rae_clean_estimate_trajectory import (
    PATHS,
    clean_estimate,
    endpoint_observation_factors,
    invert_path_endpoint_observation,
    project_to_endpoint_chord,
    rms_matched_projection,
    trajectory_prediction_summary,
)
from experiments.rae_cycle_direction_intervention import sample_rms
from experiments.rae_layerwise_path import plan_layerwise_path, random_detail_basis


def test_clean_estimate_inverts_linear_flow_state() -> None:
    clean = torch.randn(5, 7, 3, 3)
    noise = torch.randn_like(clean)
    time = torch.linspace(0.1, 0.9, len(clean))
    expanded = time.reshape(-1, 1, 1, 1)
    state = (1.0 - expanded) * clean + expanded * noise
    velocity = noise - clean
    torch.testing.assert_close(clean_estimate(state, velocity, time), clean)


def test_chord_projection_recovers_progress_and_orthogonal_residual() -> None:
    start = torch.randn(6, 4, 3, 3)
    end = torch.randn_like(start)
    chord = end - start
    residual = torch.randn_like(start)
    coefficient = (residual * chord).flatten(1).sum(1) / chord.square().flatten(1).sum(1)
    residual = residual - coefficient.reshape(-1, 1, 1, 1) * chord
    expected = torch.linspace(-0.2, 1.2, len(start))
    query = start + expected.reshape(-1, 1, 1, 1) * chord + residual
    progress, projection, curvature = project_to_endpoint_chord(query, start, end)
    torch.testing.assert_close(progress, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        ((query - projection) * chord).flatten(1).sum(1),
        torch.zeros(len(query)),
        atol=2e-5,
        rtol=0,
    )
    assert bool((curvature > 0).all())


def test_rms_matched_projection_matches_query_rms() -> None:
    query = torch.randn(8, 5, 4, 4) * 1.8
    projection = torch.randn_like(query) * 0.4
    matched = rms_matched_projection(query, projection)
    torch.testing.assert_close(sample_rms(matched), sample_rms(query))


def test_path_aware_inverse_recovers_teacher_endpoint() -> None:
    clean = torch.randn(4, 12, 4, 4)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.2, 0.5, 0.75, 0.9])
    basis = random_detail_basis(12, 3, seed=17)
    for mode, detail_scale in (
        ("static", 1.0),
        ("annealed", 1.0),
        ("annealed", 2.3),
        ("reverse", 1.0),
    ):
        plan = plan_layerwise_path(
            clean,
            noise,
            time,
            basis,
            mode=mode,
            power=2.0,
            detail_scale=detail_scale,
        )
        observation = clean_estimate(plan.state, plan.target, time)
        recovered = invert_path_endpoint_observation(
            observation,
            time,
            basis,
            mode=mode,
            power=2.0,
            detail_scale=detail_scale,
        )
        torch.testing.assert_close(recovered, clean, atol=2e-5, rtol=2e-5)


def test_endpoint_observation_factor_exposes_reverse_semantic_conditioning() -> None:
    time = torch.tensor([0.9])
    static_sem, static_detail = endpoint_observation_factors(time, "static", power=2.0)
    reverse_sem, reverse_detail = endpoint_observation_factors(time, "reverse", power=2.0)
    torch.testing.assert_close(static_sem, torch.ones_like(static_sem))
    torch.testing.assert_close(static_detail, torch.ones_like(static_detail))
    assert float(reverse_sem) < 0.05
    torch.testing.assert_close(reverse_detail, torch.ones_like(reverse_detail))


def test_path_aware_inverse_recovers_floored_power_and_rational_endpoints() -> None:
    clean = torch.randn(4, 12, 4, 4)
    noise = torch.randn_like(clean)
    time = torch.tensor([0.2, 0.5, 0.75, 0.9])
    basis = random_detail_basis(12, 3, seed=107)
    for family, power, floor, alpha in (
        ("power", 1.0, 0.05, 1.0),
        ("power", 2.0, 0.2, 1.0),
        ("rational", 2.0, 0.15, 0.5),
    ):
        plan = plan_layerwise_path(
            clean,
            noise,
            time,
            basis,
            mode="annealed",
            power=power,
            family=family,
            floor=floor,
            alpha=alpha,
        )
        observation = clean_estimate(plan.state, plan.target, time)
        recovered = invert_path_endpoint_observation(
            observation,
            time,
            basis,
            mode="annealed",
            power=power,
            family=family,
            floor=floor,
            alpha=alpha,
            detail_scale=1.0,
        )
        torch.testing.assert_close(recovered, clean, atol=2e-5, rtol=2e-5)


def test_prediction_summary_accepts_constructed_positive_case() -> None:
    distribution_rows = []
    latent_rows = []
    for path in PATHS:
        for step in (8, 16, 24):
            latent_rows.append(
                {
                    "path": path,
                    "step_index": step,
                    "kind": "actual",
                    "curvature_ratio": 0.2,
                    "progress": step / 49,
                }
            )
            for method, value in (("actual", 1.0), ("chord", 1.4), ("rms_chord", 1.3)):
                distribution_rows.append(
                    {
                        "path": path,
                        "step_index": step,
                        "method": method,
                        "projected_frechet": value,
                    }
                )
    summary, gate = trajectory_prediction_summary(
        pd.DataFrame(distribution_rows), pd.DataFrame(latent_rows)
    )
    assert len(summary) == 4
    assert gate["pass"] is True
