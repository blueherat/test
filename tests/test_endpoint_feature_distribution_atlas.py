from __future__ import annotations

import torch
import pandas as pd

from experiments.endpoint_feature_distribution_atlas import (
    FixedSpatialFeatureProjector,
    make_feature_chunks,
    paired_feature_metrics,
    projected_distribution_metrics,
    summarize_feature_chunks,
)
from experiments.summarize_endpoint_feature_distribution_atlas import (
    aggregate_atlas,
    branch_label,
)


def test_identity_features_have_exact_calibration_floor():
    features = torch.randn(
        16,
        8,
        6,
        6,
        generator=torch.Generator().manual_seed(11),
    )
    paired = paired_feature_metrics(features, features.clone())

    torch.testing.assert_close(
        paired["spatial_variance_ratio"],
        torch.ones(16),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        paired["centered_cosine"],
        torch.ones(16),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        paired["raw_mse"],
        torch.zeros(16),
        rtol=0,
        atol=0,
    )


def test_feature_contraction_is_detected_by_spatial_and_population_variance():
    reference = torch.randn(
        64,
        12,
        generator=torch.Generator().manual_seed(13),
    )
    candidate = 0.5 * reference
    distribution = projected_distribution_metrics(reference, candidate, seed=17)
    spatial = paired_feature_metrics(reference, candidate)

    assert abs(distribution["projected_covariance_trace_ratio"] - 0.25) < 1e-10
    assert (
        abs(distribution["projected_marginal_variance_ratio_gmean"] - 0.25)
        < 1e-10
    )
    torch.testing.assert_close(
        spatial["spatial_variance_ratio"],
        torch.full((64,), 0.25),
        rtol=1e-5,
        atol=1e-6,
    )


def test_projector_and_chunk_summary_are_deterministic():
    generator = torch.Generator().manual_seed(19)
    reference = torch.randn(16, 6, 8, 8, generator=generator)
    candidate = reference + 0.1 * torch.randn(
        16,
        6,
        8,
        8,
        generator=generator,
    )
    first_projector = FixedSpatialFeatureProjector(output_dim=9, seed=23)
    second_projector = FixedSpatialFeatureProjector(output_dim=9, seed=23)
    context = {"branch": "candidate", "num_steps": 4}
    first = make_feature_chunks(
        context=context,
        reference_features=(reference,),
        candidate_features=(candidate,),
        layer_indices=(3,),
        layer_fractions=(0.5,),
        projector=first_projector,
    )
    second = make_feature_chunks(
        context=context,
        reference_features=(reference,),
        candidate_features=(candidate,),
        layer_indices=(3,),
        layer_fractions=(0.5,),
        projector=second_projector,
    )

    torch.testing.assert_close(
        first[0]["reference_projection"],
        second[0]["reference_projection"],
        rtol=0,
        atol=0,
    )
    first_rows = summarize_feature_chunks(first, seed=29)
    second_rows = summarize_feature_chunks(second, seed=29)
    assert first_rows == second_rows
    assert first_rows[0]["sample_count"] == 16
    assert first_rows[0]["projection_dim"] == 9


def test_summary_keeps_branch_update_and_guidance_distinct():
    rows = []
    for update in (10, 50):
        for layer_fraction in (0.2, 1.0):
            rows.append(
                {
                    "atlas_name": "raev2",
                    "branch": "full_lpl",
                    "branch_update": update,
                    "guidance_scale": 1.78,
                    "noise_to_signal_ratio": 1.0,
                    "num_steps": 4,
                    "layer_fraction": layer_fraction,
                    "spatial_variance_ratio_gmean": 0.9,
                    "projected_covariance_trace_ratio": 0.8,
                    "projected_covariance_relative_error": 0.2,
                    "projected_normalized_frechet": 0.1,
                }
            )
    frame = pd.DataFrame(rows)
    frame["display_branch"] = branch_label(frame)
    condition, layer = aggregate_atlas(frame)

    assert condition["display_branch"].nunique() == 2
    assert layer["display_branch"].nunique() == 2


def test_branch_label_falls_back_to_old_rae_checkpoint_per_row():
    frame = pd.DataFrame(
        [
            {
                "atlas_name": "old_rae",
                "branch": pd.NA,
                "checkpoint": "lpl",
                "branch_update": pd.NA,
                "guidance_scale": pd.NA,
            },
            {
                "atlas_name": "raev2",
                "branch": "guided",
                "checkpoint": pd.NA,
                "branch_update": 50,
                "guidance_scale": 1.78,
            },
        ]
    )

    assert branch_label(frame).tolist() == [
        "old_rae:lpl",
        "raev2:guided@50,ig=1.78",
    ]
