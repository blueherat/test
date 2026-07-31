from __future__ import annotations

import torch

from experiments.endpoint_feature_distribution_atlas import (
    FixedSpatialFeatureProjector,
    make_feature_chunks,
    paired_feature_metrics,
    projected_distribution_metrics,
    summarize_feature_chunks,
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
