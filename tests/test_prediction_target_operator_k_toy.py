from __future__ import annotations

import torch

from experiments.run_prediction_target_extrapolation_toy_v4 import CurvedEmbedding
from experiments.run_prediction_target_operator_hybrid_from_rank_baseline import (
    hybrid_velocity,
)
from experiments.run_prediction_target_operator_k_toy import (
    TargetSpec,
    generalized_target,
    project_data_subspace,
    scalar_denominator,
    velocity_from_operator_output,
)


def test_operator_target_recovers_exact_velocity() -> None:
    device = torch.device("cpu")
    embedding = CurvedEmbedding(
        32,
        curvature=0.0,
        frequency_scale=6.0,
        seed=7,
        device=device,
        scale_mode="unit_rms",
    )
    generator = torch.Generator().manual_seed(11)
    intrinsic = torch.randn(17, 2, generator=generator)
    clean = embedding.embed(intrinsic)
    epsilon = torch.randn(clean.shape, generator=generator)
    time = torch.linspace(0.05, 0.95, len(clean))
    state = (1.0 - time[:, None]) * clean + time[:, None] * epsilon
    truth = epsilon - clean

    for tangent_k, normal_k in ((0.5, 0.5), (0.5, 0.8), (0.5, 1.0), (0.8, 0.8)):
        spec = TargetSpec("test", tangent_k=tangent_k, normal_k=normal_k)
        target = generalized_target(clean, epsilon, embedding, spec)
        recovered = velocity_from_operator_output(
            target,
            state,
            time,
            embedding,
            spec,
            conversion_clip=1e-6,
        )
        torch.testing.assert_close(recovered, truth, atol=2e-5, rtol=2e-5)


def test_projected_operator_matches_existing_oracle_hybrid() -> None:
    device = torch.device("cpu")
    embedding = CurvedEmbedding(
        24,
        curvature=0.0,
        frequency_scale=6.0,
        seed=13,
        device=device,
        scale_mode="unit_rms",
    )
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(19, 24, generator=generator)
    time = torch.linspace(0.1, 0.9, len(state))
    old_tangent_velocity_output = project_data_subspace(
        torch.randn(state.shape, generator=generator), embedding
    )
    operator_output = -0.5 * old_tangent_velocity_output
    spec = TargetSpec(
        "projected",
        tangent_k=0.5,
        normal_k=1.0,
        project_normal_output=True,
    )
    recovered = velocity_from_operator_output(
        operator_output,
        state,
        time,
        embedding,
        spec,
        conversion_clip=1e-6,
    )
    expected = hybrid_velocity(
        old_tangent_velocity_output,
        state,
        time,
        embedding,
        clip=1e-6,
    )
    torch.testing.assert_close(recovered, expected, atol=2e-5, rtol=2e-5)


def test_operator_denominator_is_positive_inside_time_interval() -> None:
    times = torch.linspace(0.001, 0.999, 1000)
    for k in (0.0, 0.5, 0.65, 0.8, 0.9, 1.0):
        denominator = scalar_denominator(times, k)
        assert torch.all(denominator > 0)
