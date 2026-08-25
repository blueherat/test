from __future__ import annotations

import math

import torch

from experiments.frechet_residual_score_toy import score_field
from experiments.run_advfd_smoothed_retraining_transport import (
    build_rotated_ring_pair,
    component_average_field,
    normalized_mean_update,
    reverse_kl,
    weighted_direction_rms,
)


def test_rotated_ring_pair_matches_mean_and_covariance() -> None:
    target, source = build_rotated_ring_pair(rotation=0.22)
    torch.testing.assert_close(target.moments().mean, source.moments().mean)
    torch.testing.assert_close(
        target.moments().covariance,
        source.moments().covariance,
        atol=1e-12,
        rtol=1e-12,
    )
    assert reverse_kl(target, source, quadrature_order=20) > 0.1


def test_component_score_step_lowers_reverse_kl() -> None:
    target, source = build_rotated_ring_pair(rotation=0.12)
    field = score_field(target, source)
    direction = component_average_field(source, field, quadrature_order=20)
    updated, _ = normalized_mean_update(
        source, direction, displacement_rms=0.005
    )
    assert reverse_kl(target, updated, quadrature_order=20) < reverse_kl(
        target, source, quadrature_order=20
    )


def test_normalized_mean_update_has_requested_rms() -> None:
    _, source = build_rotated_ring_pair(rotation=0.22)
    angles = torch.arange(8, dtype=torch.float64) * (2.0 * math.pi / 8.0)
    direction = torch.stack((angles.cos(), angles.sin()), dim=1)
    updated, scale = normalized_mean_update(
        source, direction, displacement_rms=0.03
    )
    displacement = updated.means - source.means
    torch.testing.assert_close(
        weighted_direction_rms(displacement, source.weights),
        torch.tensor(0.03, dtype=torch.float64),
    )
    assert scale > 0
