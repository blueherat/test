from __future__ import annotations

import torch

from experiments.analyze_imagenet100_sit_cross_direction_response import (
    distribution_response_metrics,
)


def test_distribution_response_identical_shifts_have_unit_cosines() -> None:
    generator = torch.Generator().manual_seed(7)
    baseline = torch.randn(16, 5, generator=generator)
    candidate = 1.2 * baseline + 0.3
    metrics = distribution_response_metrics(baseline, candidate, candidate.clone())
    assert abs(metrics["mean_shift_cosine"] - 1.0) < 1e-10
    assert abs(metrics["covariance_shift_cosine"] - 1.0) < 1e-10
    assert abs(metrics["joint_moment_shift_cosine"] - 1.0) < 1e-10


def test_distribution_response_detects_opposite_mean_shift() -> None:
    generator = torch.Generator().manual_seed(11)
    baseline = torch.randn(32, 4, generator=generator)
    left = baseline + 0.5
    right = baseline - 0.5
    metrics = distribution_response_metrics(baseline, left, right)
    assert abs(metrics["mean_shift_cosine"] + 1.0) < 1e-10
