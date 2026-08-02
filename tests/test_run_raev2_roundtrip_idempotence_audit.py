from __future__ import annotations

import numpy as np
import torch

from experiments.run_raev2_roundtrip_idempotence_audit import (
    clamp_sample_metrics,
    sample_cosine,
    sample_rms,
)


def test_sample_rms_is_per_sample() -> None:
    value = torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]])
    expected = torch.tensor([np.sqrt(12.5), np.sqrt(2.0)], dtype=torch.float32)
    torch.testing.assert_close(sample_rms(value), expected)


def test_sample_cosine_handles_aligned_and_orthogonal_vectors() -> None:
    left = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    right = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    torch.testing.assert_close(sample_cosine(left, right), torch.tensor([1.0, 0.0]))


def test_clamp_metrics_separates_low_and_high_clipping() -> None:
    raw = torch.tensor([[[[-1.0, 0.5, 2.0, 0.25]]]])
    result = clamp_sample_metrics(raw)
    torch.testing.assert_close(result["below_zero_fraction"], torch.tensor([0.25]))
    torch.testing.assert_close(result["above_one_fraction"], torch.tensor([0.25]))
    torch.testing.assert_close(result["clipped_fraction"], torch.tensor([0.5]))
    assert float(result["clamp_distortion_rms"][0]) > 0
