import pytest
import torch

from experiments.small_image_gradient_allocation import gradient_metrics


def test_gradient_metrics_separate_allocation_from_conflict():
    coarse = torch.tensor([1.0, 0.0])
    detail = torch.tensor([-0.5, 1.0])
    weighted_coarse = 0.2 * coarse
    weighted_detail = 2.0 * detail
    metrics = gradient_metrics(coarse, detail, weighted_coarse, weighted_detail)

    assert metrics["coarse_detail_cosine_unweighted"] < 0
    assert metrics["allocation_multiplier"] == pytest.approx(10.0)
    assert metrics["coarse_descent_weighted"] < metrics["coarse_descent_baseline"]


def test_aligned_gradients_are_not_mislabeled_as_conflict():
    coarse = torch.tensor([1.0, 0.0])
    detail = torch.tensor([0.5, 1.0])
    metrics = gradient_metrics(coarse, detail, 0.2 * coarse, 2.0 * detail)
    assert metrics["coarse_detail_cosine_unweighted"] > 0
    assert metrics["allocation_multiplier"] == pytest.approx(10.0)
