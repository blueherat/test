import pandas as pd
import pytest
import torch

from experiments.rae_layerwise_path import random_detail_basis
from experiments.run_rae_path_gradient_interference import (
    component_losses,
    cross_split_metrics,
    evaluate_crossover_gradients,
    gradient_pair_metrics,
)


def test_component_losses_are_an_orthogonal_partition() -> None:
    prediction = torch.randn(2, 8, 4, 4)
    target = torch.randn_like(prediction)
    basis = random_detail_basis(8, 2, seed=4)
    semantic, detail = component_losses(prediction, target, basis)
    total = (prediction - target).square().mean()
    assert float(semantic + detail) == pytest.approx(float(total), abs=1e-6)


def test_gradient_pair_metrics_detect_conflict() -> None:
    metrics = gradient_pair_metrics(
        torch.tensor([1.0, 0.0]), torch.tensor([-0.5, 0.0])
    )
    assert metrics["semantic_basis_cosine"] == pytest.approx(-1.0)
    assert metrics["basis_over_semantic_norm"] == pytest.approx(0.5)
    assert metrics["semantic_descent_ratio"] == pytest.approx(0.5)
    assert metrics["basis_descent_ratio"] == pytest.approx(-1.0)


def test_cross_split_metrics_use_calibration_update_on_test_gradient() -> None:
    metrics = cross_split_metrics(
        torch.tensor([1.0, 0.0]),
        torch.tensor([-0.25, 0.0]),
        torch.tensor([2.0, 0.0]),
        torch.tensor([-0.5, 0.0]),
    )
    assert metrics["semantic_direction_stability"] == pytest.approx(1.0)
    assert metrics["basis_direction_stability"] == pytest.approx(1.0)
    assert metrics["cross_split_semantic_basis_cosine"] == pytest.approx(-1.0)
    assert metrics["cross_split_semantic_descent_ratio"] == pytest.approx(0.75)


def test_crossover_gradient_summary_detects_symmetric_late_effect() -> None:
    values = {
        "floor_to_floor": 0.85,
        "floor_to_static": 1.02,
        "static_to_static": 1.03,
        "static_to_floor": 0.90,
    }
    table = pd.DataFrame(
        [
            {
                "condition": condition,
                "checkpoint_step": 5000,
                "parameter_group": "last_block",
                "time": 0.1,
                "semantic_descent_ratio": value,
            }
            for condition, value in values.items()
        ]
    )
    result = evaluate_crossover_gradients(table)
    assert result["pass_late_path_gradient_prediction"]
