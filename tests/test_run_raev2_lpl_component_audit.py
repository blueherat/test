import numpy as np
import torch

from experiments.lpl_component_metrics import (
    component_metrics,
    normalized_descent_effect,
)
from experiments.run_raev2_lpl_component_audit import (
    select_prediction_target,
    selected_dataset_indices,
)


def test_component_metrics_detect_exact_gradient_split() -> None:
    error = torch.tensor([[2.0, 0.0]])
    variance = torch.tensor([[-1.0, 0.0]])
    full = error + variance
    metrics = component_metrics(
        flow_gradient=torch.tensor([[1.0, 0.0]]),
        raw_gradient=torch.tensor([[3.0, 0.0]]),
        error_gradient=error,
        variance_gradient=variance,
        full_gradient=full,
        log_variance_gradient=torch.tensor([[1.0, 0.0]]),
    )

    torch.testing.assert_close(
        metrics["gradient_split_relative_residual"],
        torch.zeros(1),
    )
    torch.testing.assert_close(
        metrics["variance_over_error_gradient_rms"],
        torch.full((1,), 0.5),
    )
    torch.testing.assert_close(
        metrics["flow_projected_gradient_fraction"],
        torch.ones(1),
    )
    torch.testing.assert_close(
        metrics["flow_projection_conflict"],
        torch.zeros(1),
    )
    assert float(metrics["variance_descent_log_variance_change"]) > 0.0


def test_component_metrics_marks_opposing_lpl_as_conflicting() -> None:
    metrics = component_metrics(
        flow_gradient=torch.tensor([[1.0, 0.0]]),
        raw_gradient=torch.tensor([[1.0, 0.0]]),
        error_gradient=torch.tensor([[-0.5, 0.0]]),
        variance_gradient=torch.tensor([[-0.5, 0.0]]),
        full_gradient=torch.tensor([[-1.0, 0.0]]),
        log_variance_gradient=torch.tensor([[1.0, 0.0]]),
    )

    torch.testing.assert_close(
        metrics["flow_projected_gradient_fraction"],
        torch.zeros(1),
    )
    torch.testing.assert_close(
        metrics["flow_projection_conflict"],
        torch.ones(1),
    )


def test_normalized_descent_effect_reports_descent_sign() -> None:
    gradient = torch.tensor([[3.0, 4.0]])
    effect = normalized_descent_effect(gradient, gradient)
    assert float(effect) < 0.0


def test_selected_dataset_indices_support_exact_subset(tmp_path) -> None:
    path = tmp_path / "indices.npy"
    np.save(path, np.array([8, 3, 5, 1], dtype=np.int64))

    selected = selected_dataset_indices(
        samples=3,
        dataset_size=10,
        sample_indices_path=path,
    )

    np.testing.assert_array_equal(selected, np.array([8, 3, 5]))


def test_select_prediction_target_matches_internal_guidance_formula() -> None:
    full = torch.tensor([2.0])
    base = torch.tensor([1.0])

    torch.testing.assert_close(
        select_prediction_target(
            (full, base),
            target="guided",
            guidance_scale=1.75,
        ),
        torch.tensor([2.75]),
    )
