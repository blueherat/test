import torch

from experiments.run_raev2_lpl_component_audit import (
    component_metrics,
    normalized_descent_effect,
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
    assert float(metrics["variance_descent_log_variance_change"]) > 0.0


def test_normalized_descent_effect_reports_descent_sign() -> None:
    gradient = torch.tensor([[3.0, 4.0]])
    effect = normalized_descent_effect(gradient, gradient)
    assert float(effect) < 0.0
