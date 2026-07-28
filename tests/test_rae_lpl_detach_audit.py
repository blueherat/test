from __future__ import annotations

import pandas as pd
import torch

from experiments.rae_lpl_detach_audit import (
    controlled_lpl_per_sample,
    decoder_feature_objective_per_sample,
    gradient_decomposition_metrics,
    lpl_loss_variants_per_sample,
    scale_gradient,
)
from experiments.run_rae_lpl_detach_audit import summarize
from experiments.rae_strict_lpl import decoder_outlier_mask, strict_lpl_per_sample


def test_detach_and_full_have_identical_forward_values() -> None:
    target = torch.randn(2, 4, 16, 16)
    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    losses, details = lpl_loss_variants_per_sample(
        [target], [prediction], eps=1e-6
    )
    strict, _ = strict_lpl_per_sample([target], [prediction], eps=1e-6)

    torch.testing.assert_close(
        losses["prediction_detach"], losses["prediction_full"]
    )
    torch.testing.assert_close(losses["prediction_full"], strict)
    assert details["prediction_over_target_variance_layers"].shape == (2, 1)


def test_feature_objective_selector_preserves_strict_full_gradient() -> None:
    target = torch.randn(2, 4, 8, 8)
    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    selected, _ = decoder_feature_objective_per_sample(
        "full", [target], [prediction], eps=1e-6
    )
    strict, _ = strict_lpl_per_sample([target], [prediction], eps=1e-6)
    selected_gradient = torch.autograd.grad(
        selected.sum(), prediction, retain_graph=True
    )[0]
    strict_gradient = torch.autograd.grad(strict.sum(), prediction)[0]

    torch.testing.assert_close(selected, strict)
    torch.testing.assert_close(selected_gradient, strict_gradient)


def test_feature_objective_selector_detach_changes_only_gradient() -> None:
    target = torch.randn(2, 4, 8, 8)
    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    detached, details = decoder_feature_objective_per_sample(
        "detach", [target], [prediction], eps=1e-6
    )
    full, _ = decoder_feature_objective_per_sample(
        "lpl", [target], [prediction], eps=1e-6
    )
    detached_gradient = torch.autograd.grad(
        detached.sum(), prediction, retain_graph=True
    )[0]
    full_gradient = torch.autograd.grad(full.sum(), prediction)[0]

    torch.testing.assert_close(detached, full)
    assert not torch.allclose(detached_gradient, full_gradient)
    assert details["mask_keep_fraction"].shape == (2, 1)


def test_detach_removes_only_prediction_variance_gradient() -> None:
    target = torch.tensor([[[[-1.0, 1.0], [-1.0, 1.0]]]])
    prediction = (0.5 * target).requires_grad_(True)
    losses, _ = lpl_loss_variants_per_sample(
        [target],
        [prediction],
        outlier_quantile=0.25,
        outlier_opening=1,
        outlier_closing=1,
        eps=1e-6,
    )
    detach_gradient = torch.autograd.grad(
        losses["prediction_detach"].sum(), prediction, retain_graph=True
    )[0]
    full_gradient = torch.autograd.grad(
        losses["prediction_full"].sum(), prediction
    )[0]

    torch.testing.assert_close(
        losses["prediction_detach"], losses["prediction_full"]
    )
    assert not torch.allclose(detach_gradient, full_gradient)


def test_full_minus_detach_is_exact_denominator_gradient() -> None:
    target = torch.tensor([[[[-1.5, -0.5], [0.5, 1.5]]]])
    prediction = torch.tensor(
        [[[[-1.0, -0.25], [0.75, 2.0]]]],
        requires_grad=True,
    )
    eps = 1e-6
    losses, _ = lpl_loss_variants_per_sample(
        [target],
        [prediction],
        outlier_quantile=0.25,
        outlier_opening=1,
        outlier_closing=1,
        eps=eps,
    )
    detach_gradient = torch.autograd.grad(
        losses["prediction_detach"].sum(),
        prediction,
        retain_graph=True,
    )[0]
    full_gradient = torch.autograd.grad(
        losses["prediction_full"].sum(),
        prediction,
        retain_graph=True,
    )[0]

    mask = decoder_outlier_mask(
        prediction,
        quantile=0.25,
        opening=1,
        closing=1,
    ).to(prediction.dtype)
    count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean = (prediction * mask).sum(dim=(-2, -1), keepdim=True) / count
    variance = ((prediction - mean) * mask).square().sum(
        dim=(-2, -1),
        keepdim=True,
    ) / count
    numerator = ((prediction - target).square() * mask).sum(dim=(-2, -1))
    denominator_only = (
        numerator.detach() / (variance[..., 0, 0] + eps)
    ).mean(dim=1).sum()
    denominator_gradient = torch.autograd.grad(
        denominator_only,
        prediction,
    )[0]

    torch.testing.assert_close(
        full_gradient - detach_gradient,
        denominator_gradient,
    )


def test_controlled_lpl_has_one_forward_value_and_exact_gradient_split() -> None:
    torch.manual_seed(7)
    target = torch.randn(2, 3, 8, 8)
    prediction = (target + 0.4 * torch.randn_like(target)).requires_grad_(True)
    gradients = {}
    losses = {}
    for name, scales in {
        "error": (1.0, 0.0),
        "variance": (0.0, 1.0),
        "full": (1.0, 1.0),
        "half": (0.25, 0.75),
    }.items():
        loss, _ = controlled_lpl_per_sample(
            [target],
            [prediction],
            error_gradient_scale=scales[0],
            variance_gradient_scale=scales[1],
        )
        losses[name] = loss.detach()
        gradients[name] = torch.autograd.grad(
            loss.sum(), prediction, retain_graph=True
        )[0]

    torch.testing.assert_close(losses["error"], losses["full"])
    torch.testing.assert_close(losses["variance"], losses["full"])
    torch.testing.assert_close(losses["half"], losses["full"])
    torch.testing.assert_close(
        gradients["full"],
        gradients["error"] + gradients["variance"],
        atol=2e-6,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        gradients["half"],
        0.25 * gradients["error"] + 0.75 * gradients["variance"],
        atol=2e-6,
        rtol=2e-5,
    )


def test_explicit_variance_only_matches_full_minus_detach() -> None:
    target = torch.randn(2, 4, 8, 8)
    prediction = (target + 0.3 * torch.randn_like(target)).requires_grad_(True)
    losses, _ = lpl_loss_variants_per_sample([target], [prediction])
    gradients = {
        name: torch.autograd.grad(
            losses[name].sum(), prediction, retain_graph=True
        )[0]
        for name in ("prediction_detach", "prediction_full", "variance_only")
    }

    torch.testing.assert_close(
        gradients["variance_only"],
        gradients["prediction_full"] - gradients["prediction_detach"],
        atol=2e-6,
        rtol=2e-5,
    )


def test_scale_gradient_changes_only_backward() -> None:
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    scaled = scale_gradient(value, 0.3)
    torch.testing.assert_close(scaled, value)
    gradient = torch.autograd.grad(scaled.sum(), value)[0]
    torch.testing.assert_close(gradient, torch.full_like(value, 0.3))


def test_gradient_decomposition_reports_variance_increasing_stats_step() -> None:
    raw = torch.tensor([[1.0, 0.0]])
    detach = torch.tensor([[2.0, 0.0]])
    stats = torch.tensor([[-1.0, 0.0]])
    full = detach + stats
    log_variance = torch.tensor([[1.0, 0.0]])
    metrics = gradient_decomposition_metrics(raw, detach, full, log_variance)

    torch.testing.assert_close(
        metrics["stats_over_full_gradient_rms"], torch.ones(1)
    )
    torch.testing.assert_close(
        metrics["stats_descent_log_variance_cosine"], torch.ones(1)
    )
    torch.testing.assert_close(
        metrics["full_detach_gradient_cosine"], torch.ones(1)
    )


def test_summary_keeps_noise_ratio_as_group_key() -> None:
    rows = pd.DataFrame(
        {
            "checkpoint": ["source", "source"],
            "state_key": ["model", "model"],
            "checkpoint_step": [0, 0],
            "noise_to_signal_ratio": [1.0, 1.0],
            "sample_index": [0, 1],
            "raw_loss": [1.0, 3.0],
        }
    )

    summary = summarize(rows)

    assert summary.to_dict(orient="records") == [
        {
            "checkpoint": "source",
            "state_key": "model",
            "checkpoint_step": 0,
            "noise_to_signal_ratio": 1.0,
            "raw_loss": 2.0,
        }
    ]
