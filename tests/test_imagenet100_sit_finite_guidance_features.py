from __future__ import annotations

import torch

from experiments.analyze_imagenet100_sit_finite_guidance_features import (
    _gamma_key,
    aggregate_feature_metrics,
    response_metrics,
)


def test_gamma_key_is_sign_preserving() -> None:
    assert _gamma_key(-0.01) == "gm0p01"
    assert _gamma_key(0.0) == "gp0"
    assert _gamma_key(0.75) == "gp0p75"


def test_response_metrics_detect_exact_prediction() -> None:
    actual = torch.randn(8, 16)
    metrics = response_metrics(actual, actual.clone())
    assert torch.allclose(metrics["relative_residual"], torch.zeros(8))
    assert torch.allclose(metrics["cosine"], torch.ones(8), atol=1e-6)


def test_linearity_feature_aggregation_recovers_exact_tangent() -> None:
    baseline = torch.randn(6, 12)
    tangent = torch.randn(6, 12)
    delta = 0.01
    features = {
        "closed_gm0p01": baseline - delta * tangent,
        "closed_gp0": baseline,
        "closed_gp0p01": baseline + delta * tangent,
        "closed_gp0p1": baseline + 0.1 * tangent,
        "closed_gp1": baseline + tangent,
    }
    manifest = {
        "study": "linearity",
        "central_delta": delta,
        "gammas": [0.1, 1.0],
    }
    rows, summary = aggregate_feature_metrics(features, manifest)
    assert len(rows) == 2
    assert rows[-1]["feature_relative_residual_mean"] < 1e-5
    assert rows[-1]["feature_cosine_mean"] > 0.99999
    assert summary["feature_linearity_at_gamma_one"] == rows[-1]


def test_feedback_feature_aggregation_detects_frozen_closed_gap() -> None:
    baseline = torch.zeros(5, 9)
    direction = torch.randn(5, 9)
    features = {
        "baseline": baseline,
        "frozen_gp0p5": 0.5 * direction,
        "closed_gp0p5": direction,
        "frozen_gp1": direction,
        "closed_gp1": 2.0 * direction,
    }
    manifest = {"study": "feedback", "gammas": [0.5, 1.0]}
    rows, summary = aggregate_feature_metrics(features, manifest)
    assert len(rows) == 2
    assert abs(rows[-1]["feature_frozen_over_closed_rms_mean"] - 0.5) < 1e-6
    assert rows[-1]["feature_response_cosine_mean"] > 0.99999
    assert summary["feature_feedback_at_gamma_one"] == rows[-1]
