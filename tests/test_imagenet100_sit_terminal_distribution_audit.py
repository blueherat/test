from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.imagenet100_sit_terminal_distribution import (
    AuditCondition,
    closed_terms,
    factorized_terms,
    gaussian_frechet_distance,
    linear_rbf_mmd2,
    parse_condition,
    sample_mean_product,
    sample_mean_square,
    sliced_wasserstein_distance,
    validate_conditions,
)
from experiments.run_imagenet100_sit_terminal_distribution_audit import (
    _integrate_conditions,
    _integrate_partitioned_conditions,
)
from experiments.analyze_imagenet100_sit_terminal_distribution_audit import (
    _c2st_auc,
    _c2st_split_null,
)
from experiments.summarize_imagenet100_sit_terminal_distribution_audit import (
    _add_null_calibration,
    _add_quality_improvements,
    _aggregate,
    _paired_feature_rows,
)


def test_condition_parser_and_validation() -> None:
    condition = parse_condition("candidate:factorized:1.5:1.35")
    assert condition == AuditCondition("candidate", "factorized", 1.5, 1.35)
    validate_conditions((condition,))
    with pytest.raises(ValueError, match="unique"):
        validate_conditions((condition, condition))
    with pytest.raises(ValueError, match="closed"):
        AuditCondition("bad", "closed", 1.0, 1.2)


def test_factorized_decomposition_matches_direct_formula() -> None:
    anchor_baseline = torch.tensor([[1.0, 2.0]])
    anchor_current = torch.tensor([[3.0, 5.0]])
    other_baseline = torch.tensor([[0.5, 1.5]])
    terms = factorized_terms(
        anchor_baseline,
        anchor_current,
        other_baseline,
        gamma=1.5,
        response_scale=1.35,
    )
    assert torch.allclose(terms["drift"], terms["direct_drift"])
    assert torch.allclose(
        terms["drift"],
        anchor_baseline
        + 1.35 * (anchor_current - anchor_baseline)
        + 1.5 * (anchor_baseline - other_baseline),
    )

    replay = factorized_terms(
        anchor_baseline,
        anchor_current,
        other_baseline,
        gamma=1.0,
        response_scale=0.0,
    )
    frozen = factorized_terms(
        anchor_baseline,
        anchor_current,
        other_baseline,
        gamma=1.0,
        response_scale=1.0,
    )
    gap = anchor_baseline - other_baseline
    assert torch.allclose(replay["drift"], anchor_baseline + gap)
    assert torch.allclose(frozen["drift"], anchor_current + gap)


def test_control_action_decomposition_is_exact() -> None:
    generator = torch.Generator().manual_seed(7)
    forcing = torch.randn(5, 3, 2, generator=generator)
    response = torch.randn(5, 3, 2, generator=generator)
    control = forcing + response
    left = sample_mean_square(control)
    right = (
        sample_mean_square(forcing)
        + sample_mean_square(response)
        + 2.0 * sample_mean_product(forcing, response)
    )
    assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)


def test_closed_terms_use_current_gap() -> None:
    anchor = torch.tensor([[2.0, 3.0]])
    weak = torch.tensor([[0.5, 1.0]])
    terms = closed_terms(anchor, weak, gamma=2.0)
    assert torch.allclose(terms["drift"], anchor + 2.0 * (anchor - weak))
    assert torch.count_nonzero(terms["response_control"]) == 0


class _LinearFields:
    def anchor(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del labels
        return 0.2 * state + time_value

    def other(
        self,
        state: torch.Tensor,
        time_value: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        del labels
        return -0.1 * state - 0.5 * time_value


def test_grouped_ode_matches_each_condition_integrated_alone() -> None:
    fields = _LinearFields()
    noise = torch.tensor([[[[0.2]]], [[[0.7]]]], dtype=torch.float32)
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.linspace(0.0, 1.0, 7)
    conditions = (
        AuditCondition("factorized", "factorized", 1.2, 1.3),
        AuditCondition("closed", "closed", 0.8, 1.0),
    )
    grouped, _ = _integrate_conditions(
        fields,
        noise,
        labels,
        conditions,
        times,
        atol=1e-8,
        rtol=1e-7,
    )
    for grouped_index, condition in enumerate(conditions, start=1):
        individual, _ = _integrate_conditions(
            fields,
            noise,
            labels,
            (condition,),
            times,
            atol=1e-8,
            rtol=1e-7,
        )
        assert torch.allclose(grouped[:, 0], individual[:, 0], atol=2e-5, rtol=2e-5)
        assert torch.allclose(
            grouped[:, grouped_index],
            individual[:, 1],
            atol=2e-5,
            rtol=2e-5,
        )


def test_partitioned_integration_preserves_condition_order() -> None:
    fields = _LinearFields()
    noise = torch.tensor([[[[0.2]]], [[[0.7]]]], dtype=torch.float32)
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.linspace(0.0, 1.0, 7)
    conditions = (
        AuditCondition("closed_first", "closed", 0.8, 1.0),
        AuditCondition("factorized", "factorized", 1.2, 1.3),
        AuditCondition("closed_last", "closed", 1.1, 1.0),
    )
    partitioned, _ = _integrate_partitioned_conditions(
        fields,
        noise,
        labels,
        conditions,
        times,
        atol=1e-8,
        rtol=1e-7,
        closed_atol=1e-8,
        closed_rtol=1e-7,
    )
    for partitioned_index, condition in enumerate(conditions, start=1):
        individual, _ = _integrate_conditions(
            fields,
            noise,
            labels,
            (condition,),
            times,
            atol=1e-8,
            rtol=1e-7,
        )
        assert torch.allclose(
            partitioned[:, partitioned_index],
            individual[:, 1],
            atol=2e-5,
            rtol=2e-5,
        )


def test_distribution_metrics_have_expected_toy_behavior() -> None:
    generator = np.random.default_rng(11)
    first = generator.normal(size=(64, 4))
    shifted = first + 0.5
    assert gaussian_frechet_distance(first, first) == pytest.approx(0.0, abs=1e-9)
    assert gaussian_frechet_distance(first, shifted) > 0.9
    assert sliced_wasserstein_distance(first, first) == pytest.approx(0.0)
    assert sliced_wasserstein_distance(first, shifted) == pytest.approx(0.5)
    assert linear_rbf_mmd2(first, first, bandwidth=1.0) == pytest.approx(0.0)


def test_cross_seed_summary_and_split_null_calibration() -> None:
    quality = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "condition": ["baseline", "guided", "baseline", "guided"],
            "fid": [10.0, 8.0, 12.0, 9.0],
            "sfid": [5.0, 4.0, 6.0, 5.0],
            "inception_score": [2.0, 2.5, 1.8, 2.4],
        }
    )
    improved = _add_quality_improvements(quality)
    guided = improved[improved["condition"] == "guided"]
    assert guided["fid_improvement_vs_baseline"].tolist() == [2.0, 3.0]
    aggregate = _aggregate(improved, ["condition"])
    baseline = aggregate[aggregate["condition"] == "baseline"].iloc[0]
    assert baseline["fid_mean"] == pytest.approx(11.0)
    assert baseline["seed_count"] == 2

    pairwise = pd.DataFrame(
        {
            "seed": [0, 0, 0, 1, 1, 1],
            "condition_a": ["a", "b", "a", "a", "b", "a"],
            "condition_b": ["a", "b", "b", "a", "b", "b"],
            "comparison": [
                "within_condition_split",
                "within_condition_split",
                "cross_condition",
                "within_condition_split",
                "within_condition_split",
                "cross_condition",
            ],
            "swd_mean": [1.0, 3.0, 5.0, 2.0, 4.0, 8.0],
        }
    )
    calibrated = _add_null_calibration(pairwise, has_time=False)
    cross = calibrated[calibrated["comparison"] == "cross_condition"]
    assert cross["swd_mean_excess_over_split_null"].tolist() == pytest.approx(
        [3.0, 5.0]
    )


def test_c2st_split_null_calibrates_finite_sample_auc() -> None:
    generator = np.random.default_rng(19)
    features = generator.normal(size=(256, 8))
    null = _c2st_split_null(features, reps=4, rng=generator)
    shifted = _c2st_auc(
        features,
        features + 1.0,
        reps=4,
        rng=generator,
        max_groups=128,
    )
    assert 0.5 <= null["c2st_auc_mean"] < 0.7
    assert shifted["c2st_auc_mean"] > 0.9


def test_paired_feature_rows_preserve_sample_pairing(tmp_path) -> None:
    seed_root = tmp_path / "seed0"
    activation_root = seed_root / "adm_activations"
    activation_root.mkdir(parents=True)
    (seed_root / "manifest.json").write_text(
        '{"branches": ["baseline", "guided"]}',
        encoding="utf-8",
    )
    baseline = np.zeros((4, 3), dtype=np.float32)
    guided = np.ones((4, 3), dtype=np.float32)
    np.savez(activation_root / "baseline.npz", pool_3=baseline)
    np.savez(activation_root / "guided.npz", pool_3=guided)
    rows = _paired_feature_rows(tmp_path, (0,))
    assert len(rows) == 1
    assert rows.iloc[0]["feature_paired_rms_mean"] == pytest.approx(1.0)
