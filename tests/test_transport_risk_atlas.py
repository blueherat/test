from __future__ import annotations

import pandas as pd
import pytest

from experiments.transport_risk_atlas import (
    build_prospective_atlas,
    evaluate_prospective_score,
    summarize_by_basis,
)


def _frames():
    study_rows = []
    gradient_rows = []
    settings = {
        "dct": (0.20, 0.20, 4.0, 1.50),
        "pca": (-0.30, 0.15, 4.2, 1.80),
        "random": (0.95, 0.98, 4.1, 0.98),
    }
    for seed in range(3):
        for basis, (cosine, descent, pressure, fid_ratio) in settings.items():
            study_rows.append(
                {
                    "basis": basis,
                    "seed": seed,
                    "teacher_ratio_all": 0.95,
                    "rollout_feature_fid_ratio": fid_ratio + 0.01 * seed,
                    "weight_mean": 1.0,
                    "weight_min": 0.2,
                    "weight_max": 2.0,
                }
            )
            gradient_rows.append(
                {
                    "basis": basis,
                    "seed": seed,
                    "checkpoint_variant": "baseline",
                    "parameter_group": "all",
                    "coarse_detail_cosine_unweighted": cosine,
                    "allocation_multiplier": pressure,
                    "coarse_descent_ratio": descent,
                }
            )
            gradient_rows.append(
                {
                    "basis": basis,
                    "seed": seed,
                    "checkpoint_variant": "weighted",
                    "parameter_group": "all",
                    "coarse_detail_cosine_unweighted": -1.0,
                    "allocation_multiplier": 1000.0,
                    "coarse_descent_ratio": -1000.0,
                }
            )
    return pd.DataFrame(study_rows), pd.DataFrame(gradient_rows)


def test_score_uses_only_baseline_gradient_rows_and_separates_isospectral_bases():
    study, gradients = _frames()
    atlas = build_prospective_atlas(study, gradients, dataset="toy")
    means = atlas.groupby("basis")["optimization_risk_score"].mean()

    assert means["pca"] > means["dct"] > means["random"]
    assert atlas["isospectral_signature"].nunique() == 1
    assert atlas["coarse_descent_ratio"].min() > 0


def test_endpoint_targets_cannot_change_prospective_score():
    study, gradients = _frames()
    first = build_prospective_atlas(study, gradients, dataset="toy")
    study["rollout_feature_fid_ratio"] = list(
        reversed(study["rollout_feature_fid_ratio"].tolist())
    )
    second = build_prospective_atlas(study, gradients, dataset="toy")

    keys = ["basis", "seed"]
    first = first.sort_values(keys).reset_index(drop=True)
    second = second.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_series_equal(
        first["optimization_risk_score"],
        second["optimization_risk_score"],
    )


def test_evaluation_and_summary_report_expected_signal():
    study, gradients = _frames()
    atlas = build_prospective_atlas(study, gradients, dataset="toy")
    evaluation = evaluate_prospective_score(atlas)
    summary = summarize_by_basis(atlas)

    assert evaluation["spearman"] > 0.8
    assert evaluation["roc_auc"] == pytest.approx(1.0)
    assert evaluation["between_basis_spearman"] > 0.8
    assert evaluation["structured_vs_random_pair_accuracy"] == pytest.approx(1.0)
    assert evaluation["leakage_audit_passed"] is True
    assert summary["seeds"].eq(3).all()


def test_duplicate_baseline_gradient_keys_are_rejected():
    study, gradients = _frames()
    gradients = pd.concat([gradients, gradients.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate keys"):
        build_prospective_atlas(study, gradients, dataset="toy")
