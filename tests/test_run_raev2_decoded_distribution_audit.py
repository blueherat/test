from __future__ import annotations

import numpy as np

from experiments.run_raev2_decoded_distribution_audit import (
    feature_probe_scores,
    feature_statistics,
    fid_between_statistics,
    fit_feature_probe,
    time_suffix,
)
from experiments.run_raev2_distribution_auc import paired_auc


def test_time_suffix_is_stable() -> None:
    assert time_suffix(0.0) == "t0p000000"
    assert time_suffix(0.2) == "t0p200000"


def test_identical_features_give_chance_auc() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(20, 8)).astype(np.float32)
    train_mask = np.zeros(20, dtype=bool)
    train_mask[:12] = True
    weight, intercept, _ = fit_feature_probe(
        features, features.copy(), train_mask, ridge_ratio=1e-4
    )
    negative = feature_probe_scores(features[~train_mask], weight, intercept)
    positive = feature_probe_scores(features[~train_mask], weight, intercept)
    assert paired_auc(negative, positive) == 0.5


def test_shifted_features_are_linearly_separable_on_heldout_data() -> None:
    rng = np.random.default_rng(11)
    negative = rng.normal(size=(80, 6)).astype(np.float32)
    positive = negative.copy()
    positive[:, 0] += 2.0
    train_mask = np.zeros(80, dtype=bool)
    train_mask[:60] = True
    weight, intercept, _ = fit_feature_probe(
        negative, positive, train_mask, ridge_ratio=1e-4
    )
    neg_scores = feature_probe_scores(negative[~train_mask], weight, intercept)
    pos_scores = feature_probe_scores(positive[~train_mask], weight, intercept)
    assert paired_auc(neg_scores, pos_scores) > 0.8


def test_fid_is_zero_for_identical_feature_statistics() -> None:
    rng = np.random.default_rng(13)
    features = rng.normal(size=(32, 5)).astype(np.float32)
    stats = feature_statistics(features)
    assert fid_between_statistics(stats, stats) < 1e-8
