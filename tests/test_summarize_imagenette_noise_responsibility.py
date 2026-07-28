import numpy as np
import pandas as pd

from experiments.summarize_imagenette_noise_responsibility import (
    _ridge_predict,
    capacity_permutation_pvalue,
    paired_region_statistics,
    total_shuffle_curve_features,
)


def test_region_statistics_average_times_within_sample_before_ci():
    rows = []
    for sample in range(10):
        for time in (0.4, 0.6, 0.8, 0.9):
            rows.append(
                {
                    "latent_dim": 16,
                    "seed": 0,
                    "sample_index": sample,
                    "time": time,
                    "delta_shuffle": 1.0 + sample / 100,
                    "delta_null": 2.0,
                    "delta_within_class": 0.5,
                }
            )
    result = paired_region_statistics(pd.DataFrame(rows))
    assert set(result.region) == {"mid_noise", "high_noise"}
    assert set(result["count"]) == {10}
    np.testing.assert_allclose(result.delta_shuffle_mean, 1.045)
    assert (result.delta_shuffle_ci95_low > 1.0).all()


def test_capacity_permutation_detects_consistent_decreasing_shape():
    features = pd.DataFrame(
        [
            {"seed": seed, "latent_dim": latent_dim, "high_noise_fraction": value}
            for seed in range(3)
            for latent_dim, value in ((16, 0.70), (64, 0.50), (256, 0.30))
        ]
    )
    slope, pvalue = capacity_permutation_pvalue(features)
    assert slope < 0.0
    assert pvalue < 0.05


def test_curve_feature_table_includes_frequency_fraction():
    rows = []
    for latent_dim in (16, 64, 256):
        rows.append(
            {
                "latent_dim": latent_dim,
                "seed": 0,
                "source": "total",
                "band": "all",
                "metric": "delta_shuffle",
                "low_noise_fraction": 0.2,
                "mid_noise_fraction": 0.3,
                "high_noise_fraction": 0.5,
                "positive_region_sum": 1.0,
            }
        )
        for band, value in (("low", 3.0), ("mid", 2.0), ("high", 1.0)):
            rows.append(
                {
                    "latent_dim": latent_dim,
                    "seed": 0,
                    "source": "frequency",
                    "band": band,
                    "metric": "delta_shuffle",
                    "low_noise_fraction": 0.2,
                    "mid_noise_fraction": 0.3,
                    "high_noise_fraction": 0.5,
                    "positive_region_sum": value,
                }
            )
    features = total_shuffle_curve_features(pd.DataFrame(rows))
    np.testing.assert_allclose(features.high_frequency_fraction, 1 / 6)


def test_ridge_prediction_uses_training_standardization_only():
    train_x = np.array([[0.0], [1.0], [2.0], [3.0]])
    train_y = 2.0 * train_x[:, 0] + 1.0
    prediction = _ridge_predict(train_x, train_y, np.array([[4.0]]), alpha=0.0)
    np.testing.assert_allclose(prediction, [9.0], atol=1e-8)
