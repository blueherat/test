from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_rae_predictability_gain import (
    gain_explanation_per_seed,
    matched_pair_ratios,
)


def test_gain_explanation_recovers_time_dependent_predictability() -> None:
    names = [f"basis_{index}" for index in range(8)]
    variance = np.exp(np.linspace(-1.0, 1.0, len(names)))
    predictability = np.array([0.1, 0.8, 0.3, 0.7, 0.2, 0.9, 0.4, 0.6])
    metrics = pd.DataFrame(
        {
            "basis": names,
            "basis_family": "test",
            "val_final_variance_per_dimension": variance,
            "val_r2": predictability,
        }
    )
    rows = []
    for seed in (1, 2):
        for time_value in (0.1, 0.9):
            beta = 0.0 if time_value == 0.1 else 2.0
            for name, value, score in zip(names, variance, predictability):
                rows.append(
                    {
                        "seed": seed,
                        "time": time_value,
                        "basis": name,
                        "basis_family": "test",
                        "total_gain": value * np.exp(beta * score),
                    }
                )
    result = gain_explanation_per_seed(metrics, pd.DataFrame(rows))
    low = result[result["time"] == 0.1]
    high = result[result["time"] == 0.9]
    assert float(low["variance_only_r2"].min()) > 0.999
    assert float(low["predictability_beta"].abs().max()) < 1e-8
    assert float(high["combined_r2"].min()) > 0.999
    assert float(high["predictability_beta"].min()) > 0.4


def test_matched_pair_ratio_is_computed_per_seed_and_time() -> None:
    metrics = pd.DataFrame(
        {
            "basis": ["high", "low"],
            "basis_family": ["a", "b"],
            "val_final_variance_per_dimension": [1.01, 1.0],
            "val_r2": [0.9, 0.4],
        }
    )
    sensitivity = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "time": [0.9, 0.9, 0.9, 0.9],
            "basis": ["high", "low", "high", "low"],
            "total_gain": [4.0, 2.0, 6.0, 2.0],
        }
    )
    result = matched_pair_ratios(
        metrics, sensitivity, pairs=(("high", "low"),)
    )
    assert list(result["gain_ratio"]) == [2.0, 3.0]
