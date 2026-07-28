from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.analyze_rae_path_crossover import generation_gate, paired_median_interval


def test_paired_interval_detects_consistent_improvement() -> None:
    result = paired_median_interval(
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([2.0, 3.0, 4.0, 5.0]),
        draws=1000,
    )
    assert result["median_difference"] == -1.0
    assert result["ci95_high"] < 0.0
    assert result["left_lower_fraction"] == 1.0


def test_generation_gate_requires_all_three_bidirectional_checks() -> None:
    table = pd.DataFrame(
        {
            "condition": [
                "floor_to_floor",
                "floor_to_static",
                "static_to_static",
                "static_to_floor",
            ],
            "frechet_inception_distance": [150.0, 130.0, 145.0, 160.0],
            "kernel_inception_distance_mean": [0.13, 0.10, 0.14, 0.15],
        }
    )
    result = generation_gate(table)
    assert result["pass"]
