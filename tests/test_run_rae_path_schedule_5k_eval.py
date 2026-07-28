import pandas as pd
import pytest

from experiments.run_rae_path_schedule_5k_eval import summarize


def test_5k_gate_requires_baseline_direction_candidate_improvement_and_static_proximity():
    table = pd.DataFrame(
        [
            {
                "condition": "static",
                "frechet_inception_distance": 100.0,
                "kernel_inception_distance_mean": 0.100,
            },
            {
                "condition": "annealed",
                "frechet_inception_distance": 110.0,
                "kernel_inception_distance_mean": 0.110,
            },
            {
                "condition": "floor020_p2",
                "frechet_inception_distance": 101.0,
                "kernel_inception_distance_mean": 0.101,
            },
        ]
    )
    result = summarize(table)
    assert result["gate_pass"] is True
    assert result["candidate_mean_relative_degradation_vs_static"] == pytest.approx(0.01)
