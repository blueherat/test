from __future__ import annotations

import pandas as pd

from experiments.analyze_rae_layerwise_path_generation import analyze


def test_generation_gate_requires_three_consistent_seeds_and_controls() -> None:
    rows = []
    for seed in (1, 2, 3):
        for condition, fid, kid in (
            ("static", 100.0, 0.100),
            ("annealed", 90.0, 0.090),
            ("reverse", 105.0, 0.105),
            ("random", 102.0, 0.102),
        ):
            rows.append(
                {
                    "seed": seed,
                    "path_mode": "annealed" if condition == "random" else condition,
                    "subspace_kind": (
                        "random_energy_matched" if condition == "random" else "middle_guided"
                    ),
                    "frechet_inception_distance": fid,
                    "kernel_inception_distance_mean": kid,
                }
            )
    result = analyze(pd.DataFrame(rows))
    assert result["status"] == "complete"
    assert result["gate_pass_both_metrics"] is True
    assert result["futility_stop"] is False


def test_one_opposite_seed_makes_three_of_three_gate_impossible() -> None:
    table = pd.DataFrame(
        [
            {
                "seed": 1,
                "path_mode": condition,
                "subspace_kind": "middle_guided",
                "frechet_inception_distance": value,
                "kernel_inception_distance_mean": value / 1000.0,
            }
            for condition, value in (
                ("static", 100.0),
                ("annealed", 110.0),
                ("reverse", 120.0),
            )
        ]
        + [
            {
                "seed": 1,
                "path_mode": "annealed",
                "subspace_kind": "random_energy_matched",
                "frechet_inception_distance": 105.0,
                "kernel_inception_distance_mean": 0.105,
            }
        ]
    )
    result = analyze(table)
    assert result["status"] == "stopped_futility"
    assert result["futility_stop"] is True
