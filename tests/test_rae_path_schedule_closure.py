import pandas as pd

from experiments.run_rae_path_schedule_closure import (
    evaluate_closure_prediction,
    evaluate_crossover_closure,
)


def test_closure_prediction_accepts_floor_near_annealed() -> None:
    table = pd.DataFrame(
        {
            "source": ["clean_test", "static", "annealed", "floor020_p2"],
            "cycle_relative_rms_median": [0.4, 1.0, 1.2, 1.19],
            "local_decoder_sensitivity_median": [1.3, 1.5, 1.8, 1.79],
        }
    )
    result = evaluate_closure_prediction(table)
    assert result["pass"]
    assert result["predictions"]["p2_floor_matches_annealed"]
    assert 0.75 <= result["details"]["fid_floor_position"] <= 1.25


def test_crossover_closure_detects_late_path_effect() -> None:
    values = {
        "floor_to_floor": (1.30, 1.91),
        "floor_to_static": (1.23, 1.87),
        "static_to_static": (1.21, 1.86),
        "static_to_floor": (1.29, 1.90),
    }
    table = pd.DataFrame(
        [
            {
                "source": source,
                "cycle_relative_rms_median": cycle,
                "local_decoder_sensitivity_median": sensitivity,
            }
            for source, (cycle, sensitivity) in values.items()
        ]
    )
    result = evaluate_crossover_closure(table)
    assert result["directions"]["floor_to_static_improves_both"]
    assert result["directions"]["static_to_floor_worsens_both"]
