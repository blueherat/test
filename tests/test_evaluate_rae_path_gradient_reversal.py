import pandas as pd

from experiments.evaluate_rae_path_gradient_reversal import paired_mean_bootstrap


def test_paired_bootstrap_preserves_matched_batch_difference() -> None:
    rows = []
    for batch in range(8):
        for condition, value in (
            ("static", 1.1 + 0.01 * batch),
            ("floor020_p2", 0.9 + 0.01 * batch),
        ):
            rows.append(
                {
                    "condition": condition,
                    "checkpoint_step": 5000,
                    "time": 0.1,
                    "parameter_group": "last_block",
                    "split": "calibration" if batch < 4 else "test",
                    "batch_index": batch,
                    "semantic_descent_ratio": value,
                }
            )
    result = paired_mean_bootstrap(
        pd.DataFrame(rows),
        step=5000,
        time=0.1,
        parameter_group="last_block",
        other="floor020_p2",
        seed=3,
        repetitions=1000,
    )
    assert abs(result["mean"] - 0.2) < 1e-12
    assert result["ci_lower"] > 0.0
    assert result["static_win_fraction"] == 1.0
