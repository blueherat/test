import pandas as pd

from experiments.summarize_rae_raev2_lpl_time_mechanism import (
    PANELS,
    metric_uncertainty,
    time_region,
)


def test_time_region_uses_declared_low_mid_high_boundaries() -> None:
    assert time_region(0.25) == "low_t_le_0.5"
    assert time_region(0.5) == "low_t_le_0.5"
    assert time_region(0.5001) == "mid_0.5_to_0.75"
    assert time_region(0.75) == "mid_0.5_to_0.75"
    assert time_region(0.8) == "high_t_gt_0.75"


def test_metric_uncertainty_reports_mean_sem_and_count() -> None:
    rows = []
    for value in (1.0, 3.0):
        row = {
            "system": "rae",
            "prediction_target": "single",
            "time": 0.25,
        }
        row.update({column: value for column, _ in PANELS})
        rows.append(row)

    result = metric_uncertainty(
        pd.DataFrame(rows),
        ["system", "prediction_target", "time"],
    )

    assert int(result.loc[0, "sample_count"]) == 2
    assert float(result.loc[0, "latent_relative_error_rms_mean"]) == 2.0
    assert float(result.loc[0, "latent_relative_error_rms_sem"]) == 1.0
