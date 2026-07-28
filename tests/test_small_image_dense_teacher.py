import pandas as pd

from experiments.small_image_dense_teacher import summarize_dense


def test_dense_summary_uses_ratio_of_window_means():
    rows = []
    for time, baseline, weighted in ((0.2, 2.0, 1.0), (0.4, 1.0, 0.8), (0.6, 3.0, 2.4), (0.8, 2.0, 3.0)):
        for variant, field, drift in (
            ("baseline", baseline, 2.0),
            ("weighted", weighted, 3.0),
        ):
            rows.append(
                {
                    "basis": "dct",
                    "seed": 0,
                    "time": time,
                    "variant": variant,
                    "field_mse": field,
                    "component_drift_nrmse": drift,
                }
            )
    restart = pd.DataFrame(
        [
            {
                "basis": "dct",
                "seed": 0,
                "start_context": "teacher_restart",
                "schedule": "middle",
                "metric": "feature_fid",
                "ratio": 1.4,
            }
        ]
    )
    result = summarize_dense(pd.DataFrame(rows), restart).iloc[0]

    assert result["middle_field_mse_ratio_of_means"] == 0.8
    assert result["middle_field_improved_fraction"] == 1.0
    assert result["middle_mean_component_drift_ratio"] == 1.5
    assert result["teacher_restart_middle_fid_ratio"] == 1.4
