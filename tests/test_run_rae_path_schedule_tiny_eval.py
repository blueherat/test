import pandas as pd

from experiments.run_rae_path_schedule_tiny_eval import summarize


def _row(condition: str, fid: float, kid: float, risk: float | None):
    return {
        "condition": condition,
        "frechet_inception_distance": fid,
        "kernel_inception_distance_mean": kid,
        "offline_total_risk_ratio": risk,
    }


def test_summary_applies_preregistered_candidate_gate():
    table = pd.DataFrame(
        [
            _row("static", 10.0, 0.010, None),
            _row("annealed", 14.0, 0.014, 1.0),
            _row("floor005_p1", 10.5, 0.0105, 0.69),
            _row("floor015_rat05", 11.0, 0.011, 0.70),
            _row("floor030_p2", 13.0, 0.013, 0.72),
            _row("floor020_p2", 15.0, 0.015, 0.75),
        ]
    )
    result = summarize(table)
    assert result["both_metric_improvement_count"] == 3
    assert result["best_fid_condition"] == "static"
    assert result["predictions"]["p2_at_least_two_candidates_improve_both"] is True
    assert result["predictions"]["p3_static_best_or_close"] is True
