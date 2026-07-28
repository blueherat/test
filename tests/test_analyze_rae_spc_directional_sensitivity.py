from __future__ import annotations

import pandas as pd

from experiments.analyze_rae_spc_directional_sensitivity import (
    build_paired_sensitivity,
    summarize_exploratory,
)


def test_paired_directional_sensitivity_detects_selective_suppression() -> None:
    rows = []
    for seed in range(5):
        for condition in ("static", "spc"):
            for direction in ("guided", "control"):
                gain = 10.0 if direction == "guided" else 1.0
                if condition == "spc" and direction == "guided":
                    gain *= 0.2
                rows.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "checkpoint_step": 2000,
                        "time": 0.85,
                        "direction": direction,
                        "total_gain": gain,
                    }
                )
    paired = build_paired_sensitivity(pd.DataFrame(rows))
    summary = summarize_exploratory(paired)
    assert (paired["guided_spc_over_static"] == 0.2).all()
    assert (paired["control_spc_over_static"] == 1.0).all()
    assert summary["rows"][0]["guided_ratio_at_most_half_count"] == 5
