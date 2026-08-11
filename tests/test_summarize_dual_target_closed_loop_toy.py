from __future__ import annotations

import pandas as pd

from experiments.summarize_dual_target_closed_loop_toy import (
    mechanism_classification,
)


def test_mechanism_classification_separates_closed_loop_cases() -> None:
    endpoint = pd.DataFrame(
        [
            {
                "seed": 1,
                "ambient_dim": 2,
                "oracle_improves_best_shared_branch": True,
            },
            {
                "seed": 2,
                "ambient_dim": 512,
                "oracle_improves_best_shared_branch": False,
            },
        ]
    )
    teacher = pd.DataFrame(
        [
            {
                "seed": seed,
                "ambient_dim": dimension,
                "time": time_value,
                "oracle_over_best_branch": 0.7,
                "oracle_improves_best_branch": True,
            }
            for seed, dimension in ((1, 2), (2, 512))
            for time_value in (0.1, 0.5, 0.9)
        ]
    )

    result = mechanism_classification(endpoint, teacher).set_index("ambient_dim")

    assert result.loc[2, "case"] == "A_teacher_and_closed_loop_improve"
    assert result.loc[512, "case"] == "B_teacher_improves_closed_loop_worsens"
