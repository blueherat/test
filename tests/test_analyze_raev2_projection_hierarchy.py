from __future__ import annotations

import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analyze_raev2_projection_hierarchy import (
    derive_row,
    positive_second_moment_root,
)


def test_second_moment_root_recovers_known_scale() -> None:
    base_power = 2.0
    base_gap_cross = 0.0
    gap_power = 0.5
    target_power = base_power + 4.0 * gap_power
    assert math.isclose(
        positive_second_moment_root(
            base_power, base_gap_cross, gap_power, target_power
        ),
        2.0,
    )


def test_nested_projection_statistics_are_reconstructed() -> None:
    # B and D are orthogonal; Y = B + D + R with R orthogonal to both.
    row = derive_row(
        step=0,
        time=0.8,
        band="all",
        means={
            "A": 5.0,  # ||B + D||^2 = 4 + 1
            "C": 1.0,  # <B + D, D> = 1
            "Q": 1.0,
            "T": 7.0,
            "E": 0.0,
            "mse_full": 2.0,
            "mse_base": 3.0,
        },
    )
    assert math.isclose(float(row["base_power"]), 4.0)
    assert math.isclose(float(row["base_gap_cross"]), 0.0)
    assert math.isclose(float(row["residual_gap_cross"]), 0.0)
    assert math.isclose(float(row["residual_full_cross"]), 0.0)
    assert math.isclose(float(row["mse_optimal_scale_from_base"]), 1.0)
    assert math.isclose(
        float(row["target_second_moment_scale_from_base"]), math.sqrt(3.0)
    )
