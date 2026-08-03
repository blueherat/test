from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.summarize_raev2_ig_direction_audits import (
    summarize_directions,
    summarize_scales,
)


def test_cross_seed_direction_summary_tracks_sign_consistency() -> None:
    frame = pd.DataFrame(
        {
            "state_key": ["ema", "ema"],
            "time": [0.5, 0.5],
            "gamma_population": [-0.1, -0.2],
            "positive_alignment_fraction": [0.1, 0.2],
            "full_mse_mean": [1.0, 1.2],
            "base_mse_mean": [2.0, 2.4],
            "base_over_full_mse": [2.0, 2.0],
            "oracle_relative_gain_mean": [0.1, 0.2],
            "nearest_solver_step": [5, 5],
            "nearest_solver_time": [0.5, 0.5],
            "nearest_h_over_t": [0.1, 0.1],
        }
    )
    summary = summarize_directions(frame)
    assert summary.negative_gamma_seeds.iloc[0] == 2
    assert np.isclose(summary.gamma_mean.iloc[0], -0.15)


def test_cross_seed_scale_summary_tracks_negative_gain() -> None:
    frame = pd.DataFrame(
        {
            "state_key": ["ema", "ema"],
            "time": [0.5, 0.5],
            "scale": [1.78, 1.78],
            "gain_over_full_mean": [-0.3, -0.4],
            "positive_gain_fraction": [0.0, 0.0],
        }
    )
    summary = summarize_scales(frame)
    assert summary.negative_gain_seeds.iloc[0] == 2
    assert np.isclose(summary.gain_over_full_mean.iloc[0], -0.35)
