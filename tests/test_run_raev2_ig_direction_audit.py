from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from experiments.run_raev2_ig_direction_audit import (
    nearest_solver_step,
    summarize_direction_rows,
    summarize_scale_rows,
)


def test_nearest_solver_step_reports_h_over_t() -> None:
    grid = torch.tensor([1.0, 0.6, 0.2, 0.0])
    result = nearest_solver_step(grid, 0.58, t_eps=1e-5)
    assert result["nearest_solver_step"] == 1
    assert np.isclose(result["nearest_step_size"], 0.4)
    assert np.isclose(result["nearest_h_over_t"], 2.0 / 3.0)


def test_direction_summary_uses_ratio_of_expectations() -> None:
    raw = pd.DataFrame(
        {
            "state_key": ["ema", "ema"],
            "time": [0.5, 0.5],
            "sample_id": [0, 1],
            "alignment": [2.0, 4.0],
            "a_term": [-2.0, -4.0],
            "direction_mean_square": [1.0, 3.0],
            "positive_alignment": [True, True],
            "gamma_star": [2.0, 4.0 / 3.0],
            "alignment_cosine": [0.8, 0.9],
            "full_mse": [4.0, 4.0],
            "base_mse": [9.0, 9.0],
            "oracle_relative_gain": [0.5, 0.6],
            "direction_rms": [1.0, np.sqrt(3.0)],
            "residual_rms": [2.0, 2.0],
            "nearest_solver_step": [3, 3],
            "nearest_solver_time": [0.5, 0.5],
            "nearest_h_over_t": [0.2, 0.2],
            "direction_velocity_rms": [2.0, 2.0 * np.sqrt(3.0)],
            "direction_step_rms": [0.2, 0.2 * np.sqrt(3.0)],
        }
    )
    summary = summarize_direction_rows(raw, bootstrap_repeats=20, seed=7)
    assert np.isclose(summary.gamma_population.iloc[0], 1.5)
    assert np.isclose(summary.a_mean.iloc[0], -3.0)
    assert summary.positive_alignment_fraction.iloc[0] == 1.0


def test_scale_summary_preserves_paired_gain_sign() -> None:
    raw = pd.DataFrame(
        {
            "state_key": ["ema", "ema", "ema", "ema"],
            "time": [0.5] * 4,
            "scale": [1.0, 1.0, 1.5, 1.5],
            "sample_id": [0, 1, 0, 1],
            "mse": [2.0, 4.0, 1.0, 2.0],
            "gain_over_full": [0.0, 0.0, 0.5, 0.5],
        }
    )
    summary = summarize_scale_rows(raw, bootstrap_repeats=20, seed=11)
    guided = summary[summary.scale.eq(1.5)].iloc[0]
    assert guided.gain_over_full_mean == 0.5
    assert guided.positive_gain_fraction == 1.0
