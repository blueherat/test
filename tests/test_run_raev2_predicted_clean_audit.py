from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.run_raev2_predicted_clean_audit import (
    HeadOutputHook,
    condition_name,
    guided_clean_prediction,
    metric_effect_rows,
)


def test_guided_clean_prediction_matches_official_formula_and_interval() -> None:
    full = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])
    base = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
    time = torch.tensor([0.05, 0.4, 1.0])
    result = guided_clean_prediction(
        full,
        base,
        time,
        scale=1.78,
        interval=(0.1, 1.0),
    )
    expected = torch.tensor([[[1.0]], [[2.78]], [[3.78]]])
    torch.testing.assert_close(result, expected)


def test_guided_scale_one_recovers_full_inside_interval() -> None:
    generator = torch.Generator().manual_seed(4)
    full = torch.randn(3, 5, generator=generator)
    base = torch.randn(3, 5, generator=generator)
    result = guided_clean_prediction(
        full,
        base,
        torch.tensor([0.2, 0.4, 0.8]),
        scale=1.0,
        interval=(0.1, 1.0),
    )
    torch.testing.assert_close(result, full)


def test_head_output_hook_captures_and_consumes_one_pair() -> None:
    hook = HeadOutputHook(in_channels=2)
    full = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    base = full + 10
    hook(None, (), (full, base))
    captured_full, captured_base = hook.pop()
    assert captured_full.shape == (2, 2, 2)
    assert captured_base.shape == (2, 2, 2)
    torch.testing.assert_close(captured_full, full[:, :2])
    torch.testing.assert_close(captured_base, base[:, :2])
    with pytest.raises(RuntimeError, match="did not observe"):
        hook.pop()


def test_metric_effect_rows_separates_head_history_and_total() -> None:
    values = {
        "full_on_full": (0.60, 10.0, 5.0),
        "ig_on_full": (0.55, 8.0, 4.0),
        "full_on_ig": (0.58, 9.0, 4.5),
        "ig_on_ig": (0.52, 6.0, 3.0),
    }
    summary = pd.DataFrame(
        [
            {
                "requested_time": 0.2,
                "actual_time": 0.198,
                "condition": condition,
                "auc": metrics[0],
                "fid_real": metrics[1],
                "fid_reconstruction": metrics[2],
            }
            for condition, metrics in values.items()
        ]
    )
    effects = metric_effect_rows(summary).set_index("effect")
    total = effects.loc["on_policy_total"]
    assert np.isclose(total["auc_delta"], -0.08)
    assert np.isclose(total["auc_separability_delta"], -0.08)
    assert np.isclose(total["fid_real_delta"], -4.0)
    assert np.isclose(total["fid_reconstruction_delta"], -2.0)
    head = effects.loc["head_on_full_state"]
    assert np.isclose(head["fid_real_delta"], -2.0)
    history = effects.loc["history_under_ig_head"]
    assert np.isclose(history["fid_real_delta"], -2.0)


def test_condition_name_rejects_unknown_factor_levels() -> None:
    assert condition_name("ig", "full") == "ig_on_full"
    with pytest.raises(ValueError):
        condition_name("base", "full")
