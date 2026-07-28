from __future__ import annotations

import pandas as pd
import torch

from experiments.evaluate_rae_path_crossover import exactly_equal, summarize_generation
from experiments.evaluate_rae_layerwise_path_generation import sample_folder_name


def test_exactly_equal_checks_nested_tensor_state() -> None:
    left = {"state": [torch.tensor([1.0, 2.0])], "step": 3}
    right = {"state": [torch.tensor([1.0, 2.0])], "step": 3}
    assert exactly_equal(left, right)
    right["state"][0][0] = 4.0
    assert not exactly_equal(left, right)


def test_generation_summary_recovers_symmetric_late_path_effect() -> None:
    rows = []
    values = {
        "floor_to_floor": (120.0, 0.12),
        "floor_to_static": (100.0, 0.10),
        "static_to_static": (90.0, 0.09),
        "static_to_floor": (110.0, 0.11),
    }
    for condition, (fid, kid) in values.items():
        rows.append(
            {
                "condition": condition,
                "frechet_inception_distance": fid,
                "kernel_inception_distance_mean": kid,
            }
        )
    summary = summarize_generation(pd.DataFrame(rows))
    assert summary["directions"]["floor_to_static_improves_both"]
    assert summary["directions"]["static_to_floor_worsens_both"]
    fid = summary["late_path_effects"]["frechet_inception_distance"]
    assert fid["late_floor_with_early_floor"] == 20.0
    assert fid["late_floor_with_early_static"] == 20.0
    assert fid["difference_in_differences"] == 0.0


def test_online_sample_folder_does_not_collide_with_ema() -> None:
    ema = sample_folder_name(1000, 5000, 50, "ema")
    model = sample_folder_name(1000, 5000, 50, "model")
    assert ema != model
    assert model.endswith("_model")
