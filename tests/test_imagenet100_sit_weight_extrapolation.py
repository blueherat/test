from __future__ import annotations

import pytest
import torch

from experiments.imagenet100_sit_weight_extrapolation import (
    extrapolate_state_dict,
    validate_state_dict_pair,
    validate_weight_extrapolation_pair,
    velocity_extrapolation,
)
from experiments.run_imagenet100_sit_weight_extrapolation_fid1k import (
    checkpoint_metadata,
)


def _checkpoint(step: int, value: float = 1.0) -> dict:
    return {
        "protocol": "imagenet100_sit_linear_flow_v1",
        "step": step,
        "config": {
            "model_name": "SiT-S/2",
            "cfg_dropout": 0.1,
            "global_batch_size": 256,
            "seed": 0,
        },
        "data_manifest_sha256": "data",
        "official_sit": {"commit": "same"},
        "ema": {"weight": torch.tensor([value], dtype=torch.float32)},
    }


def test_weight_extrapolation_uses_strong_minus_weak_orientation() -> None:
    strong = {"weight": torch.tensor([2.0, 4.0])}
    weak = {"weight": torch.tensor([1.0, 7.0])}

    actual = extrapolate_state_dict(strong, weak, scale=1.5)

    torch.testing.assert_close(actual["weight"], torch.tensor([3.5, -0.5]))


def test_zero_scale_is_an_exact_strong_copy() -> None:
    strong = {"weight": torch.randn(3), "counter": torch.tensor([4])}
    weak = {"weight": torch.randn(3), "counter": torch.tensor([4])}

    actual = extrapolate_state_dict(strong, weak, scale=0.0)

    assert torch.equal(actual["weight"], strong["weight"])
    assert torch.equal(actual["counter"], strong["counter"])
    assert actual["weight"].data_ptr() != strong["weight"].data_ptr()


def test_nonfloating_state_must_be_identical() -> None:
    with pytest.raises(ValueError, match="non-floating state differs"):
        validate_state_dict_pair(
            {"counter": torch.tensor([1])},
            {"counter": torch.tensor([2])},
        )


def test_checkpoint_pair_allows_only_training_progress_to_differ() -> None:
    validate_weight_extrapolation_pair(
        _checkpoint(800_000, 2.0),
        _checkpoint(500_000, 1.0),
        weights="ema",
    )

    bad = _checkpoint(500_000, 1.0)
    bad["config"]["seed"] = 1
    with pytest.raises(ValueError, match="config.seed"):
        validate_weight_extrapolation_pair(
            _checkpoint(800_000, 2.0),
            bad,
            weights="ema",
        )


def test_velocity_extrapolation_matches_autoguidance_formula() -> None:
    strong = torch.tensor([2.0, 5.0])
    weak = torch.tensor([1.0, 9.0])

    actual = velocity_extrapolation(strong, weak, scale=2.0)

    torch.testing.assert_close(actual, torch.tensor([4.0, -3.0]))


def test_checkpoint_metadata_preserves_selected_weight_type(tmp_path) -> None:
    checkpoint = _checkpoint(800_000, 2.0)
    checkpoint["model"] = checkpoint.pop("ema")
    checkpoint["weight_extrapolation"] = {
        "weights": "model",
        "gamma": 0.1,
    }
    path = tmp_path / "model_weight_extrapolation.pt"
    torch.save(checkpoint, path)

    metadata = checkpoint_metadata(path)

    assert metadata["weights"] == "model"
