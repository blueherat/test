from __future__ import annotations

import torch

from experiments.evaluate_raev2_common_adapter_pairing import load_adapters
from experiments.raev2_common_adapter import (
    COMMON_ADAPTER_FORMAT,
    CommonResidualAdapter,
)
from experiments.sample_raev2_common_adapter import _load_adapter


def _write_checkpoint(path) -> None:
    adapter = CommonResidualAdapter(channels=4, hidden_channels=8)
    torch.save(
        {
            "format": COMMON_ADAPTER_FORMAT,
            "adapter_config": adapter.config_dict(),
            "adapter": adapter.state_dict(),
            "adapter_ema": adapter.state_dict(),
            "common_adapter": {
                "source_sha256": "source-sha",
                "source_state_key": "model",
                "branch_update": 10,
                "objective": "lpl",
                "lpl_variant": "raw",
                "lpl_prediction_target": "full",
            },
        },
        path,
    )


def test_sampling_loader_preserves_lpl_prediction_target(tmp_path):
    checkpoint = tmp_path / "adapter.pt"
    _write_checkpoint(checkpoint)

    _, metadata = _load_adapter(
        checkpoint,
        channels=4,
        zero_hidden_channels=8,
        source_sha256="source-sha",
        source_state_key="model",
        adapter_state_key="adapter",
        device=torch.device("cpu"),
    )

    assert metadata["lpl_prediction_target"] == "full"


def test_pairing_loader_preserves_lpl_prediction_target(tmp_path):
    checkpoint = tmp_path / "adapter.pt"
    _write_checkpoint(checkpoint)

    loaded = load_adapters(
        [("full_lpl", checkpoint)],
        source_sha256="source-sha",
        source_state_key="model",
        state_key="adapter",
        device=torch.device("cpu"),
    )

    assert loaded[0][2]["lpl_prediction_target"] == "full"
