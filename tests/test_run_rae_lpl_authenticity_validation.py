from argparse import Namespace
from pathlib import Path

import pytest

from experiments.run_rae_lpl_authenticity_validation import (
    PRIOR_SPECS,
    PriorSpec,
    build_command,
    prepare_source_branch,
)


def _args(**overrides) -> Namespace:
    values = {
        "mode": "train",
        "objective": "full",
        "data_path": Path("/data/train"),
        "results_dir": Path("/data/results"),
        "seed": 4101,
        "endpoint": 5000,
        "calibration_batches": 256,
        "calibration_mode": "mean_contribution",
        "lpl_weight": 1e-3,
        "skip_checkpoint_save": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_primary_calibration_uses_main_paper_loss_contribution() -> None:
    command = build_command(
        _args(mode="calibrate", lpl_weight=None),
        PriorSpec(Path("/config.yaml"), Path("/source.pt")),
        "calibration",
    )

    assert command[command.index("--calibration-mode") + 1] == "mean_contribution"
    assert command[command.index("--calibration-target-lpl-over-flow") + 1] == "0.25"
    assert command[command.index("--calibration-target-variance-ratio") + 1] == "0.1"
    assert command[command.index("--calibration-batches") + 1] == "256"


def test_full_training_requires_precommitted_positive_weight() -> None:
    with pytest.raises(ValueError, match="positive --lpl-weight"):
        build_command(
            _args(lpl_weight=None),
            PriorSpec(Path("/config.yaml"), Path("/source.pt")),
            "full",
        )


def test_flow_branch_forces_zero_lpl_weight() -> None:
    command = build_command(
        _args(objective="flow", lpl_weight=999.0),
        PriorSpec(Path("/config.yaml"), Path("/source.pt")),
        "flow",
    )

    assert command[command.index("--lpl-weight") + 1] == "0"


def test_official_source_branch_keeps_zero_training_provenance(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "source.pt"
    config.write_text("stage_1: {}\n", encoding="utf-8")
    checkpoint.write_bytes(b"official")

    branch = prepare_source_branch(
        results_dir=tmp_path / "results",
        prior_name="model",
        spec=PriorSpec(config, checkpoint),
    )

    manifest = (branch / "manifest.json").read_text(encoding="utf-8")
    assert '"objective": "official_source"' in manifest
    assert '"training_updates": 0' in manifest
    assert (branch / "config.yaml").read_text(encoding="utf-8") == "stage_1: {}\n"


def test_cross_tokenizer_official_specs_are_explicit() -> None:
    mae = PRIOR_SPECS["mae_dit_xl_ep80"]
    siglip = PRIOR_SPECS["siglip2_dit_xl_ep80"]

    assert mae.config.name == "rae_strict_lpl_dit_xl_mae.yaml"
    assert mae.checkpoint.parts[-5:] == (
        "MAE",
        "b16",
        "ImageNet256",
        "DiT-XL-ep80",
        "stage2_model.pt",
    )
    assert siglip.config.name == "rae_strict_lpl_dit_xl_siglip2.yaml"
    assert siglip.checkpoint.parts[-5:] == (
        "SigLIP2",
        "b16",
        "ImageNet256",
        "DiT-XL-ep80",
        "stage2_model.pt",
    )
