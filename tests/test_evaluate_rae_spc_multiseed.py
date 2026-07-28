from __future__ import annotations

from pathlib import Path

from experiments.evaluate_rae_layerwise_path_generation import branches
from experiments.evaluate_rae_spc_multiseed import (
    branch_name,
    planned_branches,
    sample_command,
)


def test_planned_evaluation_has_two_conditions_per_seed() -> None:
    planned = planned_branches((11, 13), 5000, 2000)
    assert len(planned) == 4
    assert {(seed, condition) for seed, condition, _ in planned} == {
        (11, "static"),
        (11, "spc"),
        (13, "static"),
        (13, "spc"),
    }


def test_branch_names_match_training_names() -> None:
    assert branch_name(11, "static", 5000, 2000) == "seed11_static_s0_to5000"
    assert branch_name(11, "spc", 5000, 2000) == (
        "seed11_spc_floor020_p2_rank16_switch2000_s0_to5000"
    )


def test_generation_branch_discovery_accepts_schedule_names(tmp_path: Path) -> None:
    branch = tmp_path / "seed11_spc_floor020_p2_rank16_switch2000_s0_to5000"
    branch.mkdir()
    (branch / "manifest.json").write_text("{}", encoding="utf-8")
    assert branches(tmp_path, branch.name) == [branch]
    assert branches(tmp_path) == [branch]


def test_sampling_command_fixes_online_protocol() -> None:
    command = sample_command(
        Path("/tmp/results"),
        "seed11_static_s0_to5000",
        endpoint=5000,
        sample_count=5000,
        steps=50,
        device=2,
        weight_source="model",
        per_process_batch=8,
    )
    joined = " ".join(command)
    assert "--sample-count 5000" in joined
    assert "--steps 50" in joined
    assert "--devices 2" in joined
    assert "--weight-source model" in joined
