from __future__ import annotations

from pathlib import Path

from experiments.run_rae_path_crossover_train import (
    BRANCHES,
    branch_experiment_name,
    training_command,
)


def test_crossover_is_complete_two_by_two_design() -> None:
    cells = {(branch.early_path, branch.late_path) for branch in BRANCHES}
    assert cells == {
        ("floor", "floor"),
        ("floor", "static"),
        ("static", "floor"),
        ("static", "static"),
    }


def test_crossover_command_is_exact_single_gpu_fork() -> None:
    branch = next(item for item in BRANCHES if item.name == "floor_to_static")
    command, name = training_command(
        branch, results=Path("/tmp/crossover"), endpoint=5000
    )
    joined = " ".join(command)
    assert "--nproc_per_node=1" in command
    assert "--fork-full-state" in command
    assert "--path-mode static" in joined
    assert "--path-floor 0.0" in joined
    assert str(branch.source_checkpoint) in command
    assert name == branch_experiment_name(branch, 5000)


def test_static_origin_isolates_loader_rng_but_floor_origin_replays_restart() -> None:
    for branch in BRANCHES:
        command, _ = training_command(
            branch, results=Path("/tmp/crossover"), endpoint=5000
        )
        assert ("--isolate-loader-rng" in command) == (branch.early_path == "static")
