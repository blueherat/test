from pathlib import Path

from experiments.run_raev2_ig_endpoint_pipeline import (
    SEEDS,
    Stage,
    build_stages,
    stage_succeeded,
)


def test_stage_success_requires_zero_exit_and_every_output(tmp_path: Path) -> None:
    stage = Stage("test", (), tmp_path, ("a.csv", "b.json"))
    (tmp_path / "exit_code").write_text("0\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("a\n", encoding="utf-8")
    assert not stage_succeeded(stage)
    (tmp_path / "b.json").write_text("{}\n", encoding="utf-8")
    assert stage_succeeded(stage)
    (tmp_path / "exit_code").write_text("1\n", encoding="utf-8")
    assert not stage_succeeded(stage)


def test_pipeline_has_replicated_endpoint_and_predicted_clean_stages() -> None:
    stages = build_stages()
    names = [stage.name for stage in stages]
    assert names[:3] == [
        "endpoint_seed_20260801_external",
        "endpoint_seed_20260802",
        "summarize_endpoint_seeds",
    ]
    assert "predicted_clean_seed_20260802" in names
    assert "summarize_predicted_clean_seeds" in names
    for seed in SEEDS:
        assert f"kid_seed_{seed}" in names
        assert f"precision_recall_seed_{seed}" in names


def test_endpoint_command_uses_true_endpoint_and_hard_control() -> None:
    stage = next(stage for stage in build_stages() if stage.name == "endpoint_seed_20260802")
    command = list(stage.command)
    requested_times = [
        command[index + 1] for index, value in enumerate(command) if value == "--time"
    ]
    assert requested_times == ["0", "1"]
    assert command[command.index("--seed") + 1] == "20260802"


def test_predicted_clean_command_uses_matching_reference_seed() -> None:
    stage = next(
        stage for stage in build_stages() if stage.name == "predicted_clean_seed_20260802"
    )
    command = list(stage.command)
    reference = command[command.index("--decoded-reference-run") + 1]
    assert "seed20260802" in reference
    assert command[command.index("--seed") + 1] == "20260802"
