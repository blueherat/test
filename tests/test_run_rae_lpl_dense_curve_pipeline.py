from pathlib import Path

from experiments.run_rae_lpl_dense_curve_pipeline import (
    ENDPOINTS,
    branch_complete,
    branch_path,
    train_command,
)


def test_dense_curve_uses_every_500_step() -> None:
    assert ENDPOINTS == tuple(range(500, 5001, 500))


def test_branch_complete_requires_every_dense_checkpoint(tmp_path: Path) -> None:
    branch = branch_path(tmp_path, 4102, "flow")
    (branch / "checkpoints").mkdir(parents=True)
    (branch / "manifest.json").write_text("{}", encoding="utf-8")
    (branch / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    for step in ENDPOINTS:
        (branch / "checkpoints" / f"step-{step:07d}.pt").touch()

    assert branch_complete(branch)
    (branch / "checkpoints" / "step-0001000.pt").unlink()
    assert not branch_complete(branch)


def test_full_training_command_preserves_fixed_lpl_weight(tmp_path: Path) -> None:
    command = train_command(
        results=tmp_path,
        seed=4102,
        objective="full",
        lpl_weight=0.125,
        devices="0,1,2,3",
    )

    assert command[command.index("--objective") + 1] == "full"
    assert command[command.index("--lpl-weight") + 1] == "0.125"
