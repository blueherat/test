from __future__ import annotations

from pathlib import Path

from experiments.run_rae_spc_multiseed_train import Job, jobs, training_command


def test_multiseed_design_is_paired() -> None:
    planned = jobs((11, 13, 17))
    assert {(job.seed, job.condition) for job in planned} == {
        (seed, condition)
        for seed in (11, 13, 17)
        for condition in ("static", "spc")
    }


def test_spc_is_a_continuous_in_process_switch() -> None:
    command = training_command(
        Job(1201, "spc"), results=Path("/tmp/results"), endpoint=5000, switch_step=2000
    )
    joined = " ".join(command)
    assert "--ckpt" not in command
    assert "--path-mode annealed" in joined
    assert "--path-switch-step 2000" in joined
    assert "--path-mode-after-switch static" in joined
    assert "--ema-reset-step 2000" in joined
    assert "--cache-order-seed 1201" in joined


def test_static_control_matches_randomness_and_ema_reset() -> None:
    control = training_command(
        Job(1201, "static"), results=Path("/tmp/results"), endpoint=5000, switch_step=2000
    )
    joined = " ".join(control)
    assert "--path-mode static" in joined
    assert "--path-switch-step" not in control
    assert "--ema-reset-step 2000" in joined
    assert "--cache-order-seed 1201" in joined
