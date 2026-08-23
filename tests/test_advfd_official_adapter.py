from types import SimpleNamespace

import pytest

from experiments.advfd_cleanroom.run_official_advfd_packed import (
    build_schedule_horizon_adapter,
)


def test_schedule_horizon_is_visible_only_during_lr_update() -> None:
    observed = []

    def adjust_learning_rate(_optimizer, step, args):
        observed.append((step, args.total_steps))
        return step / args.total_steps

    adapted = build_schedule_horizon_adapter(adjust_learning_rate, 125_000)
    args = SimpleNamespace(total_steps=25_000)

    value = adapted(None, 10_000, args)

    assert value == pytest.approx(0.08)
    assert observed == [(10_000, 125_000)]
    assert args.total_steps == 25_000


def test_schedule_horizon_rejects_a_horizon_shorter_than_the_run() -> None:
    adapted = build_schedule_horizon_adapter(lambda *_: None, 10_000)
    args = SimpleNamespace(total_steps=25_000)

    with pytest.raises(ValueError, match="cannot be shorter"):
        adapted(None, 0, args)


def test_schedule_horizon_restores_total_steps_after_failure() -> None:
    def fail(_optimizer, _step, _args):
        raise RuntimeError("failure inside scheduler")

    adapted = build_schedule_horizon_adapter(fail, 125_000)
    args = SimpleNamespace(total_steps=25_000)

    with pytest.raises(RuntimeError, match="inside scheduler"):
        adapted(None, 0, args)
    assert args.total_steps == 25_000
