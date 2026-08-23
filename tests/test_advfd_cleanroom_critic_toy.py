import torch

from experiments.advfd_cleanroom.run_critic_generalization_toy import (
    ToyConfig,
    make_datasets,
    run_condition,
)


def test_null_and_shift_datasets_have_expected_shapes() -> None:
    config = ToyConfig(train_samples=16, heldout_samples=32)
    for regime in ("matched", "shift"):
        datasets = make_datasets(config, seed=7, regime=regime)
        assert datasets["real_train"].shape == (16, 2)
        assert datasets["fake_train"].shape == (16, 2)
        assert datasets["real_heldout"].shape == (32, 2)
        assert datasets["fake_heldout"].shape == (32, 2)


def test_all_calibrations_complete_a_short_critic_update() -> None:
    config = ToyConfig(
        train_samples=32,
        heldout_samples=64,
        hidden_dim=16,
        depth=2,
        feature_dim=4,
        steps=2,
        eval_every=1,
    )
    for mode in ("none", "real", "pooled"):
        rows = run_condition(
            config,
            seed=19,
            regime="matched",
            mode=mode,
            device=torch.device("cpu"),
        )
        assert [row["step"] for row in rows] == [0, 1, 2]
        assert all(row["train_fd"] >= 0.0 for row in rows)
