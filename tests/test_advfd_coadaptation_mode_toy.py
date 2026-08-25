import torch

from experiments.advfd_cleanroom.run_advfd_coadaptation_mode_toy import (
    GameConfig,
    RingGenerator,
    mode_metrics,
    parse_conditions,
    ring_centers,
    run_game,
)


def test_mode_metrics_distinguish_complete_and_collapsed_samples() -> None:
    config = GameConfig(modes=8, heldout_samples=80)
    centers = ring_centers(config, torch.device("cpu"))
    complete = centers.repeat_interleave(10, dim=0)
    collapsed = centers[:1].repeat(80, 1)
    assert mode_metrics(complete, config)["mode_coverage_1pct"] == 8
    assert mode_metrics(collapsed, config)["mode_coverage_1pct"] == 1
    assert mode_metrics(collapsed, config)["mode_mass_tv"] > 0.8


def test_generator_initialization_is_mode_complete() -> None:
    config = GameConfig(modes=8)
    generator = RingGenerator(config)
    assigned = torch.cdist(generator.centers.detach(), ring_centers(config, torch.device("cpu"))).argmin(dim=1)
    assert assigned.unique().numel() == config.modes


def test_short_static_and_adaptive_games_are_finite() -> None:
    config = GameConfig(
        modes=4,
        train_batch=24,
        heldout_samples=48,
        steps=2,
        eval_every=1,
        critic_hidden=16,
        critic_depth=2,
        critic_features=4,
        static_features=4,
    )
    for name, component, static_weight, adaptive_weight in parse_conditions(
        "static,full_w0.5,mean_w0.5"
    ):
        rows, snapshots = run_game(
            config,
            seed=9,
            condition=name,
            component=component,
            static_weight=static_weight,
            adaptive_weight=adaptive_weight,
            device=torch.device("cpu"),
        )
        assert [row["step"] for row in rows] == [0, 1, 2]
        assert set(snapshots) == {0, 1, 2}
        assert all(torch.isfinite(torch.tensor(row["target_nll"])) for row in rows)
