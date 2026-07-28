from dataclasses import replace

import torch

from experiments.mnist_spectral_rollout_toy import (
    MNISTToyConfig,
    train_paired_velocity_fields,
)
from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.small_image_stream_factorial import train_paired_mixed_streams
from experiments.small_image_time_order_study import (
    TimeOrderStudyConfig,
    build_time_schedule,
    train_paired_with_time_schedule,
)


def _assert_models_equal(first: dict, second: dict) -> None:
    for variant in first:
        first_state = first[variant].state_dict()
        second_state = second[variant].state_dict()
        assert first_state.keys() == second_state.keys()
        for key in first_state:
            torch.testing.assert_close(
                first_state[key], second_state[key], rtol=0.0, atol=0.0
            )


def _configs() -> tuple[TimeOrderStudyConfig, MNISTToyConfig]:
    study = TimeOrderStudyConfig(
        devices=("cpu",),
        train_size=8,
        test_size=4,
        batch_size=2,
        steps=3,
        width=4,
        depth=1,
        time_shift=1.0,
        save=False,
    )
    toy = MNISTToyConfig(
        train_size=8,
        test_size=4,
        batch_size=2,
        steps=3,
        width=4,
        depth=1,
        learning_rate=2e-4,
        seed=4,
        device="cpu",
        save=False,
    )
    return study, toy


def test_iid_seed4_schedule_exactly_replays_original_training() -> None:
    study, toy = _configs()
    clean = torch.randn(8, 1, 28, 28, generator=torch.Generator().manual_seed(11))
    analyzer = DCTDirectionLoss(28, torch.ones(8), gamma=0.5)
    schedule = build_time_schedule("iid_seed4", study, device="cpu")
    original, _ = train_paired_velocity_fields(
        clean, toy, analyzer, init_seed=4, stream_seed=4
    )
    replayed = train_paired_with_time_schedule(
        clean,
        toy,
        analyzer,
        schedule,
        init_seed=4,
        batch_noise_seed=4,
    )
    _assert_models_equal(original, replayed)


def test_iid_seed3_schedule_exactly_replays_crossed_time_stream() -> None:
    study, toy = _configs()
    clean = torch.randn(8, 1, 28, 28, generator=torch.Generator().manual_seed(12))
    analyzer = DCTDirectionLoss(28, torch.ones(8), gamma=0.5)
    schedule = build_time_schedule("iid_seed3", study, device="cpu")
    crossed = train_paired_mixed_streams(
        clean,
        toy,
        analyzer,
        init_seed=4,
        batch_seed=4,
        noise_seed=4,
        time_seed=3,
    )
    replayed = train_paired_with_time_schedule(
        clean,
        toy,
        analyzer,
        schedule,
        init_seed=4,
        batch_noise_seed=4,
    )
    _assert_models_equal(crossed, replayed)


def test_step_permutation_preserves_exact_time_multiset() -> None:
    study, _ = _configs()
    original = build_time_schedule("iid_seed3", study, device="cpu")
    permuted = build_time_schedule("step_permuted_seed3", study, device="cpu")
    torch.testing.assert_close(
        original.flatten().sort().values,
        permuted.flatten().sort().values,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(original, permuted)


def test_arbitrary_step_permutations_preserve_one_exact_multiset() -> None:
    study, _ = _configs()
    first = build_time_schedule(
        "step_permuted_seed3_perm17", study, device="cpu"
    )
    second = build_time_schedule(
        "step_permuted_seed3_perm18", study, device="cpu"
    )
    torch.testing.assert_close(
        first.flatten().sort().values,
        second.flatten().sort().values,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(first, second)


def test_arbitrary_iid_seed_schedule_is_supported() -> None:
    study, _ = _configs()
    schedule = build_time_schedule("iid_seed17", study, device="cpu")
    assert schedule.shape == (study.steps, study.batch_size)


def test_stratified_schedule_has_one_draw_per_batch_stratum() -> None:
    study, _ = _configs()
    study = replace(study, batch_size=4, steps=5)
    schedule = build_time_schedule("stratified_seed3", study, device="cpu")
    strata = torch.floor(schedule * study.batch_size).long().sort(dim=1).values
    expected = torch.arange(study.batch_size).expand(study.steps, -1)
    torch.testing.assert_close(strata, expected, rtol=0.0, atol=0.0)


def test_arbitrary_stratified_seed_schedule_is_supported() -> None:
    study, _ = _configs()
    study = replace(study, batch_size=4, steps=2)
    first = build_time_schedule("stratified_seed17", study, device="cpu")
    second = build_time_schedule("stratified_seed18", study, device="cpu")
    assert first.shape == second.shape == (2, 4)
    assert not torch.equal(first, second)
