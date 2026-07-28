import pandas as pd
import torch

from experiments.mnist_spectral_rollout_toy import shifted_uniform
from experiments.rae_spectral_direction_loss import DCTDirectionLoss
from experiments.small_image_basis_transport import OrthogonalDirectionLoss
from experiments.small_image_time_sequence_audit import (
    compare_time_streams,
    replay_time_draws,
    summarize_band_weight_exposure,
    summarize_time_draws,
)


def test_replay_time_draws_matches_training_draw_order() -> None:
    seed = 7
    generator = torch.Generator().manual_seed(seed + 101)
    expected = []
    for _ in range(3):
        torch.randint(11, (4,), generator=generator)
        torch.randn((4, 1, 2, 2), generator=generator)
        expected.append(
            shifted_uniform(4, 1.0, device=torch.device("cpu"), generator=generator)
        )
    actual = replay_time_draws(
        seed=seed,
        train_size=11,
        batch_size=4,
        steps=3,
        spatial_shape=(1, 2, 2),
        time_shift=1.0,
        device="cpu",
    )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0.0, atol=0.0)


def test_time_summaries_and_pairwise_comparison_are_consistent() -> None:
    streams = {
        1: torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        2: torch.tensor([[0.4, 0.3], [0.2, 0.1]]),
    }
    summary, histogram, per_step = summarize_time_draws(
        streams, histogram_bins=4, training_windows=2
    )
    comparison = compare_time_streams(streams, histogram)
    assert len(summary) == 6
    assert len(histogram) == 8
    assert len(per_step) == 4
    assert comparison.loc[0, "wasserstein_1"] == 0.0
    assert comparison.loc[0, "ks_distance"] == 0.0
    assert comparison.loc[0, "histogram_tv"] == 0.0
    assert comparison.loc[0, "mean_abs_step_mean_difference"] > 0.0


def test_band_weight_exposure_has_one_row_per_seed_window_and_band() -> None:
    streams = {1: torch.tensor([[0.1, 0.2], [0.7, 0.9]])}
    analyzer = DCTDirectionLoss(2, [1.0, 2.0], gamma=0.5)
    result = summarize_band_weight_exposure(
        streams, analyzer, training_windows=2, device="cpu"
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 6
    assert set(result["window"]) == {"all", "window_01", "window_02"}
    assert set(result["band"]) == {0, 1}


def test_component_weights_are_aggregated_to_declared_bands() -> None:
    analyzer = OrthogonalDirectionLoss(
        torch.eye(4),
        torch.tensor([1.0, 1.5, 2.0, 3.0]),
        torch.tensor([0, 0, 1, 1]),
        gamma=0.5,
    )
    result = summarize_band_weight_exposure(
        {1: torch.tensor([[0.2, 0.8]])},
        analyzer,
        training_windows=1,
        device="cpu",
    )
    assert len(result) == 4
    assert set(result["band"]) == {0, 1}
