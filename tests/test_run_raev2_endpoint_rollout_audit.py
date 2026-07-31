import torch

from experiments.run_raev2_endpoint_rollout_audit import (
    select_endpoint_prediction,
    shifted_time_grid,
)


def test_shifted_time_grid_has_requested_endpoints_and_is_monotonic() -> None:
    grid = shifted_time_grid(0.75, num_steps=8, shift=8.0)
    torch.testing.assert_close(grid[0], torch.tensor(0.75))
    torch.testing.assert_close(grid[-1], torch.tensor(0.0))
    assert torch.all(grid[:-1] > grid[1:])


def test_shifted_time_grid_matches_official_full_grid() -> None:
    actual = shifted_time_grid(1.0, num_steps=10, shift=8.0)
    raw = torch.linspace(1.0, 0.0, 11)
    expected = 8.0 * raw / (1.0 + 7.0 * raw)
    torch.testing.assert_close(actual, expected)


def test_select_endpoint_prediction_respects_modes_and_ig_interval() -> None:
    full = torch.tensor([[3.0], [3.0], [3.0], [3.0]])
    base = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    mode_ids = torch.tensor([0, 1, 2, 2])
    time = torch.tensor([0.5, 0.5, 0.5, 0.05])
    actual = select_endpoint_prediction(
        full,
        base,
        mode_ids=mode_ids,
        time=time,
        ig_scale=1.5,
        ig_interval=(0.1, 1.0),
    )
    torch.testing.assert_close(
        actual,
        torch.tensor([[3.0], [1.0], [4.0], [3.0]]),
    )
