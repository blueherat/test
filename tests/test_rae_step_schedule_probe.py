import torch

from experiments.rae_step_schedule_probe import (
    candidate_time_grids,
    rollout_endpoint,
    unique_indices,
)
from experiments.rae_teacher_rollout_gap import official_time_grid


class ConstantVelocity(torch.nn.Module):
    def forward(self, state, time, y):
        del time, y
        return torch.full_like(state, 3.0)


def test_candidate_grids_share_endpoints_and_are_descending():
    official = official_time_grid()
    grids = candidate_time_grids(official)
    assert set(grids) == {
        "official_50",
        "official_numsteps_25",
        "official_numsteps_16",
        "shifted_subsample_25",
        "shifted_subsample_16",
        "uniform_actual_t_20",
        "hybrid_early5_uniformlate_20",
        "hybrid_early4_uniformlate_16",
    }
    for grid in grids.values():
        assert torch.all(grid[:-1] > grid[1:])
        torch.testing.assert_close(grid[[0, -1]], official[[0, -1]])
    assert unique_indices(16, 50).unique().numel() == 16


def test_all_grids_are_exact_for_constant_velocity():
    initial = torch.zeros((2, 1, 1, 1))
    labels = torch.zeros(2, dtype=torch.long)
    official = official_time_grid()
    expected = torch.full_like(initial, 3.0 * float(official[-1] - official[0]))
    for grid in candidate_time_grids(official).values():
        endpoint = rollout_endpoint(ConstantVelocity(), initial, labels, grid)
        torch.testing.assert_close(endpoint, expected, atol=2e-6, rtol=0)


def test_recomputed_grids_respect_nondefault_time_shift():
    shift = 3.0
    official = official_time_grid(time_shift=shift)
    grids = candidate_time_grids(official, time_shift=shift)
    torch.testing.assert_close(
        grids["official_numsteps_25"], official_time_grid(25, time_shift=shift)
    )
