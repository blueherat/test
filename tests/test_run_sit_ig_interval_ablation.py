import numpy as np
import torch

from experiments.run_sit_ig_interval_ablation import (
    IntervalCondition,
    brownian_noise,
    linear_path_sde_drift,
    paper_and_missing_conditions,
    sde_time_grid,
    simulate_conditions,
)


class ConstantDualHead(torch.nn.Module):
    def forward(self, state, times, labels):
        del times, labels
        return torch.full_like(state, 0.2), torch.full_like(state, -0.1)


def test_ablation_contains_all_contiguous_boundary_intervals():
    conditions = paper_and_missing_conditions()
    intervals = {
        (condition.low, condition.high)
        for condition in conditions
        if condition.scale == 2.3
    }
    assert intervals == {
        (0.0, 0.3),
        (0.0, 0.7),
        (0.0, 1.0),
        (0.3, 0.7),
        (0.3, 1.0),
        (0.7, 1.0),
    }


def test_linear_path_sde_drift_matches_official_equations():
    state = torch.tensor([[[[0.4]]]], dtype=torch.float64)
    velocity = torch.tensor([[[[-0.2]]]], dtype=torch.float64)
    time = 0.7
    score = (-(1.0 - time) * velocity - state) / time
    expected = velocity - time * score
    torch.testing.assert_close(linear_path_sde_drift(state, velocity, time), expected)


def test_brownian_noise_is_sample_and_step_deterministic():
    ids = np.array([2, 9])
    first = brownian_noise(ids, (2, 2), seed=7, step=3)
    second = brownian_noise(ids, (2, 2), seed=7, step=3)
    different = brownian_noise(ids, (2, 2), seed=7, step=4)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_simulation_shares_noise_and_changes_only_active_ig_branch():
    conditions = (
        IntervalCondition("baseline", 1.0, 0.0, 1.0, "test"),
        IntervalCondition("also_baseline", 1.0, 0.7, 1.0, "test"),
        IntervalCondition("guided", 2.0, 0.7, 1.0, "test"),
    )
    noise = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    labels = torch.tensor([1, 2])
    result = simulate_conditions(
        model=ConstantDualHead(),
        noise=noise,
        labels=labels,
        sample_ids=np.array([0, 1]),
        grid=sde_time_grid(2),
        conditions=conditions,
        brownian_seed_value=11,
    )
    torch.testing.assert_close(result[0], result[1])
    assert not torch.equal(result[0], result[2])
