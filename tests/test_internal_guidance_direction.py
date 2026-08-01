import math

import torch

from experiments.internal_guidance_direction import (
    direction_gram_metrics,
    direction_metrics,
    euler_ig_scale_sweep_rollout,
    euler_ig_rollout,
    first_step_impulse_policy,
    fixed_scale_policy,
    guided_prediction,
    scale_sweep_metrics,
)


def test_direction_metrics_recovers_known_optimal_scale() -> None:
    base = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    full = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    direction = full - base
    target = full + 0.5 * direction
    metrics = direction_metrics(full, base, target)

    torch.testing.assert_close(metrics["gamma_star"], torch.full((2,), 0.5))
    torch.testing.assert_close(metrics["scale_star"], torch.full((2,), 1.5))
    torch.testing.assert_close(metrics["oracle_mse"], torch.zeros(2))
    assert bool(metrics["positive_alignment"].all())


def test_worse_base_does_not_imply_a_useful_direction() -> None:
    full = torch.tensor([[1.0, 0.0]])
    base = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.75, 2.0]])
    metrics = direction_metrics(full, base, target)

    assert metrics["base_mse"].item() > metrics["full_mse"].item()
    assert metrics["alignment"].item() < 0
    assert metrics["gamma_star"].item() < 0


def test_scale_sweep_is_paired_to_full_scale_one() -> None:
    base = torch.zeros((1, 2))
    full = torch.ones((1, 2))
    target = torch.full((1, 2), 1.5)
    sweep = scale_sweep_metrics(full, base, target, (0.0, 1.0, 1.5, 2.0))

    torch.testing.assert_close(sweep["mse"][:, 0], torch.tensor([2.25, 0.25, 0.0, 0.25]))
    torch.testing.assert_close(guided_prediction(full, base, 1.0), full)


def test_gram_metrics_detects_redundant_and_orthogonal_directions() -> None:
    d1 = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
    d2 = 3.0 * d1
    redundant = direction_gram_metrics((d1, d2))
    assert math.isclose(float(redundant["effective_rank"]), 1.0, abs_tol=1e-5)

    d3 = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    orthogonal = direction_gram_metrics((d1, d3))
    assert math.isclose(float(orthogonal["effective_rank"]), 2.0, abs_tol=1e-5)


class TwoHeadConstantVelocity(torch.nn.Module):
    def forward(self, state, time, labels):
        del time, labels
        full = torch.ones_like(state)
        base = torch.zeros_like(state)
        return full, base, None


def test_euler_rollout_respects_fixed_and_impulse_scales() -> None:
    model = TwoHeadConstantVelocity()
    initial = torch.zeros((2, 1, 1, 1))
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.tensor([1.0, 0.5, 0.0])

    full = euler_ig_rollout(
        model, initial, labels, times, scale_policy=fixed_scale_policy(1.0)
    )
    doubled = euler_ig_rollout(
        model, initial, labels, times, scale_policy=fixed_scale_policy(2.0)
    )
    impulse = euler_ig_rollout(
        model, initial, labels, times, scale_policy=first_step_impulse_policy(2.0)
    )

    torch.testing.assert_close(full.endpoint, torch.full_like(initial, -1.0))
    torch.testing.assert_close(doubled.endpoint, torch.full_like(initial, -2.0))
    torch.testing.assert_close(impulse.endpoint, torch.full_like(initial, -1.5))


def test_batched_scale_sweep_matches_individual_rollouts() -> None:
    model = TwoHeadConstantVelocity()
    initial = torch.zeros((2, 1, 1, 1))
    labels = torch.zeros(2, dtype=torch.long)
    times = torch.tensor([1.0, 0.5, 0.0])
    scales = (1.0, 2.0)

    persistent = euler_ig_scale_sweep_rollout(
        model, initial, labels, times, scales, mode="persistent"
    )
    impulse = euler_ig_scale_sweep_rollout(
        model, initial, labels, times, scales, mode="first_step_impulse"
    )

    torch.testing.assert_close(persistent[0], torch.full_like(initial, -1.0))
    torch.testing.assert_close(persistent[1], torch.full_like(initial, -2.0))
    torch.testing.assert_close(impulse[0], torch.full_like(initial, -1.0))
    torch.testing.assert_close(impulse[1], torch.full_like(initial, -1.5))
