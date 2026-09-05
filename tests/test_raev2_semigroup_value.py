from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_semigroup_value import (  # noqa: E402
    RAEv2NormalizedOUValue,
    clean_gap_to_ou_score_gap,
    clean_prediction_to_ou_score,
    noise_time_from_ou_time,
    normalized_hjb_running_cost,
    normalized_hjb_target,
    ou_potential_gradient_to_clean_correction,
    ou_relative_retention,
    ou_to_state,
    rae_ou_coefficients,
    semigroup_value_guided_clean,
    state_to_ou,
)
from experiments.train_raev2_semigroup_value import (  # noqa: E402
    balanced_class_labels,
    build_hjb_target,
    sample_curriculum_batch,
    update_ema,
)


class ConstantDualClean(nn.Module):
    def forward(self, state, time, *, context, attn_mask=None):
        del time, context, attn_mask
        return torch.ones_like(state), torch.zeros_like(state)


def test_balanced_class_labels_cover_every_class() -> None:
    labels = balanced_class_labels(12, 5)
    torch.testing.assert_close(
        torch.bincount(labels, minlength=5), torch.tensor([3, 3, 2, 2, 2])
    )
    with pytest.raises(ValueError):
        balanced_class_labels(4, 5)


def test_ou_coefficients_are_normalized_and_invert_time() -> None:
    times = torch.tensor([0.0, 0.1, 0.5, 0.8, 0.99], dtype=torch.float64)
    _, signal, noise, semigroup = rae_ou_coefficients(times)
    torch.testing.assert_close(signal.square() + noise.square(), torch.ones_like(times))
    torch.testing.assert_close(noise_time_from_ou_time(semigroup), times)


def test_state_coordinate_round_trip() -> None:
    state = torch.randn(4, 3, 2, 2)
    times = torch.tensor([0.1, 0.3, 0.6, 0.9])
    torch.testing.assert_close(
        ou_to_state(state_to_ou(state, times), times), state
    )


def test_clean_gap_score_mapping_matches_two_score_subtraction() -> None:
    state = torch.randn(3, 2, 2, 2)
    full = torch.randn_like(state)
    base = torch.randn_like(state)
    times = torch.tensor([0.2, 0.5, 0.8])
    actual = clean_prediction_to_ou_score(
        full, ou_state=state, noise_time=times
    ) - clean_prediction_to_ou_score(base, ou_state=state, noise_time=times)
    expected = clean_gap_to_ou_score_gap(full - base, noise_time=times)
    torch.testing.assert_close(actual, expected)


def test_score_gradient_to_clean_correction_inverts_tweedie_factor() -> None:
    gradient = torch.randn(3, 4)
    times = torch.tensor([0.2, 0.5, 0.8])
    correction = ou_potential_gradient_to_clean_correction(
        gradient, noise_time=times
    )
    _, signal, noise, _ = rae_ou_coefficients(times)
    torch.testing.assert_close(
        correction, gradient * (noise.square() / signal)[:, None]
    )


def test_guided_clean_has_no_free_correction_scale() -> None:
    full = torch.tensor([[3.0, 5.0]])
    base = torch.tensor([[1.0, 2.0]])
    gradient = torch.tensor([[0.1, -0.2]])
    time = torch.tensor([0.5])
    actual, correction = semigroup_value_guided_clean(
        full,
        base,
        gradient,
        noise_time=time,
        beta=1.78,
    )
    expected_correction = ou_potential_gradient_to_clean_correction(
        2.0 * gradient, noise_time=time
    )
    torch.testing.assert_close(correction, expected_correction)
    torch.testing.assert_close(actual, base + 1.78 * (full - base) + correction)


def test_normalized_hjb_preserves_extensive_cost_and_quadratic_term() -> None:
    gap = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]])
    beta = 1.5
    cost = normalized_hjb_running_cost(gap, beta=beta)
    torch.testing.assert_close(
        cost,
        beta * (beta - 1.0) * gap.square().flatten(1).mean(1),
    )

    particles = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    gradient = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) / 10.0
    step = torch.tensor([0.1, 0.2])
    actual = normalized_hjb_target(
        particles,
        gradient,
        running_cost_per_dimension=cost,
        semigroup_step=step,
        ambient_dimension=2,
    )
    expected = particles.mean(0) + step * (
        2.0 * gradient.square().sum(1) + cost
    )
    torch.testing.assert_close(actual, expected)


def test_value_network_enforces_switch_and_noise_gradient_boundaries() -> None:
    model = RAEv2NormalizedOUValue(8, 5, width=16, depth=2)
    with torch.no_grad():
        model.spatial_output.bias.fill_(3.0)
        model.baseline_output[-1].bias.fill_(2.0)
    state = torch.randn(2, 8, 4, 4, requires_grad=True)
    times = torch.tensor([0.5, 0.999999])
    values = model(state, times, torch.tensor([1, 2]))
    assert values[0].item() == pytest.approx(0.0, abs=1e-6)
    gradient = torch.autograd.grad(values.sum(), state)[0]
    torch.testing.assert_close(gradient[0], torch.zeros_like(gradient[0]))
    assert gradient[1].square().mean().sqrt().item() < 1e-4
    assert values[1].item() == pytest.approx(2.0, abs=2e-3)


def test_retention_and_invalid_domains() -> None:
    retention = ou_relative_retention(torch.tensor([0.5, 0.8]), switch_time=0.5)
    assert retention[0].item() == pytest.approx(1.0)
    assert 0.0 < retention[1].item() < 1.0
    with pytest.raises(ValueError):
        rae_ou_coefficients(torch.tensor([1.0]))
    with pytest.raises(ValueError):
        ou_relative_retention(torch.tensor([0.4]), switch_time=0.5)


def test_first_hjb_level_reduces_exactly_to_running_cost() -> None:
    target = RAEv2NormalizedOUValue(2, 3, width=8, depth=1)
    target.eval().requires_grad_(False)
    state = torch.randn(2, 2, 2, 2)
    old_time = torch.full((2,), 0.5)
    labels = torch.tensor([0, 2])
    step = torch.full((2,), 0.03)
    values, diagnostics = build_hjb_target(
        ConstantDualClean(),
        target,
        ou_state=state,
        old_time=old_time,
        labels=labels,
        semigroup_step=step,
        beta=1.78,
        particles=2,
        precision="fp32",
        generator=torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(values, step * diagnostics["running_cost"])
    torch.testing.assert_close(
        diagnostics["gradient_term"], torch.zeros_like(step)
    )


def test_curriculum_sampling_preserves_bank_labels_and_adjacent_levels() -> None:
    bank = {
        "ou_states": torch.arange(4 * 2, dtype=torch.float16).reshape(4, 2, 1, 1),
        "labels": torch.tensor([0, 1, 2, 3]),
    }
    levels = torch.linspace(0.3, 0.9, 4)
    state, old_time, new_time, labels, step = sample_curriculum_batch(
        bank,
        semigroup_levels=levels,
        maximum_level=3,
        batch_size=6,
        generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
    )
    assert state.shape == (6, 2, 1, 1)
    assert labels.shape == old_time.shape == new_time.shape == step.shape == (6,)
    assert torch.all(new_time > old_time)
    assert torch.allclose(step, torch.full_like(step, 0.2))
    assert torch.all((labels >= 0) & (labels <= 3))


def test_value_ema_updates_parameters_without_changing_source() -> None:
    source = nn.Linear(2, 1, bias=False)
    target = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(2.0)
        target.weight.zero_()
    update_ema(target, source, 0.75)
    torch.testing.assert_close(target.weight, torch.full_like(target.weight, 0.5))
    torch.testing.assert_close(source.weight, torch.full_like(source.weight, 2.0))
