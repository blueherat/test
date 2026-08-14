from __future__ import annotations

import pytest
import torch

from experiments.nominal_guidance_transfer import (
    donor_inputs,
    intervention_guidance,
    nominal_transfer_derivatives,
    nominal_transfer_metrics,
    samplewise_gap_projection,
)


def test_samplewise_projection_reconstructs_current_gap() -> None:
    nominal = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    current = torch.tensor([[[3.0, 4.0]], [[5.0, 6.0]]])

    result = samplewise_gap_projection(nominal, current)

    torch.testing.assert_close(result.coefficient, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        result.current_parallel + result.current_orthogonal,
        current,
    )
    torch.testing.assert_close(
        nominal + result.delta_parallel + result.delta_orthogonal,
        current,
    )
    dot = (result.current_orthogonal * nominal).flatten(1).sum(dim=1)
    torch.testing.assert_close(dot, torch.zeros_like(dot))
    assert result.valid.all()


def test_zero_nominal_gap_is_marked_invalid() -> None:
    nominal = torch.zeros(2, 3)
    current = torch.ones(2, 3)

    result = samplewise_gap_projection(nominal, current)

    assert not result.valid.any()
    torch.testing.assert_close(result.coefficient, torch.zeros(2))
    torch.testing.assert_close(result.current_orthogonal, current)


def test_nominal_derivatives_have_exact_intervention_semantics() -> None:
    anchor_base = torch.tensor([[2.0, 1.0]])
    other_base = torch.tensor([[1.0, 1.0]])
    anchor_frozen = torch.tensor([[4.0, 2.0]])
    anchor_gain = torch.tensor([[5.0, 2.0]])
    other_gain = torch.tensor([[3.0, 0.0]])  # gap=(2,2), projection=(2,0)
    anchor_direction = torch.tensor([[6.0, 2.0]])
    other_direction = torch.tensor([[5.0, 0.0]])  # gap=(1,2), orth=(0,2)
    anchor_closed = torch.tensor([[7.0, 3.0]])
    other_closed = torch.tensor([[4.0, 1.0]])

    result = nominal_transfer_derivatives(
        anchor_baseline=anchor_base,
        other_baseline=other_base,
        anchor_frozen=anchor_frozen,
        anchor_gain=anchor_gain,
        other_gain=other_gain,
        anchor_direction=anchor_direction,
        other_direction=other_direction,
        anchor_closed=anchor_closed,
        other_closed=other_closed,
        gamma=0.5,
    )

    torch.testing.assert_close(result.baseline, anchor_base)
    torch.testing.assert_close(result.frozen, torch.tensor([[4.5, 2.0]]))
    torch.testing.assert_close(result.replay, torch.tensor([[2.5, 1.0]]))
    torch.testing.assert_close(result.gain_only, torch.tensor([[6.0, 2.0]]))
    torch.testing.assert_close(result.direction_only, torch.tensor([[6.5, 3.0]]))
    torch.testing.assert_close(result.closed, torch.tensor([[8.5, 4.0]]))


def test_constant_gap_has_no_transfer_change() -> None:
    nominal = torch.randn(7, 4, 3)
    metrics = nominal_transfer_metrics(
        nominal,
        nominal.clone(),
        state_shift=torch.ones_like(nominal),
    )

    assert metrics["cosine"].min().item() == pytest.approx(1.0, abs=1e-6)
    assert metrics["coefficient"].mean().item() == pytest.approx(1.0, abs=1e-6)
    assert metrics["change_rms"].max().item() == 0.0
    assert metrics["delta_orthogonal_rms"].max().item() < 1e-7
    assert metrics["state_shift_valid"].all()


def test_zero_state_shift_has_undefined_secant_gain() -> None:
    nominal = torch.randn(3, 4)
    metrics = nominal_transfer_metrics(
        nominal,
        nominal,
        state_shift=torch.zeros_like(nominal),
    )

    assert not metrics["state_shift_valid"].any()
    assert torch.isnan(metrics["effective_secant_gain"]).all()


def test_intervention_guidance_splits_gain_and_direction_updates() -> None:
    nominal = torch.tensor([[1.0, 0.0]])
    current = torch.tensor([[2.0, 3.0]])

    torch.testing.assert_close(
        intervention_guidance(nominal, current, mode="frozen"),
        nominal,
    )
    torch.testing.assert_close(
        intervention_guidance(nominal, current, mode="gain_only"),
        torch.tensor([[2.0, 0.0]]),
    )
    torch.testing.assert_close(
        intervention_guidance(nominal, current, mode="direction_only"),
        torch.tensor([[1.0, 3.0]]),
    )
    torch.testing.assert_close(
        intervention_guidance(nominal, current, mode="closed"),
        current,
    )


def test_donor_inputs_form_a_noise_class_factorial() -> None:
    noise = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    labels = torch.tensor([0, 4, 9])

    paired = donor_inputs(noise, labels, mode="paired", num_classes=10)
    same_noise = donor_inputs(
        noise,
        labels,
        mode="same_noise_other_class",
        num_classes=10,
    )
    same_class = donor_inputs(
        noise,
        labels,
        mode="other_noise_same_class",
        num_classes=10,
    )
    neither = donor_inputs(
        noise,
        labels,
        mode="other_noise_other_class",
        num_classes=10,
    )

    assert torch.equal(paired[0], noise) and torch.equal(paired[1], labels)
    assert torch.equal(same_noise[0], noise)
    assert torch.equal(same_noise[1], torch.tensor([1, 5, 0]))
    assert torch.equal(same_class[0], torch.roll(noise, 1, 0))
    assert torch.equal(same_class[1], labels)
    assert torch.equal(neither[0], torch.roll(noise, 1, 0))
    assert torch.equal(neither[1], torch.tensor([1, 5, 0]))
