from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from experiments.audit_imagenet100_sit_dual_endpoint import (
    RmsAccumulator,
    endpoint_tensors,
    parse_times,
    scalar_summary,
    stratified_indices,
)


def pack_output(
    epsilon: torch.Tensor,
    clean: torch.Tensor,
    gate_probability: float,
) -> torch.Tensor:
    logit = math.log(gate_probability / (1.0 - gate_probability))
    gate_logits = torch.full(
        (len(clean), 1, *clean.shape[2:]),
        logit,
        dtype=clean.dtype,
    )
    return torch.cat((epsilon, clean, gate_logits), dim=1)


def test_parse_times_requires_a_strict_unit_interval_grid() -> None:
    assert parse_times("0,.001,0.5,1") == (0.0, 0.001, 0.5, 1.0)
    with pytest.raises(Exception):
        parse_times("0.5,0.5")
    with pytest.raises(Exception):
        parse_times("-0.1,0.5")


def test_stratified_indices_are_balanced_unique_and_deterministic() -> None:
    labels = np.repeat(np.arange(4), 10)
    first = stratified_indices(labels, 18, seed=7, num_classes=4)
    second = stratified_indices(labels, 18, seed=7, num_classes=4)
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 18
    assert np.bincount(labels[first], minlength=4).tolist() == [5, 5, 4, 4]


def test_scalar_and_rms_summaries_use_declared_grains() -> None:
    summary = scalar_summary(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(1.5)

    accumulator = RmsAccumulator()
    accumulator.update(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    rms = accumulator.summary()
    assert rms["rms"] == pytest.approx(math.sqrt(25.0 / 4.0))
    assert rms["sample_rms_mean"] == pytest.approx(math.sqrt(12.5) / 2.0)


def test_perfect_native_heads_give_zero_velocity_error() -> None:
    clean = torch.randn(2, 4, 3, 3)
    epsilon = torch.randn_like(clean)
    times = torch.tensor([0.2, 0.8])
    tensors = endpoint_tensors(
        pack_output(epsilon, clean, 0.4),
        clean,
        epsilon,
        times,
        gate_activation="sigmoid",
        denominator_floor=1e-3,
    )
    for name in (
        "clean_native_error",
        "epsilon_native_error",
        "x_velocity_error",
        "epsilon_velocity_error",
        "dynamic_velocity_error",
        "x_error_contribution",
        "epsilon_error_contribution",
        "pre_switch_dynamic_error",
    ):
        assert torch.allclose(tensors[name], torch.zeros_like(tensors[name]), atol=1e-6)


def test_endpoint_contribution_exposes_gate_over_one_minus_t() -> None:
    clean = torch.zeros(1, 4, 2, 2)
    epsilon = torch.zeros_like(clean)
    clean_prediction = torch.ones_like(clean)
    output = pack_output(epsilon, clean_prediction, 0.03)
    tensors = endpoint_tensors(
        output,
        clean,
        epsilon,
        torch.tensor([0.99]),
        gate_activation="sigmoid",
        denominator_floor=1e-3,
    )
    assert torch.allclose(
        tensors["x_error_contribution"],
        torch.full_like(clean, 3.0),
        atol=5e-6,
    )
    assert torch.allclose(
        tensors["dynamic_velocity_error"],
        tensors["pre_switch_dynamic_error"],
        atol=5e-6,
    )
