from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.audit_pfr_ou_location_shape_energy import (
    GroupedEnergyMoments,
    GroupedProjectionMoments,
)


def test_grouped_energy_exactly_splits_constant_and_varying_parts() -> None:
    values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [-2.0, 1.0], [2.0, -1.0]]
    )
    groups = torch.tensor([0, 0, 1, 1])
    moments = GroupedEnergyMoments.create(num_groups=2, dimension=2)
    moments.update(values[:3], groups[:3])
    moments.update(values[3:], groups[3:])

    aggregate, per_group = moments.summarize()

    expected_total = values.square().mean()
    means = torch.stack((values[:2].mean(0), values[2:].mean(0)))
    expected_constant = means.square().mean()
    assert aggregate["total_energy"] == pytest.approx(float(expected_total))
    assert aggregate["class_constant_energy"] == pytest.approx(
        float(expected_constant)
    )
    assert aggregate["state_varying_energy"] == pytest.approx(
        float(expected_total - expected_constant)
    )
    assert aggregate["recomposition_error"] < 1e-12
    assert aggregate["unbiased_recomposition_error"] < 1e-12
    assert len(per_group) == 2


def test_pure_class_constant_field_has_no_state_varying_energy() -> None:
    values = torch.tensor(
        [[1.0, -1.0], [1.0, -1.0], [2.0, 3.0], [2.0, 3.0]]
    )
    groups = torch.tensor([0, 0, 1, 1])
    moments = GroupedEnergyMoments.create(num_groups=2, dimension=2)
    moments.update(values, groups)

    aggregate, _ = moments.summarize()

    assert aggregate["class_constant_fraction"] == pytest.approx(1.0)
    assert aggregate["unbiased_class_constant_fraction"] == pytest.approx(1.0)
    assert aggregate["state_varying_energy"] == pytest.approx(0.0)


def test_unbiased_constant_estimator_removes_sample_mean_noise_in_expectation() -> None:
    generator = torch.Generator().manual_seed(145)
    samples_per_group = 4096
    groups = torch.arange(3).repeat_interleave(samples_per_group)
    offsets = torch.tensor([[1.0, -2.0], [0.5, 0.25], [-1.5, 1.0]])
    noise = torch.randn(len(groups), 2, generator=generator)
    values = offsets[groups] + noise
    moments = GroupedEnergyMoments.create(num_groups=3, dimension=2)
    moments.update(values, groups)

    aggregate, _ = moments.summarize()

    expected_constant = float(offsets.square().mean())
    assert aggregate["unbiased_class_constant_energy"] == pytest.approx(
        expected_constant, abs=0.04
    )


def test_projection_granularity_distinguishes_adaptive_coefficients() -> None:
    references = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    values = torch.tensor(
        [[1.0, 0.0], [3.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    groups = torch.tensor([0, 0, 1, 1])
    moments = GroupedProjectionMoments.create(num_groups=2)
    moments.update(values[:3], references[:3], groups[:3])
    moments.update(values[3:], references[3:], groups[3:])

    aggregate, per_group = moments.summarize()

    assert aggregate["pointwise_explained_fraction"] == pytest.approx(1.0)
    assert aggregate["classwise_explained_fraction"] == pytest.approx(10.0 / 12.0)
    assert aggregate["global_explained_fraction"] == pytest.approx(0.75)
    assert aggregate["global_coefficient"] == pytest.approx(1.5)
    assert [row["coefficient"] for row in per_group] == pytest.approx([2.0, 1.0])
