from __future__ import annotations

import torch

from experiments.run_imagenet100_sit_guidance_density_action import (
    COMPONENT_NAMES,
    approximate_linear_flow_score,
    component_jacobian_symmetry_probe,
    density_action_terms,
    hutchinson_component_divergence,
    stack_guidance_components,
)


def test_stack_guidance_components_preserves_exact_decompositions() -> None:
    anchor = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    x_other = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    v_other = torch.tensor([[0.5, -1.0], [-2.0, 1.0]])
    stacked = stack_guidance_components(anchor, x_other, v_other)
    assert stacked.shape == (len(COMPONENT_NAMES), *anchor.shape)
    torch.testing.assert_close(stacked[3], anchor - x_other)
    torch.testing.assert_close(stacked[4], anchor - v_other)
    torch.testing.assert_close(stacked[5], stacked[7] + stacked[8])
    torch.testing.assert_close(stacked[6], stacked[9] + stacked[10])


def test_linear_flow_score_formula() -> None:
    state = torch.tensor([[2.0, -1.0]])
    velocity = torch.tensor([[0.5, 1.5]])
    expected = (0.25 * velocity - state) / 0.75
    torch.testing.assert_close(
        approximate_linear_flow_score(state, velocity, 0.25), expected
    )


def test_rotation_is_exactly_density_inactive_for_standard_gaussian() -> None:
    state = torch.tensor([[1.0, 2.0], [-0.5, 0.25]])
    probes = torch.tensor(
        [
            [[1.0, 1.0], [1.0, -1.0]],
            [[1.0, -1.0], [-1.0, 1.0]],
        ]
    )

    def rotation(value: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.stack((-value[:, 1], value[:, 0]), dim=1),))

    components, divergence = hutchinson_component_divergence(
        rotation, state, probes
    )
    terms = density_action_terms(components, -state, divergence)
    torch.testing.assert_close(
        terms["divergence_per_dim"], torch.zeros_like(divergence)
    )
    torch.testing.assert_close(
        terms["score_work_per_dim"],
        torch.zeros_like(terms["score_work_per_dim"]),
    )
    torch.testing.assert_close(
        terms["density_action_per_dim"],
        torch.zeros_like(terms["density_action_per_dim"]),
    )


def test_radial_field_has_expected_standard_gaussian_density_action() -> None:
    state = torch.tensor([[1.0, 0.0], [2.0, -1.0]])
    probes = torch.tensor(
        [
            [[1.0, 1.0], [1.0, -1.0]],
            [[-1.0, 1.0], [-1.0, -1.0]],
        ]
    )

    def radial(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0)

    components, divergence = hutchinson_component_divergence(radial, state, probes)
    terms = density_action_terms(components, -state, divergence)
    # Per-dimension action is (d - ||z||^2) / d for d=2.
    expected = torch.tensor([[0.5, -1.5]])
    torch.testing.assert_close(
        terms["density_action_per_dim"],
        expected.unsqueeze(0).expand_as(terms["density_action_per_dim"]),
    )


def test_component_symmetry_distinguishes_gradient_and_rotation_fields() -> None:
    state = torch.tensor([[0.3, -0.7], [1.1, 0.2]])
    probe = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    def fields(value: torch.Tensor) -> torch.Tensor:
        symmetric = torch.stack((2.0 * value[:, 0], 3.0 * value[:, 1]), dim=1)
        rotation = torch.stack((-value[:, 1], value[:, 0]), dim=1)
        return torch.stack((symmetric, rotation))

    metrics = component_jacobian_symmetry_probe(fields, state, probe, [0, 1])
    torch.testing.assert_close(
        metrics["antisymmetric_energy_fraction"][0],
        torch.zeros_like(metrics["antisymmetric_energy_fraction"][0]),
    )
    torch.testing.assert_close(
        metrics["antisymmetric_energy_fraction"][1],
        torch.ones_like(metrics["antisymmetric_energy_fraction"][1]),
    )
