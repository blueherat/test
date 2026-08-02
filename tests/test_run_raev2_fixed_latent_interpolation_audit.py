from __future__ import annotations

import torch

from experiments.run_raev2_fixed_latent_interpolation_audit import (
    sample_path_comparison,
)


def test_sample_path_comparison_is_exact_for_matching_paths() -> None:
    control = torch.zeros(3, 2)
    actual = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    result = sample_path_comparison(control, actual, actual.clone())

    torch.testing.assert_close(
        result["actual_counterfactual_mismatch_over_actual_step"],
        torch.zeros(3),
    )
    torch.testing.assert_close(
        result["counterfactual_over_actual_step_norm"], torch.ones(3)
    )
    torch.testing.assert_close(
        result["actual_counterfactual_step_cosine"], torch.ones(3)
    )


def test_sample_path_comparison_detects_orthogonal_path() -> None:
    control = torch.zeros(1, 2)
    actual = torch.tensor([[1.0, 0.0]])
    counterfactual = torch.tensor([[0.0, 1.0]])

    result = sample_path_comparison(control, actual, counterfactual)

    torch.testing.assert_close(
        result["actual_counterfactual_step_cosine"], torch.zeros(1)
    )
    torch.testing.assert_close(
        result["actual_counterfactual_mismatch_over_actual_step"],
        torch.tensor([2.0**0.5]),
    )
