from __future__ import annotations

import torch

from experiments.sample_imagenet100_sit_tangent_projection_fid import (
    CONDITIONS,
    _condition_latents,
)


def test_condition_latents_form_exact_projection_controls() -> None:
    baseline = torch.tensor(
        [
            [[[1.0, -2.0, 0.5]]],
            [[[0.0, 1.0, -1.0]]],
        ]
    )
    tangent = torch.tensor(
        [
            [[[1.0, 2.0, 0.0]]],
            [[[2.0, 0.0, 0.0]]],
        ]
    )
    response = torch.tensor(
        [
            [[[3.0, 1.0, 4.0]]],
            [[[1.0, -2.0, 3.0]]],
        ]
    )
    conditions, geometry = _condition_latents(
        baseline,
        tangent,
        baseline + response,
    )

    assert tuple(conditions) == CONDITIONS
    parallel = conditions["tangent_parallel"] - baseline
    orthogonal = conditions["tangent_orthogonal"] - baseline
    torch.testing.assert_close(parallel + orthogonal, response)
    torch.testing.assert_close(
        (orthogonal * tangent).flatten(1).sum(1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(conditions["tangent_raw"], baseline + tangent)
    torch.testing.assert_close(conditions["frozen"], baseline + response)
    torch.testing.assert_close(
        geometry["parallel_energy_fraction"]
        + geometry["orthogonal_energy_fraction"],
        torch.ones(2),
    )
