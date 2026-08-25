from __future__ import annotations

import torch

from experiments.run_advfd_witness_jacobian_counterexample import (
    AmbiguousWitness,
    advfd_particle_field,
)


def test_equal_witness_values_have_opposite_pullback_fields() -> None:
    target = torch.tensor([[-1.0], [1.0]], dtype=torch.float64)
    source = torch.tensor([[-2.0], [2.0]], dtype=torch.float64)
    weights = torch.full((2,), 0.5, dtype=torch.float64)
    rows = []
    for coefficient in (0.0, -1.0 / 24.0, -0.1):
        witness = AmbiguousWitness(coefficient)
        torch.testing.assert_close(witness(target), target)
        torch.testing.assert_close(witness(source), source)
        distance, field = advfd_particle_field(
            target,
            weights,
            source,
            weights,
            witness,
            objective_mode="official_regularized",
            whitening_epsilon=1e-3,
        )
        rows.append((distance, field))

    torch.testing.assert_close(
        torch.tensor([row[0] for row in rows]),
        torch.full((3,), rows[0][0]),
    )
    assert rows[0][1][0, 0] > 0 and rows[0][1][1, 0] < 0
    assert rows[1][1].abs().max() < 1e-10
    assert rows[2][1][0, 0] < 0 and rows[2][1][1, 0] > 0
