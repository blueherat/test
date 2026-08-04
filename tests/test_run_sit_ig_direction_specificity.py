import numpy as np
import torch

from experiments.run_sit_ig_direction_specificity import (
    build_conditions,
    deterministic_probe,
    direction_diagnostics,
    matched_orthogonal_direction,
)


def test_conditions_include_signed_ig_and_random_pairs():
    values = build_conditions((2, 7), gamma=0.01, probe_count=3, num_steps=10)
    assert len(values) == 1 + 2 * (2 + 2 * 3)
    assert sum(item.family == "ig" for item in values) == 4
    assert sum(item.family == "random" for item in values) == 12


def test_matched_direction_is_orthogonal_and_rms_matched():
    generator = torch.Generator().manual_seed(3)
    random = torch.randn((4, 3, 5), generator=generator, dtype=torch.float64)
    gap = torch.randn((4, 3, 5), generator=generator, dtype=torch.float64)
    result = matched_orthogonal_direction(random, gap)
    np.testing.assert_allclose((result * gap).mean(dim=(1, 2)), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        result.square().mean(dim=(1, 2)), gap.square().mean(dim=(1, 2)), rtol=1e-12
    )
    diagnostics = direction_diagnostics(result, gap)
    np.testing.assert_allclose(diagnostics[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(diagnostics[:, 1], 1.0, rtol=1e-12)


def test_deterministic_probe_depends_on_sample_step_and_probe():
    ids = np.array([2, 9])
    first = deterministic_probe(ids, (2, 2), seed=7, step=3, probe_index=1)
    second = deterministic_probe(ids, (2, 2), seed=7, step=3, probe_index=1)
    different = deterministic_probe(ids, (2, 2), seed=7, step=4, probe_index=1)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
