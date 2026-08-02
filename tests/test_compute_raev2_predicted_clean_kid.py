from __future__ import annotations

import numpy as np
import torch

from experiments.compute_raev2_predicted_clean_kid import (
    kid_effect_rows,
    polynomial_mmd_unbiased,
)


def test_polynomial_mmd_matches_direct_numpy_formula() -> None:
    x = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, -1.0]], dtype=np.float32)
    y = np.asarray([[1.0, 0.0], [-1.0, 2.0], [0.5, 0.5]], dtype=np.float32)
    k_xx = (x @ x.T / 2.0 + 1.0) ** 3
    k_yy = (y @ y.T / 2.0 + 1.0) ** 3
    k_xy = (x @ y.T / 2.0 + 1.0) ** 3
    expected = (
        (k_xx.sum() - np.trace(k_xx)) / 6
        + (k_yy.sum() - np.trace(k_yy)) / 6
        - 2 * k_xy.mean()
    )
    actual = polynomial_mmd_unbiased(torch.from_numpy(x), torch.from_numpy(y))
    assert np.isclose(float(actual), expected)


def test_polynomial_mmd_is_symmetric() -> None:
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(8, 4, generator=generator)
    y = torch.randn(7, 4, generator=generator)
    torch.testing.assert_close(
        polynomial_mmd_unbiased(x, y), polynomial_mmd_unbiased(y, x)
    )
