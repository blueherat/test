import numpy as np

from experiments.advfd_cleanroom.run_selective_amplification_analytic import (
    selective_amplification_distances,
)


def test_real_whitening_diverges_while_pooled_whitening_is_bounded() -> None:
    amplification = np.array([1.0, 10.0, 100.0, 1_000_000.0])
    real_fd, pooled_fd = selective_amplification_distances(amplification, 0.05)
    assert real_fd[-1] > 1_000_000
    assert np.all(pooled_fd < 4.0)
    expected_limit = 1.0 / (0.5 - 0.25 * 0.05)
    assert abs(pooled_fd[-1] - expected_limit) < 1e-4


def test_selective_amplification_input_validation() -> None:
    for mass in (0.0, 1.0):
        try:
            selective_amplification_distances(np.array([1.0]), mass)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid artifact mass: {mass}")
