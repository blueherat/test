"""Both independent FID formulas against diagonal-Gaussian closed forms."""
import numpy as np
from experiments.audit_raev2_prefix_ratio64k_quality import independent_fid


def test_full_covariance_reconstruction_against_known_diagonal_distance():
    # Four deterministic points with exactly Bessel covariance diag(1,16).
    centered = np.array([[np.sqrt(1.5), 0], [-np.sqrt(1.5), 0],
                         [0, np.sqrt(24)], [0, -np.sqrt(24)]])
    features = centered+np.array([1, -2])
    result = independent_fid(features, np.array([0, 1]), np.diag([4., 9.]), np.diag([2., 3.]))
    # Squared mean distance10 plus variance-root distance (1-2)^2+(4-3)^2=2.
    np.testing.assert_allclose(result['fid'], 12., atol=1e-12, rtol=0)


def test_rank_deficient_gram_reconstruction_against_known_distance():
    centered = np.array([[np.sqrt(1.5), 0], [-np.sqrt(1.5), 0],
                         [0, np.sqrt(24)], [0, -np.sqrt(24)]])
    features = np.pad(centered+np.array([1, -2]), ((0, 0), (0, 3)))
    result = independent_fid(features, np.array([0, 1, 0, 0, 0]), np.diag([4., 9., 1., 1., 1.]))
    # The three missing-variance coordinates add exactly three to the same FID.
    np.testing.assert_allclose(result['fid'], 15., atol=1e-6, rtol=0)
