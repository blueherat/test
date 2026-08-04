import numpy as np

from experiments.summarize_raev2_ig_pulse_validation import (
    build_summary_manifest,
    central_derivative,
    pair_scalars,
    sample_cosine,
    sample_rms,
    summarize_values,
)


def test_summary_manifest_marks_complete(tmp_path):
    result = build_summary_manifest(
        [tmp_path / "seed_a", tmp_path / "seed_b"],
        bootstrap_repeats=5000,
        seed=7,
        rows=12,
    )
    assert result["status"] == "complete"
    assert result["rows"] == 12


def test_central_derivative_recovers_linear_response():
    baseline = np.zeros((3, 2, 2), dtype=np.float32)
    direction = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    gamma = 0.05
    recovered = central_derivative(baseline + gamma * direction, baseline - gamma * direction, gamma)
    np.testing.assert_allclose(recovered, direction, rtol=1e-6, atol=1e-6)


def test_pair_scalars_separates_odd_and_even_response():
    baseline = np.zeros((2, 1, 2), dtype=np.float32)
    direction = np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    curvature = np.full_like(direction, 0.25)
    gamma = 0.1
    positive = baseline + gamma * direction + gamma**2 * curvature
    negative = baseline - gamma * direction + gamma**2 * curvature
    result = pair_scalars(
        baseline,
        positive,
        negative,
        gamma=gamma,
        unit_injected_norm=np.ones(2),
    )
    np.testing.assert_allclose(result.derivative, direction, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.propagation_gain, sample_rms(direction))
    expected_even = sample_rms(gamma**2 * curvature)
    np.testing.assert_allclose(
        result.even_over_odd,
        expected_even / (gamma * sample_rms(direction)),
        rtol=2e-6,
        atol=1e-9,
    )


def test_cross_gamma_vector_check_detects_direction_change_with_equal_norm():
    left = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    right = np.array([[[0.0, 1.0]], [[1.0, 0.0]]])
    np.testing.assert_allclose(sample_rms(left), sample_rms(right))
    np.testing.assert_allclose(sample_cosine(left, right), 0.0)


def test_summary_reports_heavy_tail_ratio():
    summary = summarize_values(np.array([1.0, 1.0, 1.0, 20.0]), repeats=100, seed=0)
    assert summary["mean_ci_low"] <= summary["mean"] <= summary["mean_ci_high"]
    assert summary["q95_over_median"] > 10.0
