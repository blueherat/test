from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.pfr_counterfactual_residual_theory import (
    analytic_counterexample_summary,
    discrete_coboundary,
    legacy_batch_seed_overlap,
    pfr_counterexample_fields,
    terminal_mean_witness,
)


def test_discrete_coboundary_is_exact_for_vector_sequence() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(19, 7))
    for lag in (1, 3, 8):
        bulk, boundary = discrete_coboundary(values, lag)
        np.testing.assert_allclose(bulk, boundary, rtol=1e-13, atol=1e-13)


def test_terminal_mean_witness_matches_direct_improvement() -> None:
    reference = np.asarray([[2.0, 0.0], [2.0, 2.0]])
    baseline = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    candidate = np.asarray([[1.0, 0.0], [1.0, 2.0]])
    result = terminal_mean_witness(reference, baseline, candidate)
    np.testing.assert_allclose(result["baseline_mean_error"], 5.0)
    np.testing.assert_allclose(result["candidate_mean_error"], 1.0)
    np.testing.assert_allclose(result["mean_error_improvement"], 4.0)
    assert result["benefit_margin_ratio"] > 1.0


def test_exact_pfr_toy_worsens_local_mse_but_fixes_endpoint() -> None:
    time = (np.arange(16384, dtype=np.float64) + 0.5) / 16384
    fields = pfr_counterexample_fields(time)
    delta = np.where(time < 0.5, np.minimum(1.0 / 32.0, 0.5 - time), 0.0)
    expected = fields["guided"] - fields["beta"] * (
        fields["weak_query"] - fields["weak"]
    )
    np.testing.assert_allclose(fields["pfr"], expected, rtol=1e-11, atol=1e-11)
    summary = analytic_counterexample_summary()
    np.testing.assert_allclose(
        summary["parameters"]["unscaled_response_integral"],
        -5573.0 / 983040.0,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        summary["parameters"]["unscaled_squared_response_integral"],
        4069501.0 / 25165824000.0,
        rtol=1e-14,
    )
    np.testing.assert_allclose(np.mean(fields["pfr"]), 0.0, atol=5e-9)
    np.testing.assert_allclose(fields["query_horizon"], delta, atol=0.0)
    np.testing.assert_allclose(
        np.mean(np.square(fields["pfr"])),
        summary["pfr"]["integrated_local_velocity_mse"],
        rtol=1e-6,
    )
    assert (
        summary["pfr"]["integrated_local_velocity_mse"]
        > summary["ordinary_guidance"]["integrated_local_velocity_mse"]
    )
    assert summary["pfr"]["terminal_squared_w2_and_gaussian_fid"] == 0.0
    assert summary["ordinary_guidance"]["terminal_squared_w2_and_gaussian_fid"] == 1.0

    boundaries = pfr_counterexample_fields(np.asarray([0.25, 0.5]))
    np.testing.assert_allclose(boundaries["weak"], 1.0, atol=1e-14)
    np.testing.assert_allclose(boundaries["strong"], 1.0, atol=1e-14)


def test_legacy_neighboring_run_seeds_are_not_independent() -> None:
    overlapping = legacy_batch_seed_overlap(
        0, 1, num_samples=5000, batch_size=8
    )
    assert overlapping["shared_batch_rng_seed_count"] == 624
    assert overlapping["overlapping_sample_count"] == 4992
    assert overlapping["overlapping_sample_fraction"] == 0.9984
    assert not overlapping["batch_rng_seed_disjoint"]

    disjoint = legacy_batch_seed_overlap(
        0, 1_000_003, num_samples=1000, batch_size=8
    )
    assert disjoint["shared_batch_rng_seed_count"] == 0
    assert disjoint["batch_rng_seed_disjoint"]
