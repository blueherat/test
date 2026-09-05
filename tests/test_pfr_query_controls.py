from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch_seed_schema import (
    DEFAULT_BATCH_SEED_SCHEMA,
    LEGACY_BATCH_SEED_SCHEMA,
    batch_rng_manifest,
)
from experiments.information_purification_ig import projected_information_query
from experiments.pfr_query_controls import (
    controlled_information_query,
    matched_donor_shift,
    matched_orthogonal_scramble,
    response_odd_even,
    split_spatial_response,
)
from experiments.run_imagenet100_sit_pfr_query_controls import (
    QueryControlledField,
    result_reusable,
)


def _inputs(batch: int = 4):
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(batch, 4, 3, 3, generator=generator)
    strong = torch.randn(batch, 4, 3, 3, generator=generator)
    weak = torch.randn(batch, 4, 3, 3, generator=generator)
    gamma = 0.6
    guided = strong + gamma * (strong - weak)
    return state, strong, weak, guided, gamma


def _query(kind: str):
    state, strong, weak, guided, gamma = _inputs()
    time = torch.tensor(0.2)
    query = controlled_information_query(
        state,
        time,
        strong_now=strong,
        weak_now=weak,
        guided_now=guided,
        gamma=gamma,
        horizon=1.0 / 32.0,
        intervention_time=0.5,
        kind=kind,
    )
    return query, state, time, strong, weak, guided, gamma


def _sample_dot(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float().flatten(1) * right.float().flatten(1)).sum(dim=1)


def _sample_norm(value: torch.Tensor) -> torch.Tensor:
    return value.float().flatten(1).norm(dim=1)


def test_projected_control_exactly_matches_deployed_query():
    query, state, time, strong, weak, guided, gamma = _query("projected")
    expected = projected_information_query(
        state,
        time,
        strong_now=strong,
        weak_now=weak,
        guided_now=guided,
        gamma=gamma,
        horizon=1.0 / 32.0,
        intervention_time=0.5,
    )
    torch.testing.assert_close(query.state, expected.state)
    torch.testing.assert_close(query.time, expected.time)
    torch.testing.assert_close(query.spatial_shift, expected.state - state)


def test_time_only_and_state_only_change_exactly_one_query_coordinate():
    projected, state, time, *_ = _query("projected")
    time_only, *_ = _query("time_only")
    state_only, *_ = _query("state_only")

    torch.testing.assert_close(time_only.state, state)
    torch.testing.assert_close(time_only.time, projected.time)
    torch.testing.assert_close(state_only.state, projected.state)
    torch.testing.assert_close(state_only.time, time)


def test_anti_projected_reverses_only_the_spatial_displacement():
    projected, state, *_ = _query("projected")
    anti, *_ = _query("anti_projected")
    torch.testing.assert_close(anti.spatial_shift, -projected.spatial_shift)
    torch.testing.assert_close(anti.time, projected.time)
    torch.testing.assert_close(anti.state, state - projected.spatial_shift)


def test_orthogonal_control_is_per_sample_orthogonal_and_norm_matched():
    projected, *_ = _query("projected")
    orthogonal, *_ = _query("orthogonal_projected")
    reference = projected.spatial_shift
    controlled = orthogonal.spatial_shift
    torch.testing.assert_close(
        _sample_dot(reference, controlled),
        torch.zeros(len(reference)),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        _sample_norm(controlled), _sample_norm(reference), atol=2e-6, rtol=2e-6
    )


def test_donor_control_uses_another_sample_and_restores_recipient_norm():
    projected, *_ = _query("projected")
    reference = projected.spatial_shift
    donor = matched_donor_shift(reference)
    torch.testing.assert_close(
        _sample_norm(donor), _sample_norm(reference), atol=2e-6, rtol=2e-6
    )
    donor_directions = torch.roll(reference, shifts=1, dims=0)
    cosine = _sample_dot(donor, donor_directions) / (
        _sample_norm(donor) * _sample_norm(donor_directions)
    ).clamp_min(1e-30)
    torch.testing.assert_close(cosine, torch.ones_like(cosine), atol=2e-6, rtol=2e-6)


def test_orthogonal_and_donor_helpers_reject_invalid_inputs():
    with pytest.raises(ValueError, match="at least two"):
        matched_donor_shift(torch.ones(1, 4, 2, 2))
    with pytest.raises(ValueError, match="batched"):
        matched_orthogonal_scramble(torch.ones(4))


def test_odd_even_response_exactly_reconstructs_symmetric_queries():
    center = torch.randn(3, 4, 2, 2)
    odd_truth = torch.randn_like(center)
    even_truth = torch.randn_like(center)
    positive = center + odd_truth + even_truth
    negative = center - odd_truth + even_truth
    odd, even = response_odd_even(center, positive, negative)
    torch.testing.assert_close(odd, odd_truth)
    torch.testing.assert_close(even, even_truth)
    torch.testing.assert_close(positive - center, odd + even)
    torch.testing.assert_close(negative - center, -odd + even)


def test_spatial_response_split_is_exact_and_orthogonal_to_time_response():
    weak_now = torch.randn(3, 4, 2, 2)
    temporal = torch.randn_like(weak_now)
    spatial = torch.randn_like(weak_now)
    weak_time = weak_now + temporal
    weak_projected = weak_time + spatial
    split = split_spatial_response(weak_now, weak_time, weak_projected)
    torch.testing.assert_close(split.temporal, temporal)
    torch.testing.assert_close(split.spatial, spatial)
    torch.testing.assert_close(
        split.spatial_parallel + split.spatial_orthogonal, spatial
    )
    torch.testing.assert_close(
        _sample_dot(split.spatial_orthogonal, temporal),
        torch.zeros(len(temporal)),
        atol=2e-5,
        rtol=2e-5,
    )


class _FakeRuntime:
    @staticmethod
    def _strong(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return 1.7 * state + time

    @staticmethod
    def _weak(time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return 0.4 * state.square() - 0.3 * state + 2.0 * time

    def evaluate_pair(self, time, state, labels):
        del labels
        return self._strong(time, state), self._weak(time, state)

    def evaluate_weak(self, time, state, labels):
        del labels
        return self._weak(time, state)


def test_parallel_and_orthogonal_response_fields_exactly_recompose_pfr():
    state = torch.randn(3, 4, 2, 2)
    labels = torch.zeros(3, dtype=torch.long)
    time = torch.tensor(0.2)
    runtime = _FakeRuntime()
    values = {}
    for condition in (
        "time_only",
        "projected",
        "projected_temporal_parallel",
        "projected_temporal_orthogonal",
    ):
        values[condition] = QueryControlledField(
            runtime, labels, condition, record_diagnostics=False
        )(time, state)
    torch.testing.assert_close(
        values["projected_temporal_parallel"]
        + values["projected_temporal_orthogonal"]
        - values["time_only"],
        values["projected"],
        atol=2e-6,
        rtol=2e-6,
    )


def test_result_reuse_requires_the_requested_batch_rng_schema(tmp_path: Path) -> None:
    result_path = tmp_path / "condition_result.json"
    manifest = {
        "sampling": {"num_samples": 1000, "batch_size": 8, "seed": 0},
        "batch_rng": batch_rng_manifest(0),
        "query": {"clock": "raw_t", "clock_anchor_time": 0.25},
    }
    result = {
        "condition": "projected",
        "sampling_manifest": manifest,
        "metrics": {"fid": 1.0, "sfid": 2.0, "inception_score": 3.0},
    }
    args = Namespace(
        num_samples=1000,
        batch_size=8,
        seed=0,
        batch_seed_schema=DEFAULT_BATCH_SEED_SCHEMA,
        query_clock="raw_t",
        clock_anchor_time=0.25,
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert result_reusable(result_path, "projected", args)

    manifest.pop("batch_rng")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not result_reusable(result_path, "projected", args)
    args.batch_seed_schema = LEGACY_BATCH_SEED_SCHEMA
    assert result_reusable(result_path, "projected", args)
