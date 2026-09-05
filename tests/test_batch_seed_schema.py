from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.batch_seed_schema import (
    DEFAULT_BATCH_SEED_SCHEMA,
    LEGACY_BATCH_SEED_SCHEMA,
    NAMESPACED_BATCH_SEED_SCHEMA,
    batch_rng_manifest,
    batch_seed,
    manifest_uses_batch_rng,
)


def test_namespaced_v2_gives_distinct_runs_disjoint_batch_seeds() -> None:
    seed0 = {batch_seed(0, index) for index in range(625)}
    seed1 = {batch_seed(1, index) for index in range(625)}

    assert DEFAULT_BATCH_SEED_SCHEMA == NAMESPACED_BATCH_SEED_SCHEMA
    assert seed0.isdisjoint(seed1)
    assert batch_seed(0, 624) == 624
    assert batch_seed(1, 0) == 1 << 32


def test_legacy_schema_exactly_reproduces_adjacent_run_overlap() -> None:
    seed0 = {
        batch_seed(0, index, schema=LEGACY_BATCH_SEED_SCHEMA)
        for index in range(625)
    }
    seed1 = {
        batch_seed(1, index, schema=LEGACY_BATCH_SEED_SCHEMA)
        for index in range(625)
    }

    assert len(seed0 & seed1) == 624
    assert batch_seed(1, 0, schema=LEGACY_BATCH_SEED_SCHEMA) == 1


def test_batch_rng_manifest_records_exact_versioned_formula() -> None:
    payload = batch_rng_manifest(7)
    assert payload == {
        "schema": "namespaced_v2",
        "formula": "batch_seed=(uint64(run_seed)<<32)|uint64(batch_index)",
        "run_seed": 7,
    }
    assert manifest_uses_batch_rng({"batch_rng": payload}, 7)
    assert not manifest_uses_batch_rng({"batch_rng": payload}, 8)


def test_missing_batch_rng_metadata_is_only_implicit_legacy() -> None:
    historical_manifest: dict[str, object] = {}
    assert manifest_uses_batch_rng(
        historical_manifest,
        0,
        schema=LEGACY_BATCH_SEED_SCHEMA,
    )
    assert not manifest_uses_batch_rng(historical_manifest, 0)


@pytest.mark.parametrize(
    ("run_seed", "batch_index", "message"),
    [
        (-1, 0, "run_seed"),
        (1 << 32, 0, "run_seed"),
        (0, -1, "batch_index"),
        (0, 1 << 32, "batch_index"),
    ],
)
def test_namespaced_v2_rejects_values_outside_uint32(
    run_seed: int,
    batch_index: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        batch_seed(run_seed, batch_index)
