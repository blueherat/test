"""Versioned batch-seed derivation for reproducible sampling runs.

The legacy samplers used ``run_seed + batch_index``.  Adjacent run seeds then
share every batch seed except one, so nominal run seeds 0 and 1 overlap almost
completely.  The default v2 schema reserves the high 32 bits for the run and
the low 32 bits for the batch, making the namespaces of distinct run seeds
disjoint within the documented uint32 ranges.
"""

from __future__ import annotations

from typing import Any, Mapping


LEGACY_BATCH_SEED_SCHEMA = "legacy_additive_v1"
NAMESPACED_BATCH_SEED_SCHEMA = "namespaced_v2"
DEFAULT_BATCH_SEED_SCHEMA = NAMESPACED_BATCH_SEED_SCHEMA
BATCH_SEED_SCHEMAS = (
    NAMESPACED_BATCH_SEED_SCHEMA,
    LEGACY_BATCH_SEED_SCHEMA,
)

_UINT32_MAX = (1 << 32) - 1
_TORCH_SEED_MIN = -(1 << 63)
_TORCH_SEED_MAX = (1 << 64) - 1
_FORMULAS = {
    NAMESPACED_BATCH_SEED_SCHEMA: (
        "batch_seed=(uint64(run_seed)<<32)|uint64(batch_index)"
    ),
    LEGACY_BATCH_SEED_SCHEMA: "batch_seed=run_seed+batch_index",
}


def _integer(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def batch_seed(
    run_seed: int,
    batch_index: int,
    *,
    schema: str = DEFAULT_BATCH_SEED_SCHEMA,
) -> int:
    """Derive the torch seed for one batch under a versioned schema."""

    run_seed = _integer("run_seed", run_seed)
    batch_index = _integer("batch_index", batch_index)
    if schema == NAMESPACED_BATCH_SEED_SCHEMA:
        if not 0 <= run_seed <= _UINT32_MAX:
            raise ValueError("namespaced_v2 run_seed must fit in uint32")
        if not 0 <= batch_index <= _UINT32_MAX:
            raise ValueError("namespaced_v2 batch_index must fit in uint32")
        return (run_seed << 32) | batch_index
    if schema == LEGACY_BATCH_SEED_SCHEMA:
        seed = run_seed + batch_index
        if not _TORCH_SEED_MIN <= seed <= _TORCH_SEED_MAX:
            raise ValueError("legacy additive batch seed is outside torch's seed range")
        return seed
    raise ValueError(f"unsupported batch seed schema: {schema}")


def batch_seed_formula(schema: str = DEFAULT_BATCH_SEED_SCHEMA) -> str:
    """Return the exact formula recorded in sampling manifests."""

    try:
        return _FORMULAS[schema]
    except KeyError as error:
        raise ValueError(f"unsupported batch seed schema: {schema}") from error


def batch_rng_manifest(
    run_seed: int,
    *,
    schema: str = DEFAULT_BATCH_SEED_SCHEMA,
) -> dict[str, Any]:
    """Build the canonical manifest payload for a sampling run's batch RNG."""

    batch_seed(run_seed, 0, schema=schema)
    return {
        "schema": schema,
        "formula": batch_seed_formula(schema),
        "run_seed": run_seed,
    }


def manifest_uses_batch_rng(
    manifest: Mapping[str, Any],
    run_seed: int,
    *,
    schema: str = DEFAULT_BATCH_SEED_SCHEMA,
) -> bool:
    """Check a manifest, treating missing metadata as the historical schema."""

    payload = manifest.get("batch_rng")
    if payload is None:
        return schema == LEGACY_BATCH_SEED_SCHEMA
    return payload == batch_rng_manifest(run_seed, schema=schema)
