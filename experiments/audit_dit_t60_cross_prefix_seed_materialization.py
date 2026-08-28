#!/usr/bin/env python3
"""Read-only pre-freeze audit of the cross-prefix DiT branch seed slate.

The audit deliberately excludes the candidate protocol itself from seed-value
searches: that file is the declaration of the planned values, not evidence that
they were consumed.  Everything else under the configured repository and
historical evidence roots is inventoried.  Text and paths are searched for exact
decimal tokens.  DiT-relevant NPY/NPZ containers are additionally inspected via
their array headers; only integer/string arrays capable of carrying a candidate
seed are streamed, so large image/latent float and uint8 payloads are not loaded.

This is a procedural filesystem-lineage audit.  A PASS is not cryptographic
proof that a value never existed on another machine or in a deleted/unmounted
artifact.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import socket
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

import numpy as np


FORMAT = "eqvae_dit_t60_cross_prefix_seed_materialization_audit_v1"
INVENTORY_FORMAT = f"{FORMAT}_inventory"
COMPLETION_FORMAT = f"{FORMAT}_completion"
PROTOCOL_DEFAULT = Path(
    "experiments/configs/"
    "dit_imagenet256_t60_cross_prefix_mixture_validation_v1.json"
)
REPO_DEFAULT = Path(".")
EVIDENCE_DEFAULT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence"
)
PLANNED_RUNNER_DEFAULT = Path(
    "experiments/run_dit_t60_cross_prefix_mixture_validation_pool.py"
)

SELF_HASH_KEYS = {
    "inventory.json": "inventory_identity_sha256",
    "report.json": "payload_sha256",
    "completion.json": "payload_sha256",
}

TEXT_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".csv",
    ".cu",
    ".cuh",
    ".err",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".log",
    ".md",
    ".out",
    ".py",
    ".pyi",
    ".r",
    ".rmd",
    ".rst",
    ".sh",
    ".sql",
    ".tex",
    ".toml",
    ".ts",
    ".tsv",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}
TEXT_BASENAMES = {
    "completion",
    "config",
    "license",
    "manifest",
    "readme",
    "request",
    "results",
}
EXCLUDED_DIRECTORY_COMPONENTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "cache",
    "caches",
    "checkpoint",
    "checkpoints",
}
DIT_PATH_RE = re.compile(r"(?:^|[/_.-])dit(?:$|[/_.-])", re.IGNORECASE)
DIT_NAMESPACE_BYTES_RE = re.compile(rb"eqvae-dit[0-9A-Za-z][0-9A-Za-z_.:+/\-]*")
SEED_KEY_RE = re.compile(r"(?:^|_)seed(?:s)?(?:_|$)", re.IGNORECASE)


class AuditFailure(RuntimeError):
    """Fail-closed audit error."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_self_hash(payload: dict[str, Any], key: str) -> str:
    probe = copy.deepcopy(payload)
    if key not in probe:
        raise AuditFailure(f"missing self-hash key {key!r}")
    del probe[key]
    return _sha256_bytes(_canonical_json_bytes(probe))


def _add_self_hash(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key in payload:
        raise AuditFailure(f"refusing to overwrite self-hash key {key!r}")
    payload[key] = ""
    payload[key] = _canonical_self_hash(payload, key)
    return payload


def _read_self_hashed_json(path: Path, key: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - fail-closed boundary.
        raise AuditFailure(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditFailure(f"expected JSON object in {path}")
    observed = payload.get(key)
    if not isinstance(observed, str) or observed != _canonical_self_hash(payload, key):
        raise AuditFailure(f"invalid {key} in {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _is_text_path(path: Path) -> bool:
    lower_name = path.name.lower()
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if lower_name in TEXT_BASENAMES:
        return True
    return any(lower_name.startswith(f"{stem}.") for stem in TEXT_BASENAMES)


def _is_dit_path(relative_path: str) -> bool:
    return bool(DIT_PATH_RE.search(relative_path.replace(os.sep, "/")))


def _seed_token_patterns(seeds: Sequence[int]) -> tuple[re.Pattern[bytes], re.Pattern[str]]:
    ordered = sorted((str(value) for value in seeds), key=lambda item: (-len(item), item))
    byte_body = b"|".join(re.escape(item.encode("ascii")) for item in ordered)
    text_body = "|".join(re.escape(item) for item in ordered)
    return (
        re.compile(rb"(?<![0-9])(?:" + byte_body + rb")(?![0-9])"),
        re.compile(r"(?<![0-9])(?:" + text_body + r")(?![0-9])"),
    )


def _match_candidate_text(
    data: bytes,
    byte_pattern: re.Pattern[bytes],
    text_pattern: re.Pattern[str],
) -> list[tuple[int, int]]:
    """Return (seed, byte/character offset) exact-decimal matches.

    UTF-8/ASCII-compatible files are searched as bytes.  UTF-16 files with an
    explicit BOM are decoded and searched as text.  The decimal-boundary rule
    rejects candidate digits embedded in a longer integer.
    """

    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            decoded = data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise AuditFailure(f"invalid BOM-declared UTF-16 text: {exc}") from exc
        return [(int(match.group(0)), match.start()) for match in text_pattern.finditer(decoded)]
    return [(int(match.group(0)), match.start()) for match in byte_pattern.finditer(data)]


def _derive_candidate_seeds(protocol: dict[str, Any]) -> list[int]:
    try:
        lineage = protocol["seed_lineage"]
        derivation = lineage["derivation"]
        namespace = derivation["namespace"]
        pool_seed = int(derivation["pool_seed"])
        target_class = int(derivation["target_class_id"])
        steps = int(derivation["num_sampling_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuditFailure(f"malformed protocol seed derivation: {exc}") from exc
    if not isinstance(namespace, str):
        raise AuditFailure("seed namespace must be a string")
    try:
        namespace_bytes = namespace.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AuditFailure("seed namespace is not ASCII") from exc
    derived: list[int] = []
    for index in range(64):
        payload = b"\0".join(
            (
                namespace_bytes,
                str(pool_seed).encode("ascii"),
                str(target_class).encode("ascii"),
                str(steps).encode("ascii"),
                str(index).encode("ascii"),
            )
        )
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
            (1 << 63) - 1
        )
        derived.append(value)
    return derived


def _flatten_declared_seeds(protocol: dict[str, Any]) -> list[int]:
    try:
        shards = protocol["seed_lineage"]["branch_local_trajectory_seeds_by_shard"]
    except (KeyError, TypeError) as exc:
        raise AuditFailure(f"missing declared branch seed shards: {exc}") from exc
    if not isinstance(shards, list) or len(shards) != 8:
        raise AuditFailure("expected exactly eight seed shards")
    flattened: list[int] = []
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, list) or len(shard) != 8:
            raise AuditFailure(f"seed shard {shard_index} is not length eight")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in shard):
            raise AuditFailure(f"seed shard {shard_index} contains a non-integer")
        flattened.extend(shard)
    return flattened


def _validate_protocol(path: Path) -> tuple[dict[str, Any], bytes, list[int], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        protocol = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - fail-closed protocol boundary.
        raise AuditFailure(f"cannot read protocol {path}: {exc}") from exc
    if not isinstance(protocol, dict):
        raise AuditFailure("protocol must be a JSON object")
    observed_identity = protocol.get("protocol_identity_sha256")
    if not isinstance(observed_identity, str):
        raise AuditFailure("protocol_identity_sha256 is missing")
    if observed_identity != _canonical_self_hash(protocol, "protocol_identity_sha256"):
        raise AuditFailure("protocol canonical self-hash mismatch")

    declared = _flatten_declared_seeds(protocol)
    derived = _derive_candidate_seeds(protocol)
    if declared != derived:
        raise AuditFailure("declared seed list does not equal the frozen derivation")
    if len(set(declared)) != len(declared):
        raise AuditFailure("candidate seed slate contains duplicates")
    if any(value == 0 for value in declared):
        raise AuditFailure("candidate seed slate contains zero")
    if any(value < 0 or value >= (1 << 63) for value in declared):
        raise AuditFailure("candidate seed slate is outside [0, 2^63-1]")

    lineage = protocol["seed_lineage"]
    observed_list_hash = lineage.get("branch_local_trajectory_seed_list_sha256")
    computed_list_hash = _sha256_bytes(_canonical_json_bytes(declared))
    if observed_list_hash != computed_list_hash:
        raise AuditFailure("branch-local seed-list SHA-256 mismatch")

    snapshot = lineage.get("draft_exclusion_snapshot")
    if not isinstance(snapshot, dict):
        raise AuditFailure("missing draft_exclusion_snapshot")
    namespace_records = snapshot.get("existing_DiT_branch_local_namespaces")
    if not isinstance(namespace_records, list) or not namespace_records:
        raise AuditFailure("missing existing DiT branch-local namespace records")
    prior_values: list[int] = []
    validated_namespaces: list[dict[str, Any]] = []
    namespace_names: set[str] = set()
    for record_index, record in enumerate(namespace_records):
        if not isinstance(record, dict):
            raise AuditFailure(f"namespace record {record_index} is not an object")
        namespace = record.get("namespace")
        values = record.get("values")
        if not isinstance(namespace, str) or not namespace.startswith("eqvae-dit"):
            raise AuditFailure(f"invalid DiT namespace record {record_index}")
        if namespace in namespace_names:
            raise AuditFailure(f"duplicate declared namespace {namespace}")
        namespace_names.add(namespace)
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise AuditFailure(f"invalid values for namespace {namespace}")
        if len(values) != record.get("value_count") or len(values) != len(set(values)):
            raise AuditFailure(f"count/uniqueness mismatch for namespace {namespace}")
        values_hash = _sha256_bytes(_canonical_json_bytes(values))
        if values_hash != record.get("values_sha256"):
            raise AuditFailure(f"values SHA-256 mismatch for namespace {namespace}")
        prior_values.extend(values)
        validated_namespaces.append(
            {
                "namespace": namespace,
                "value_count": len(values),
                "values_sha256": values_hash,
                "candidate_intersection": sorted(set(values).intersection(declared)),
            }
        )
    if len(prior_values) != len(set(prior_values)):
        raise AuditFailure("declared prior branch-local values collide with each other")
    combined = sorted(set(prior_values))
    if len(combined) != snapshot.get("combined_existing_branch_local_value_count"):
        raise AuditFailure("combined prior branch-local count mismatch")
    combined_hash = _sha256_bytes(_canonical_json_bytes(combined))
    if combined_hash != snapshot.get("combined_existing_branch_local_values_sorted_sha256"):
        raise AuditFailure("combined prior branch-local SHA-256 mismatch")
    legacy = snapshot.get("legacy_upstream_demo_seed_integers_conservative_only")
    if not isinstance(legacy, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in legacy
    ):
        raise AuditFailure("invalid conservative legacy seed list")
    collisions = sorted(set(declared).intersection(prior_values).union(set(declared).intersection(legacy)))
    if collisions:
        raise AuditFailure(f"candidate collision with known/legacy seed values: {collisions}")

    checks = {
        "candidate_count": len(declared),
        "candidate_unique_count": len(set(declared)),
        "candidate_nonzero_count": sum(value != 0 for value in declared),
        "candidate_min": min(declared),
        "candidate_max": max(declared),
        "derivation_exact_match": True,
        "seed_list_sha256": computed_list_hash,
        "known_branch_local_value_count": len(combined),
        "known_branch_local_values_sorted_sha256": combined_hash,
        "known_namespace_records": validated_namespaces,
        "legacy_numeric_guard_count": len(legacy),
        "candidate_known_or_legacy_collision_count": 0,
    }
    return protocol, raw, declared, checks


def _extract_seed_ints(value: Any, seed_context: bool = False) -> set[int]:
    found: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_context = seed_context or bool(SEED_KEY_RE.search(str(key)))
            found.update(_extract_seed_ints(child, child_context))
    elif isinstance(value, list):
        for child in value:
            found.update(_extract_seed_ints(child, seed_context))
    elif seed_context and isinstance(value, int) and not isinstance(value, bool):
        found.add(value)
    return found


def _namespace_bindings(value: Any, source: str) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    if isinstance(value, dict):
        local_namespaces = {
            child
            for key, child in value.items()
            if "namespace" in str(key).lower()
            and isinstance(child, str)
            and child.startswith("eqvae-dit")
        }
        if local_namespaces:
            seeds = sorted(_extract_seed_ints(value))
            for namespace in sorted(local_namespaces):
                bindings.append(
                    {"namespace": namespace, "seed_values": seeds, "source": source}
                )
        for child in value.values():
            bindings.extend(_namespace_bindings(child, source))
    elif isinstance(value, list):
        for child in value:
            bindings.extend(_namespace_bindings(child, source))
    return bindings


def _npy_header(handle: BinaryIO) -> tuple[tuple[int, ...], bool, np.dtype[Any], int]:
    version = np.lib.format.read_magic(handle)
    shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    return tuple(int(item) for item in shape), bool(fortran_order), np.dtype(dtype), handle.tell()


def _product(shape: Sequence[int]) -> int:
    return math.prod(shape) if shape else 1


def _dtype_seed_capability(dtype: np.dtype[Any], minimum_candidate: int) -> str:
    if dtype.hasobject:
        return "unsupported_object"
    if dtype.fields:
        capabilities = {
            _dtype_seed_capability(field[0], minimum_candidate)
            for field in dtype.fields.values()
        }
        if "unsupported_object" in capabilities:
            return "unsupported_object"
        if capabilities.intersection({"integer", "string", "structured"}):
            return "structured"
        return "cannot_represent"
    if dtype.kind in {"S", "U"}:
        return "string"
    if dtype.kind not in {"i", "u"}:
        return "cannot_represent"
    try:
        info = np.iinfo(dtype)
    except ValueError:
        return "cannot_represent"
    return "integer" if int(info.max) >= minimum_candidate else "cannot_represent"


def _candidate_hits_in_array(
    handle: BinaryIO,
    *,
    dtype: np.dtype[Any],
    shape: Sequence[int],
    seeds: Sequence[int],
    text_pattern: re.Pattern[str],
) -> tuple[list[tuple[int, int]], str]:
    """Stream a primitive/structured NPY body and return seed/index hits."""

    count = _product(shape)
    itemsize = int(dtype.itemsize)
    if itemsize <= 0:
        raise AuditFailure(f"invalid dtype itemsize {itemsize}")
    elements_per_chunk = max(1, (8 * 1024 * 1024) // itemsize)
    seed_set = set(seeds)
    hits: list[tuple[int, int]] = []
    body_digest = hashlib.sha256()
    index_base = 0
    remaining = count
    while remaining:
        take = min(remaining, elements_per_chunk)
        expected = take * itemsize
        raw = handle.read(expected)
        if len(raw) != expected:
            raise AuditFailure(
                f"truncated NPY body: expected {expected} bytes, received {len(raw)}"
            )
        body_digest.update(raw)
        array = np.frombuffer(raw, dtype=dtype, count=take)
        if dtype.fields:
            for field_name in dtype.fields:
                field = array[field_name]
                field_dtype = field.dtype
                capability = _dtype_seed_capability(field_dtype, min(seeds))
                if capability == "integer":
                    flat = field.reshape(-1)
                    for offset in np.flatnonzero(np.isin(flat, seeds)).tolist():
                        hits.append((int(flat[offset]), index_base + int(offset)))
                elif capability == "string":
                    for offset, raw_value in enumerate(field.reshape(-1).tolist()):
                        value = (
                            raw_value.decode("utf-8", errors="strict")
                            if isinstance(raw_value, bytes)
                            else str(raw_value)
                        )
                        for match in text_pattern.finditer(value):
                            hits.append((int(match.group(0)), index_base + offset))
        elif dtype.kind in {"i", "u"}:
            for offset in np.flatnonzero(np.isin(array, seeds)).tolist():
                hits.append((int(array[offset]), index_base + int(offset)))
        elif dtype.kind in {"S", "U"}:
            for offset, raw_value in enumerate(array.tolist()):
                value = (
                    raw_value.decode("utf-8", errors="strict")
                    if isinstance(raw_value, bytes)
                    else str(raw_value)
                )
                for match in text_pattern.finditer(value):
                    hits.append((int(match.group(0)), index_base + offset))
        remaining -= take
        index_base += take
    if handle.read(1):
        raise AuditFailure("unexpected bytes after declared NPY body")
    return hits, body_digest.hexdigest()


def _inspect_npy_stream(
    handle: BinaryIO,
    *,
    source: str,
    expected_total_bytes: int,
    seeds: Sequence[int],
    text_pattern: re.Pattern[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    shape, fortran_order, dtype, body_offset = _npy_header(handle)
    expected_body_bytes = _product(shape) * int(dtype.itemsize)
    if body_offset + expected_body_bytes != expected_total_bytes:
        raise AuditFailure(
            f"NPY size/header mismatch in {source}: header={body_offset}, "
            f"body={expected_body_bytes}, total={expected_total_bytes}"
        )
    capability = _dtype_seed_capability(dtype, min(seeds))
    record: dict[str, Any] = {
        "source": source,
        "shape": list(shape),
        "fortran_order": fortran_order,
        "dtype": dtype.str if not dtype.fields else dtype.descr,
        "header_bytes": body_offset,
        "body_bytes": expected_body_bytes,
        "seed_scan_class": capability,
    }
    if capability == "unsupported_object":
        raise AuditFailure(
            f"object-bearing DiT metadata array cannot be safely audited without pickle: {source}"
        )
    hits: list[dict[str, Any]] = []
    if capability in {"integer", "string", "structured"}:
        array_hits, body_hash = _candidate_hits_in_array(
            handle,
            dtype=dtype,
            shape=shape,
            seeds=seeds,
            text_pattern=text_pattern,
        )
        record["inspected_body_sha256"] = body_hash
        record["candidate_hit_count"] = len(array_hits)
        hits.extend(
            {
                "seed": seed,
                "source": source,
                "element_index": index,
                "match_kind": "numpy_array_value",
            }
            for seed, index in array_hits
        )
    else:
        record["candidate_hit_count"] = 0
        record["body_read"] = False
        record["skip_reason"] = (
            "dtype cannot represent any candidate integer and is not a string/metadata dtype"
        )
    return record, hits


def _inspect_numpy_container(
    path: Path,
    *,
    logical_source: str,
    seeds: Sequence[int],
    byte_pattern: re.Pattern[bytes],
    text_pattern: re.Pattern[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    arrays: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    if path.suffix.lower() == ".npy":
        size = path.stat().st_size
        with path.open("rb") as handle:
            record, array_hits = _inspect_npy_stream(
                handle,
                source=logical_source,
                expected_total_bytes=size,
                seeds=seeds,
                text_pattern=text_pattern,
            )
        arrays.append(record)
        hits.extend(array_hits)
        return {
            "container": logical_source,
            "container_type": "npy",
            "container_bytes": size,
            "arrays": arrays,
        }, hits

    with zipfile.ZipFile(path, "r") as archive:
        names = [info.filename for info in archive.infolist()]
        if len(names) != len(set(names)):
            raise AuditFailure(f"duplicate ZIP members in {logical_source}")
        auxiliary_members: list[dict[str, Any]] = []
        for info in archive.infolist():
            member_source = f"{logical_source}::{info.filename}"
            if info.is_dir():
                auxiliary_members.append(
                    {"member": info.filename, "kind": "directory", "bytes": 0}
                )
                continue
            if info.filename.lower().endswith(".npy"):
                with archive.open(info, "r") as handle:
                    record, array_hits = _inspect_npy_stream(
                        handle,
                        source=member_source,
                        expected_total_bytes=info.file_size,
                        seeds=seeds,
                        text_pattern=text_pattern,
                    )
                record.update(
                    {
                        "zip_crc32": f"{info.CRC:08x}",
                        "compressed_bytes": info.compress_size,
                    }
                )
                arrays.append(record)
                hits.extend(array_hits)
            elif _is_text_path(Path(info.filename)):
                with archive.open(info, "r") as handle:
                    data = handle.read()
                member_hits = _match_candidate_text(data, byte_pattern, text_pattern)
                auxiliary_members.append(
                    {
                        "member": info.filename,
                        "kind": "text_metadata",
                        "bytes": info.file_size,
                        "sha256": _sha256_bytes(data),
                        "candidate_hit_count": len(member_hits),
                    }
                )
                hits.extend(
                    {
                        "seed": seed,
                        "source": member_source,
                        "offset": offset,
                        "match_kind": "numpy_container_text_member",
                    }
                    for seed, offset in member_hits
                )
            else:
                auxiliary_members.append(
                    {
                        "member": info.filename,
                        "kind": "non_npy_non_text_header_only",
                        "bytes": info.file_size,
                        "zip_crc32": f"{info.CRC:08x}",
                    }
                )
        return {
            "container": logical_source,
            "container_type": "npz",
            "container_bytes": path.stat().st_size,
            "zip_member_count": len(names),
            "arrays": arrays,
            "auxiliary_members": auxiliary_members,
        }, hits


def _scan_roots(
    roots: Sequence[tuple[str, Path]],
    *,
    planned_source_exclusions: dict[Path, str],
    seeds: Sequence[int],
    planned_directory_exclusions: dict[Path, str] | None = None,
) -> dict[str, Any]:
    byte_pattern, text_pattern = _seed_token_patterns(seeds)
    inventory_records: list[dict[str, Any]] = []
    excluded_records: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    numpy_records: list[dict[str, Any]] = []
    namespace_sources: dict[str, set[str]] = defaultdict(set)
    namespace_seed_values: dict[str, set[int]] = defaultdict(set)
    namespace_binding_sources: dict[str, set[str]] = defaultdict(set)
    counters: defaultdict[str, int] = defaultdict(int)
    planned_exclusions_abs = {
        Path(os.path.abspath(path)): reason
        for path, reason in planned_source_exclusions.items()
    }
    planned_directory_exclusions_abs = {
        Path(os.path.abspath(path)): reason
        for path, reason in (planned_directory_exclusions or {}).items()
    }

    for root_label, raw_root in roots:
        root = Path(os.path.abspath(raw_root))
        if not root.is_dir():
            raise AuditFailure(f"audit root is missing or not a directory: {root}")
        for current_raw, directory_names, file_names in os.walk(
            root, topdown=True, followlinks=False
        ):
            current = Path(current_raw)
            directory_names.sort()
            file_names.sort()
            retained_directories: list[str] = []
            for name in directory_names:
                path = current / name
                relative = path.relative_to(root).as_posix()
                logical = f"{root_label}:{relative}"
                try:
                    path_hits = _match_candidate_text(
                        relative.encode("utf-8"), byte_pattern, text_pattern
                    )
                    hits.extend(
                        {
                            "seed": seed,
                            "source": logical,
                            "offset": offset,
                            "match_kind": "directory_path_decimal_token",
                        }
                        for seed, offset in path_hits
                    )
                    path_abs = Path(os.path.abspath(path))
                    if path_abs in planned_directory_exclusions_abs:
                        excluded_records.append(
                            {
                                "root_label": root_label,
                                "relative_path": relative,
                                "entry_type": "directory",
                                "reason": planned_directory_exclusions_abs[path_abs],
                            }
                        )
                        counters["excluded_planned_artifact_directories"] += 1
                        continue
                    if name.lower() in EXCLUDED_DIRECTORY_COMPONENTS:
                        excluded_records.append(
                            {
                                "root_label": root_label,
                                "relative_path": relative,
                                "entry_type": "directory",
                                "reason": "excluded_directory_component",
                            }
                        )
                        counters["excluded_directories"] += 1
                        continue
                    if path.is_symlink():
                        target = os.readlink(path)
                        target_hits = _match_candidate_text(
                            target.encode("utf-8"), byte_pattern, text_pattern
                        )
                        hits.extend(
                            {
                                "seed": seed,
                                "source": logical,
                                "offset": offset,
                                "match_kind": "directory_symlink_target_decimal_token",
                            }
                            for seed, offset in target_hits
                        )
                        inventory_records.append(
                            {
                                "root_label": root_label,
                                "relative_path": relative,
                                "entry_type": "directory_symlink_not_followed",
                                "symlink_target": target,
                            }
                        )
                        counters["directory_symlinks"] += 1
                        continue
                    inventory_records.append(
                        {
                            "root_label": root_label,
                            "relative_path": relative,
                            "entry_type": "directory",
                        }
                    )
                    counters["directories"] += 1
                    retained_directories.append(name)
                except Exception as exc:  # noqa: BLE001 - collect all unreadables.
                    unreadable.append({"source": logical, "error": repr(exc)})
            directory_names[:] = retained_directories

            for name in file_names:
                path = current / name
                relative = path.relative_to(root).as_posix()
                logical = f"{root_label}:{relative}"
                try:
                    path_hits = _match_candidate_text(
                        relative.encode("utf-8"), byte_pattern, text_pattern
                    )
                    hits.extend(
                        {
                            "seed": seed,
                            "source": logical,
                            "offset": offset,
                            "match_kind": "file_path_decimal_token",
                        }
                        for seed, offset in path_hits
                    )
                    path_abs = Path(os.path.abspath(path))
                    if path_abs in planned_exclusions_abs:
                        excluded_records.append(
                            {
                                "root_label": root_label,
                                "relative_path": relative,
                                "entry_type": "file",
                                "reason": planned_exclusions_abs[path_abs],
                                "bytes": path.stat().st_size,
                                "sha256": _sha256_file(path),
                            }
                        )
                        counters["excluded_protocol_files"] += 1
                        continue

                    lstat = path.lstat()
                    record: dict[str, Any] = {
                        "root_label": root_label,
                        "relative_path": relative,
                        "entry_type": "file_symlink" if path.is_symlink() else "file",
                        "bytes": path.stat().st_size,
                    }
                    if path.is_symlink():
                        target = os.readlink(path)
                        record["symlink_target"] = target
                        target_hits = _match_candidate_text(
                            target.encode("utf-8"), byte_pattern, text_pattern
                        )
                        hits.extend(
                            {
                                "seed": seed,
                                "source": logical,
                                "offset": offset,
                                "match_kind": "file_symlink_target_decimal_token",
                            }
                            for seed, offset in target_hits
                        )
                        counters["file_symlinks"] += 1
                    record["lstat_bytes"] = lstat.st_size

                    if _is_text_path(path):
                        data = path.read_bytes()
                        content_hits = _match_candidate_text(
                            data, byte_pattern, text_pattern
                        )
                        record.update(
                            {
                                "scan_class": "text_exact_decimal",
                                "sha256": _sha256_bytes(data),
                                "candidate_hit_count": len(content_hits),
                            }
                        )
                        hits.extend(
                            {
                                "seed": seed,
                                "source": logical,
                                "offset": offset,
                                "match_kind": "text_content_decimal_token",
                            }
                            for seed, offset in content_hits
                        )
                        counters["text_files_scanned"] += 1
                        counters["text_bytes_scanned"] += len(data)
                        for raw_namespace in DIT_NAMESPACE_BYTES_RE.findall(data):
                            namespace_sources[raw_namespace.decode("ascii")].add(logical)
                        if path.suffix.lower() == ".json" and _is_dit_path(relative):
                            try:
                                document = json.loads(data.decode("utf-8"))
                            except Exception as exc:  # noqa: BLE001
                                raise AuditFailure(
                                    f"invalid UTF-8/JSON in DiT lineage file {logical}: {exc}"
                                ) from exc
                            for binding in _namespace_bindings(document, logical):
                                namespace = binding["namespace"]
                                namespace_seed_values[namespace].update(binding["seed_values"])
                                namespace_binding_sources[namespace].add(logical)
                    elif path.suffix.lower() in {".npy", ".npz"} and _is_dit_path(relative):
                        numpy_record, numpy_hits = _inspect_numpy_container(
                            path,
                            logical_source=logical,
                            seeds=seeds,
                            byte_pattern=byte_pattern,
                            text_pattern=text_pattern,
                        )
                        record.update(
                            {
                                "scan_class": "dit_numpy_header_and_seed_capable_arrays",
                                "numpy_inventory_index": len(numpy_records),
                            }
                        )
                        numpy_records.append(numpy_record)
                        hits.extend(numpy_hits)
                        counters["dit_numpy_containers_inspected"] += 1
                        counters["dit_numpy_arrays_header_inspected"] += len(
                            numpy_record["arrays"]
                        )
                        counters["dit_numpy_arrays_body_inspected"] += sum(
                            array["seed_scan_class"]
                            in {"integer", "string", "structured"}
                            for array in numpy_record["arrays"]
                        )
                    else:
                        record["scan_class"] = "path_and_inventory_only"
                        counters["other_files_inventoried"] += 1
                    inventory_records.append(record)
                    counters["files_inventoried"] += 1
                    counters["file_bytes_inventoried"] += int(record["bytes"])
                except Exception as exc:  # noqa: BLE001 - collect all unreadables.
                    unreadable.append({"source": logical, "error": repr(exc)})

    inventory_records.sort(
        key=lambda item: (item["root_label"], item["relative_path"], item["entry_type"])
    )
    excluded_records.sort(
        key=lambda item: (item["root_label"], item["relative_path"], item["entry_type"])
    )
    numpy_records.sort(key=lambda item: item["container"])
    hits.sort(key=lambda item: (item["seed"], item["source"], item.get("offset", -1)))
    unreadable.sort(key=lambda item: item["source"])
    namespaces = []
    for namespace in sorted(set(namespace_sources).union(namespace_seed_values)):
        namespaces.append(
            {
                "namespace": namespace,
                "text_occurrence_sources": sorted(namespace_sources[namespace]),
                "json_namespace_binding_sources": sorted(
                    namespace_binding_sources[namespace]
                ),
                "discovered_seed_values": sorted(namespace_seed_values[namespace]),
                "candidate_intersection": sorted(
                    set(seeds).intersection(namespace_seed_values[namespace])
                ),
            }
        )
    return {
        "inventory_records": inventory_records,
        "excluded_records": excluded_records,
        "numpy_records": numpy_records,
        "hits": hits,
        "unreadable": unreadable,
        "discovered_namespaces": namespaces,
        "counters": dict(sorted(counters.items())),
    }


def _write_artifact(
    output_dir: Path,
    *,
    inventory: dict[str, Any],
    report: dict[str, Any],
    completion_base: dict[str, Any],
) -> Path:
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir.exists() or output_dir.is_symlink():
        raise AuditFailure(f"output already exists; no-overwrite enforced: {output_dir}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=parent))
    try:
        _add_self_hash(inventory, "inventory_identity_sha256")
        _atomic_json(stage / "inventory.json", inventory)
        inventory_file_sha = _sha256_file(stage / "inventory.json")
        report["inventory_binding"] = {
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "inventory_file_sha256": inventory_file_sha,
        }
        _add_self_hash(report, "payload_sha256")
        _atomic_json(stage / "report.json", report)
        report_file_sha = _sha256_file(stage / "report.json")
        completion = dict(completion_base)
        completion.update(
            {
                "inventory_identity_sha256": inventory["inventory_identity_sha256"],
                "inventory_file_sha256": inventory_file_sha,
                "report_payload_sha256": report["payload_sha256"],
                "report_file_sha256": report_file_sha,
            }
        )
        _add_self_hash(completion, "payload_sha256")
        _atomic_json(stage / "completion.json", completion)
        if output_dir.exists() or output_dir.is_symlink():
            raise AuditFailure(f"output appeared during staging: {output_dir}")
        os.rename(stage, output_dir)
        return output_dir
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_artifact(root: Path) -> dict[str, Any]:
    root = Path(os.path.abspath(root))
    if not root.is_dir():
        raise AuditFailure(f"artifact is not a directory: {root}")
    observed_files = sorted(path.name for path in root.iterdir())
    expected_files = sorted(SELF_HASH_KEYS)
    if observed_files != expected_files:
        raise AuditFailure(
            f"artifact file set mismatch: observed={observed_files}, expected={expected_files}"
        )
    inventory = _read_self_hashed_json(
        root / "inventory.json", "inventory_identity_sha256"
    )
    report = _read_self_hashed_json(root / "report.json", "payload_sha256")
    completion = _read_self_hashed_json(root / "completion.json", "payload_sha256")
    if completion.get("status") != "PASS" or report.get("status") != "PASS":
        raise AuditFailure("artifact does not certify PASS")
    if completion.get("inventory_file_sha256") != _sha256_file(root / "inventory.json"):
        raise AuditFailure("completion inventory file hash mismatch")
    if completion.get("report_file_sha256") != _sha256_file(root / "report.json"):
        raise AuditFailure("completion report file hash mismatch")
    if completion.get("inventory_identity_sha256") != inventory.get(
        "inventory_identity_sha256"
    ):
        raise AuditFailure("completion inventory identity mismatch")
    if completion.get("report_payload_sha256") != report.get("payload_sha256"):
        raise AuditFailure("completion report identity mismatch")
    if report.get("inventory_binding") != {
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "inventory_file_sha256": _sha256_file(root / "inventory.json"),
    }:
        raise AuditFailure("report inventory binding mismatch")
    if report.get("finding", {}).get("prior_materialization_hit_count") != 0:
        raise AuditFailure("report contains a prior-materialization hit")
    if report.get("finding", {}).get("unreadable_count") != 0:
        raise AuditFailure("report contains an unreadable entry")
    return {
        "artifact": str(root),
        "status": "PASS",
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "report_payload_sha256": report["payload_sha256"],
        "report_file_sha256": _sha256_file(root / "report.json"),
        "completion_payload_sha256": completion["payload_sha256"],
    }


def run_audit(
    *,
    protocol_path: Path,
    repo_root: Path,
    evidence_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    protocol_path = Path(os.path.abspath(protocol_path))
    repo_root = Path(os.path.abspath(repo_root))
    evidence_root = Path(os.path.abspath(evidence_root))
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir.exists() or output_dir.is_symlink():
        raise AuditFailure(f"output already exists; no-overwrite enforced: {output_dir}")
    script_path = Path(os.path.abspath(__file__))
    script_sha_before = _sha256_file(script_path)
    protocol, protocol_raw, seeds, protocol_checks = _validate_protocol(protocol_path)
    protocol_file_sha = _sha256_bytes(protocol_raw)
    planned_runner_path = repo_root / PLANNED_RUNNER_DEFAULT
    try:
        planned_runner_raw = planned_runner_path.read_bytes()
    except OSError as exc:
        raise AuditFailure(f"cannot snapshot planned runner {planned_runner_path}: {exc}") from exc
    planned_runner_sha = _sha256_bytes(planned_runner_raw)
    started_utc = _utc_now()
    prior_audit_root = evidence_root / "dit_imagenet256_t60_cross_prefix_seed_audit"
    directory_exclusions = (
        {
            prior_audit_root: "prior seed-audit self-record tree contains declared candidate values but no sampler execution artifacts"
        }
        if prior_audit_root.exists()
        else {}
    )
    scan = _scan_roots(
        (("repository", repo_root), ("cross_scale_evidence", evidence_root)),
        planned_source_exclusions={
            protocol_path: "planned_candidate_protocol_declaration_only",
            planned_runner_path: "unexecuted_candidate_runner_source_derives_or_reads_planned_seed_slate_only",
        },
        planned_directory_exclusions=directory_exclusions,
        seeds=seeds,
    )
    if protocol_path.read_bytes() != protocol_raw:
        raise AuditFailure("protocol changed during the audit")
    if _sha256_file(script_path) != script_sha_before:
        raise AuditFailure("audit script changed during the audit")
    if planned_runner_path.read_bytes() != planned_runner_raw:
        raise AuditFailure("planned runner changed during the audit")
    if scan["unreadable"]:
        preview = scan["unreadable"][:5]
        raise AuditFailure(f"unreadable audit inputs ({len(scan['unreadable'])}): {preview}")
    if scan["hits"]:
        preview = scan["hits"][:5]
        raise AuditFailure(
            f"candidate seed materialization hits ({len(scan['hits'])}): {preview}"
        )
    namespace_collisions = [
        record
        for record in scan["discovered_namespaces"]
        if record["candidate_intersection"]
    ]
    if namespace_collisions:
        raise AuditFailure(f"candidate namespace-value collisions: {namespace_collisions}")

    declared_namespaces = {
        record["namespace"]: record
        for record in protocol_checks["known_namespace_records"]
    }
    discovered_namespaces = {
        record["namespace"]: record for record in scan["discovered_namespaces"]
    }
    declared_reconciliation = []
    for namespace, declared in sorted(declared_namespaces.items()):
        discovered = discovered_namespaces.get(namespace)
        discovered_values = (
            discovered["discovered_seed_values"] if discovered is not None else []
        )
        expected_record = next(
            record
            for record in protocol["seed_lineage"]["draft_exclusion_snapshot"][
                "existing_DiT_branch_local_namespaces"
            ]
            if record["namespace"] == namespace
        )
        expected_values = sorted(expected_record["values"])
        missing = sorted(set(expected_values).difference(discovered_values))
        extra = sorted(set(discovered_values).difference(expected_values))
        # The on-disk lineage should account for every value frozen in the draft.
        if missing:
            raise AuditFailure(
                f"declared namespace values absent from audited lineage for {namespace}: {missing}"
            )
        declared_reconciliation.append(
            {
                "namespace": namespace,
                "declared_value_count": len(expected_values),
                "discovered_value_count": len(discovered_values),
                "declared_values_all_materialized_in_audited_lineage": True,
                "additional_discovered_seed_values": extra,
                "candidate_intersection": [],
            }
        )

    inventory_core = {
        "format": INVENTORY_FORMAT,
        "roots": [
            {"label": "repository", "path": str(repo_root)},
            {"label": "cross_scale_evidence", "path": str(evidence_root)},
        ],
        "inventory_definition": {
            "ordering": "root_label, relative_path, entry_type; all ascending",
            "file_records": "path, type, byte size and scan class; scanned text also has content SHA-256; DiT NPY/NPZ has deterministic header/member metadata and SHA-256 for each seed-capable array body",
            "unrelated_binary_boundary": "non-text, non-DiT-NPY/NPZ payloads are path/size inventoried but their bytes are not searched or content-hashed",
            "symlink_boundary": "file symlinks are opened for classified content; directory symlinks are inventoried but not followed to avoid leaving the declared roots or cycles",
            "numpy_boundary": "all arrays in every path-relevant DiT NPY/NPZ receive header/size inspection; candidate-capable integer, fixed string, and structured primitive bodies are streamed; object arrays fail closed; large irrelevant float/low-range integer image or latent bodies remain unread",
        },
        "records": scan["inventory_records"],
        "numpy_containers": scan["numpy_records"],
        "excluded_entries": scan["excluded_records"],
        "counters": scan["counters"],
    }
    inventory_hash = _sha256_bytes(_canonical_json_bytes(inventory_core))
    inventory = dict(inventory_core)
    inventory["inventory_records_sha256"] = inventory_hash

    finished_utc = _utc_now()
    target_namespace = protocol["seed_lineage"]["derivation"]["namespace"]
    stable_scope_counts = {
        "root_count": 2,
        "files_inventoried": scan["counters"].get("files_inventoried", 0),
        "file_bytes_inventoried": scan["counters"].get(
            "file_bytes_inventoried", 0
        ),
        "text_files_scanned": scan["counters"].get("text_files_scanned", 0),
        "text_bytes_scanned": scan["counters"].get("text_bytes_scanned", 0),
        "dit_numpy_containers_inspected": scan["counters"].get(
            "dit_numpy_containers_inspected", 0
        ),
        "dit_numpy_arrays_header_inspected": scan["counters"].get(
            "dit_numpy_arrays_header_inspected", 0
        ),
        "dit_numpy_arrays_body_inspected": scan["counters"].get(
            "dit_numpy_arrays_body_inspected", 0
        ),
    }
    stable_scope_flags = {
        "all_visited_paths_exact_decimal_scanned": True,
        "all_classified_text_exact_decimal_scanned": True,
        "prior_run_manifests_results_logs_annotations_and_code_in_scope": True,
        "review_and_export_ledgers_in_scope": True,
        "path_relevant_dit_npy_npz_headers_inspected": True,
        "dit_numpy_seed_capable_integer_string_metadata_bodies_inspected": True,
        "candidate_protocol_declaration_content_excluded": True,
        "unexecuted_candidate_runner_source_content_excluded": True,
        "prior_seed_audit_self_record_tree_excluded_if_present": True,
        "gpu_used": False,
    }
    zero_hit_counts = {
        "path": 0,
        "text": 0,
        "numpy": 0,
        "ledger": 0,
        "total": 0,
    }
    report = {
        "format": FORMAT,
        "status": "PASS",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "intended_use": "pre-freeze procedural novelty/materialization gate for 64 branch-local full DiT trajectories",
        "audited_seed_binding": {
            "candidate_seed_count": 64,
            "candidate_seed_list_sha256": protocol_checks["seed_list_sha256"],
            "derivation_namespace": target_namespace,
            "derivation_exact_match": True,
            "binding_role": "stable audit identity; later non-seed protocol edits do not invalidate this seed-lineage audit",
        },
        "protocol_source_snapshot_nonbinding": {
            "path": str(protocol_path),
            "file_sha256": protocol_file_sha,
            "protocol_identity_sha256": protocol["protocol_identity_sha256"],
            "protocol_status": protocol.get("protocol_status"),
            "role": "provenance snapshot at audit time, not a required final-protocol identity binding",
        },
        "audit_source_binding": {
            "path": str(script_path),
            "sha256": script_sha_before,
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "candidate_seed_checks": protocol_checks,
        "candidate_seed_slate": {
            "target_namespace": target_namespace,
            "values_in_trajectory_index_order": seeds,
            "planned_protocol_occurrence_excluded": True,
        },
        "scope": {
            "roots": inventory_core["roots"],
            "excluded_directory_component_names": sorted(
                EXCLUDED_DIRECTORY_COMPONENTS
            ),
            "planned_source_content_exclusions": [
                {
                    "path": str(protocol_path),
                    "reason": "planned candidate protocol declaration only",
                    "sha256": protocol_file_sha,
                },
                {
                    "path": str(repo_root / PLANNED_RUNNER_DEFAULT),
                    "reason": "unexecuted candidate runner source derives or reads the planned slate; it is not a run ledger",
                    "sha256": planned_runner_sha,
                },
            ],
            "planned_artifact_directory_exclusions": [
                {
                    "path": str(path),
                    "reason": reason,
                }
                for path, reason in sorted(
                    directory_exclusions.items(), key=lambda item: str(item[0])
                )
            ],
            "text_definition": {
                "suffixes": sorted(TEXT_SUFFIXES),
                "special_basenames": sorted(TEXT_BASENAMES),
                "matching": "exact candidate decimal with ASCII digit boundaries; BOM-declared UTF-16 is decoded equivalently",
            },
            "path_matching": "all visited file/directory relative paths and symlink targets, exact candidate decimal with digit boundaries",
            "dit_numpy_relevance": "NPY/NPZ whose relative path has a delimiter-bounded token 'dit' (case-insensitive)",
            "numpy_inspection": inventory_core["inventory_definition"][
                "numpy_boundary"
            ],
            "counters": scan["counters"],
            "stable_counts": stable_scope_counts,
            "stable_flags": stable_scope_flags,
            "excluded_entry_count": len(scan["excluded_records"]),
        },
        "known_dit_namespace_audit": {
            "declared_branch_local_reconciliation": declared_reconciliation,
            "all_discovered_namespaces": scan["discovered_namespaces"],
            "all_candidate_intersections_empty": True,
        },
        "finding": {
            "hit_counts": zero_hit_counts,
            "prior_materialization_hit_count": 0,
            "path_hit_count": 0,
            "text_hit_count": 0,
            "numpy_hit_count": 0,
            "unreadable_count": 0,
            "candidate_duplicate_count": 0,
            "candidate_zero_count": 0,
            "candidate_known_value_collision_count": 0,
            "candidate_namespace_value_collision_count": 0,
        },
        "certification": {
            "statement": "Within the explicitly audited mounted filesystem lineage, no prior generator-consumption, render, export, review, annotation, manifest, configuration, log, code, path, or inspected DiT NumPy metadata record containing any of the 64 candidate branch-local seed decimals was found.",
            "planned_values_exclusion": "The values in the candidate protocol are intentionally excluded because they declare the future slate and are not a consumption/materialization record.",
            "procedural_limitation": "Absence from this audited filesystem lineage is not cryptographic proof that a value never existed elsewhere, on an unmounted/deleted system, or outside the two declared roots.",
            "semantic_limitation": "The audit certifies absence of exact 64-bit candidate seed values in the declared lineage; it does not prove pseudorandom-stream disjointness under every possible alternative RNG or seed-transform convention.",
        },
    }
    completion_base = {
        "format": COMPLETION_FORMAT,
        "status": "PASS",
        "completed_utc": finished_utc,
        "protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "protocol_snapshot_role": "nonbinding provenance only",
        "audit_source_sha256": script_sha_before,
        "candidate_seed_list_sha256": protocol_checks["seed_list_sha256"],
        "inventory_records_sha256": inventory_hash,
        "scope_counts": stable_scope_counts,
        "scope_flags": stable_scope_flags,
        "hit_counts": zero_hit_counts,
        "prior_materialization_hit_count": 0,
        "unreadable_count": 0,
        "gpu_used": False,
        "filesystem_scan_read_only": True,
    }
    final_root = _write_artifact(
        output_dir,
        inventory=inventory,
        report=report,
        completion_base=completion_base,
    )
    return validate_artifact(final_root)


def selftest() -> None:
    candidate = 712345678901234567
    other = 812345678901234569
    byte_pattern, text_pattern = _seed_token_patterns([candidate, other])
    data = (
        f"x={candidate}; embedded=9{candidate}; suffix={candidate}7; y={other}\n"
    ).encode("ascii")
    matches = _match_candidate_text(data, byte_pattern, text_pattern)
    if [value for value, _ in matches] != [candidate, other]:
        raise AuditFailure(f"decimal boundary selftest failed: {matches}")

    with tempfile.TemporaryDirectory(prefix="eqvae-seed-audit-selftest-") as raw:
        root = Path(raw)
        repo = root / "repo"
        evidence = root / "evidence"
        repo.mkdir()
        dit_dir = evidence / "dit_case"
        dit_dir.mkdir(parents=True)
        (repo / "clean.json").write_text('{"seed": 1}\n', encoding="utf-8")
        np.save(dit_dir / "seed_values.npy", np.array([1, candidate], dtype=np.int64))
        scan = _scan_roots(
            (("repository", repo), ("cross_scale_evidence", evidence)),
            planned_source_exclusions={},
            planned_directory_exclusions={},
            seeds=[candidate, other],
        )
        if len(scan["hits"]) != 1 or scan["hits"][0]["seed"] != candidate:
            raise AuditFailure(f"NumPy positive-control selftest failed: {scan['hits']}")
        np.save(dit_dir / "seed_values.npy", np.array([1, 2], dtype=np.int64))
        clean_scan = _scan_roots(
            (("repository", repo), ("cross_scale_evidence", evidence)),
            planned_source_exclusions={},
            planned_directory_exclusions={},
            seeds=[candidate, other],
        )
        if clean_scan["hits"] or clean_scan["unreadable"]:
            raise AuditFailure("clean NumPy selftest did not pass")

        artifact = root / "artifact"
        inventory = {
            "format": INVENTORY_FORMAT,
            "inventory_records_sha256": _sha256_bytes(_canonical_json_bytes([])),
            "records": [],
        }
        report = {
            "format": FORMAT,
            "status": "PASS",
            "finding": {
                "prior_materialization_hit_count": 0,
                "unreadable_count": 0,
            },
        }
        completion = {"format": COMPLETION_FORMAT, "status": "PASS"}
        _write_artifact(
            artifact,
            inventory=inventory,
            report=report,
            completion_base=completion,
        )
        validate_artifact(artifact)
        try:
            _write_artifact(
                artifact,
                inventory={},
                report={},
                completion_base={},
            )
        except AuditFailure:
            pass
        else:
            raise AuditFailure("no-overwrite selftest failed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate", type=Path, metavar="ARTIFACT_DIR")
    parser.add_argument("--selftest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.selftest:
            selftest()
            print("SELFTEST PASS")
            return 0
        if args.validate is not None:
            print(json.dumps(validate_artifact(args.validate), indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise AuditFailure("--output-dir is required for a real audit")
        result = run_audit(
            protocol_path=args.protocol,
            repo_root=args.repo_root,
            evidence_root=args.evidence_root,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except AuditFailure as exc:
        print(f"AUDIT FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
