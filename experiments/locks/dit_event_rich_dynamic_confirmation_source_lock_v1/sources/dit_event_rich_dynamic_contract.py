#!/usr/bin/env python3
"""Shared fail-closed contracts for event-rich DiT dynamic confirmation.

This module contains no model, label, or score computation.  It validates the
immutable v3 scientific protocol, the post-anchor dynamic plan, exact pair
axes, and small cryptographic artifact envelopes used by the trace, candidate
product, and evaluator programs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVENT_PROTOCOL_LOCK = (
    ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v3"
)
DEFAULT_ENDPOINT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_event_rich_endpoint_sampling_source_lock_v1"
)
DEFAULT_DYNAMIC_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_event_rich_dynamic_confirmation_source_lock_v1"
)

RNG_DOMAIN = "eqvae.dit.event-rich.endpoint.v1"
RNG_MODULUS = 1 << 63
B_CANDIDATE = "B_blur_mean"
C_CANDIDATE = "C_c3_low_jump"
CANDIDATES = (B_CANDIDATE, C_CANDIDATE)
B_FEATURE = "decoded_local_blur_severity__mean"
C_FEATURE = (
    "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
)
B_CHECKPOINTS = (69, 79, 89, 99, 109, 119, 129, 139, 149)
C_CHECKPOINTS = tuple(range(100, 150))
CALIBRATION_SEEDS = tuple(range(1100, 1120))
CONFIRMATION_SEEDS = tuple(range(1200, 1328))
PHASE_SEEDS = {
    "calibration": CALIBRATION_SEEDS,
    "confirmation": CONFIRMATION_SEEDS,
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_EXTERNAL_TOKENS = (
    "inception",
    "dino",
    "fid",
    "embedding",
    "mahalanobis",
    "clip_score",
)
FORBIDDEN_SCORE_COLUMN_TOKENS = (
    "label",
    "severity",
    "clear_bad",
    "clean_good",
    "review",
    "consensus",
    "phenotype",
    "inception",
    "dino",
    "fid",
    "embedding",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def require_hex64(value: Any, description: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimeError(f"{description} must be lowercase 64-hex SHA-256")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_json_exclusive(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact contains symlink: {path}")
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        records.append(
            {
                "name": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def manifest_map(manifest: Mapping[str, Any], description: str) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{description} manifest file list is malformed")
    result = {str(row.get("name")): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{description} manifest repeats a filename")
    return result


def validate_manifest_tree(
    root: Path,
    *,
    expected_status: str = "complete",
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, "artifact root")
    manifest_path = require_regular(root / "manifest.json", "artifact manifest")
    completion_path = require_regular(root / "completion.json", "artifact completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = require_hex64(manifest.get("identity_sha256"), "manifest identity")
    if canonical_sha256(without_identity(manifest)) != identity:
        raise RuntimeError("manifest identity mismatch")
    if manifest.get("status") != expected_status or completion.get("complete") is not True:
        raise RuntimeError("artifact is not complete")
    if (
        completion.get("manifest_identity_sha256") != identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("completion does not bind manifest")
    listed = manifest_map(manifest, "artifact")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "completion.json"}
    }
    if set(listed) != observed:
        raise RuntimeError("artifact manifest member set differs from disk")
    for name, record in listed.items():
        path = require_regular(root / name, f"artifact member {name}")
        if record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"artifact member changed: {name}")
    return manifest, completion


def publish_artifact(
    output: Path,
    *,
    artifact_kind: str,
    payloads: Mapping[str, bytes | str],
    manifest_fields: Mapping[str, Any],
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for relative, payload in payloads.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                path.write_text(payload, encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "artifact_kind": artifact_kind,
            **dict(manifest_fields),
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "artifact_kind": artifact_kind,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
        }
        write_json(staging / "completion.json", completion)
        validate_manifest_tree(staging)
        os.replace(staging, output)
        validate_manifest_tree(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output.resolve()


def validate_event_protocol(lock_root: Path) -> dict[str, Any]:
    lock_root = require_directory(lock_root, "event-rich v3 protocol lock")
    manifest, completion = validate_manifest_tree(lock_root)
    protocol_path = require_regular(lock_root / "protocol.json", "event protocol")
    protocol = load_json(protocol_path)
    identity = require_hex64(protocol.get("identity_sha256"), "event protocol identity")
    if canonical_sha256(without_identity(protocol)) != identity:
        raise RuntimeError("event protocol identity mismatch")
    if (
        protocol.get("schema_version") != 3
        or protocol.get("status")
        != "FROZEN_BEFORE_REVIEWER_QUALIFICATION_OR_EVENT_RICH_SCREEN"
        or manifest.get("protocol_identity_sha256") != identity
        or completion.get("protocol_identity_sha256") != identity
    ):
        raise RuntimeError("not the frozen event-rich v3 scientific protocol")
    candidates = protocol.get("candidates", {})
    confirmation = protocol.get("confirmation", {})
    rng = protocol.get("endpoint_screen", {}).get("batch_rng_contract", {})
    boundary = protocol.get("method_boundary", {})
    if (
        tuple(candidates) != CANDIDATES
        or candidates[B_CANDIDATE].get("feature") != B_FEATURE
        or tuple(candidates[B_CANDIDATE].get("checkpoint_sampling_steps", ()))
        != B_CHECKPOINTS
        or candidates[B_CANDIDATE].get("orientation") != "bad_high"
        or candidates[C_CANDIDATE].get("feature") != C_FEATURE
        or candidates[C_CANDIDATE].get("orientation") != "bad_low"
        or tuple(confirmation.get("calibration_seeds", ())) != CALIBRATION_SEEDS
        or tuple(confirmation.get("confirmation_seeds", ())) != CONFIRMATION_SEEDS
        or confirmation.get("candidate_columns_physically_separate_before_any_label_join")
        is not True
        or rng.get("domain") != RNG_DOMAIN
        or rng.get("classes_per_invocation") != 1
        or rng.get("same_global_seed_classes_share_initial_or_transition_innovation")
        is not False
        or boundary.get("internal_method_candidates_only") != list(CANDIDATES)
    ):
        raise RuntimeError("event protocol dynamic-confirmation contract changed")
    return protocol


def derive_pair_seed(global_seed: int, class_id: int) -> int:
    if type(global_seed) is not int or global_seed < 0:
        raise ValueError("global_seed must be a non-negative integer")
    if type(class_id) is not int or not 0 <= class_id < 1000:
        raise ValueError("class_id must lie in [0,999]")
    payload = f"{RNG_DOMAIN}\0{global_seed}\0{class_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % RNG_MODULUS


def _int_list(value: Any, description: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise RuntimeError(f"{description} must be an integer list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise RuntimeError(f"{description} contains duplicates")
    return result


def validate_anchor_plan(path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    path = require_regular(path, "immutable post-anchor trace plan")
    plan = load_json(path)
    identity = require_hex64(plan.get("identity_sha256"), "anchor-plan identity")
    if canonical_sha256(without_identity(plan)) != identity:
        raise RuntimeError("anchor-plan identity mismatch")
    required = {
        "schema_version",
        "status",
        "protocol_identity_sha256",
        "selection_identity_sha256",
        "anchor_consensus_file_sha256",
        "B_decision",
        "C_decision",
        "active_union_classes",
        "calibration_seeds",
        "confirmation_seeds",
        "total_full_trace_rows",
        "candidate_products_must_be_physically_separate",
        "score_or_embedding_input_used",
        "identity_sha256",
    }
    if set(plan) != required:
        raise RuntimeError("anchor-plan schema changed")
    if (
        plan.get("schema_version") != 1
        or plan.get("status")
        != "PROSPECTIVE_TRACE_PLAN_LOCKED_AFTER_INDEPENDENT_ENDPOINT_ANCHOR"
        or plan.get("protocol_identity_sha256") != protocol.get("identity_sha256")
        or plan.get("candidate_products_must_be_physically_separate") is not True
        or plan.get("score_or_embedding_input_used") is not False
        or _int_list(plan.get("calibration_seeds"), "plan calibration seeds")
        != CALIBRATION_SEEDS
        or _int_list(plan.get("confirmation_seeds"), "plan confirmation seeds")
        != CONFIRMATION_SEEDS
    ):
        raise RuntimeError("anchor-plan frozen fields changed")
    roster = tuple(
        int(row["class_id"])
        for row in protocol["endpoint_screen"]["class_roster"]
    )
    roster_order = {class_id: index for index, class_id in enumerate(roster)}
    decisions: dict[str, dict[str, Any]] = {}
    for candidate, field in ((B_CANDIDATE, "B_decision"), (C_CANDIDATE, "C_decision")):
        decision = plan.get(field)
        if not isinstance(decision, dict):
            raise RuntimeError(f"anchor plan lacks {field}")
        selected = _int_list(decision.get("selected_classes"), f"{candidate} selected classes")
        if len(selected) != 6 or not set(selected) <= set(roster):
            raise RuntimeError(f"{candidate} must retain exactly six roster classes")
        if decision.get("candidate") != candidate or type(decision.get("go")) is not bool:
            raise RuntimeError(f"{candidate} decision schema changed")
        decisions[candidate] = dict(decision)
    expected_active = set(
        decisions[B_CANDIDATE]["selected_classes"]
        if decisions[B_CANDIDATE]["go"]
        else ()
    ) | set(
        decisions[C_CANDIDATE]["selected_classes"]
        if decisions[C_CANDIDATE]["go"]
        else ()
    )
    expected_active_ordered = tuple(sorted(expected_active, key=roster_order.__getitem__))
    active = _int_list(plan.get("active_union_classes"), "active union classes")
    if active != expected_active_ordered or len(active) > 12:
        raise RuntimeError("active union does not replay candidate GO decisions")
    expected_confirmation_rows = len(active) * len(CONFIRMATION_SEEDS)
    if plan.get("total_full_trace_rows") != expected_confirmation_rows:
        raise RuntimeError("anchor-plan confirmation row count changed")
    return plan


def candidate_classes(plan: Mapping[str, Any], candidate: str) -> tuple[int, ...]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    decision = plan["B_decision" if candidate == B_CANDIDATE else "C_decision"]
    return tuple(decision["selected_classes"]) if decision["go"] else ()


def exact_pairs(
    plan: Mapping[str, Any],
    *,
    candidate: str | None = None,
    phases: Iterable[str] = ("calibration", "confirmation"),
) -> tuple[tuple[str, int, int], ...]:
    classes = (
        tuple(plan["active_union_classes"])
        if candidate is None
        else candidate_classes(plan, candidate)
    )
    result: list[tuple[str, int, int]] = []
    for phase in phases:
        if phase not in PHASE_SEEDS:
            raise ValueError(f"unknown phase: {phase}")
        result.extend(
            (phase, seed, class_id)
            for seed in PHASE_SEEDS[phase]
            for class_id in classes
        )
    return tuple(result)


def pair_relative_directory(phase: str, global_seed: int, class_id: int) -> str:
    if phase not in PHASE_SEEDS or global_seed not in PHASE_SEEDS[phase]:
        raise ValueError("pair phase/seed is outside the frozen dynamic axis")
    derive_pair_seed(global_seed, class_id)
    return f"pairs/{phase}/seed{global_seed:04d}_class{class_id:04d}"


def reject_forbidden_external_name(value: str, description: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in FORBIDDEN_EXTERNAL_TOKENS):
        raise RuntimeError(f"{description} contains forbidden external-representation token")


def validate_score_columns(columns: Iterable[str], candidate: str) -> tuple[str, ...]:
    score = B_FEATURE if candidate == B_CANDIDATE else C_FEATURE
    expected = ("phase", "global_seed", "class_id", score)
    observed = tuple(columns)
    if observed != expected:
        raise RuntimeError(
            f"{candidate} score columns must be exactly {expected}; observed={observed}"
        )
    for column in observed:
        lowered = column.lower()
        if column != score and any(token in lowered for token in FORBIDDEN_SCORE_COLUMN_TOKENS):
            raise RuntimeError(f"forbidden supervised/external score column: {column}")
    return expected


def run_self_test() -> None:
    expected = {
        (1000, 0): 3026363209052735318,
        (1000, 1): 3606479167075842380,
        (1011, 999): 8394018843514802193,
    }
    if {pair: derive_pair_seed(*pair) for pair in expected} != expected:
        raise AssertionError("pair-keyed RNG known answers changed")
    validate_score_columns(
        ("phase", "global_seed", "class_id", B_FEATURE), B_CANDIDATE
    )
    poison = (
        "phase",
        "global_seed",
        "class_id",
        B_FEATURE,
        "label",
    )
    try:
        validate_score_columns(poison, B_CANDIDATE)
    except RuntimeError:
        pass
    else:
        raise AssertionError("label-column poison escaped score schema")
    for name in ("fid.csv", "DINO_distance.json", "inception_features.npz"):
        try:
            reject_forbidden_external_name(name, "synthetic input")
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"external-representation poison escaped: {name}")
    print("self-test passed: RNG, single-score schemas, and external-metric poisons")


if __name__ == "__main__":
    run_self_test()
