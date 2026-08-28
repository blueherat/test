#!/usr/bin/env python3
"""Fail-closed shared contract for the prospective scientific-v4 B/E study.

This module is deliberately free of model inference, endpoint inspection, visual
labels, and candidate values.  It validates immutable source/artifact envelopes,
the exact pair-keyed RNG axis, the blur-focused method lock, and the eventual
scientific-v4 protocol/trace plan.  The older event-rich B/C v3 namespace is not
accepted here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
METHOD_LOCK_ID = "cc4dc5e7c06c25f4d8567a42fb4f0387097a6296c587543830bfeaa4771f6921"
DEFAULT_METHOD_LOCK = ROOT / "experiments/locks/dit_blur_focused_eprocess_protocol_lock_v2_2"
SCIENTIFIC_PROTOCOL_ID = "af65e362cd8c543f898a3dbeb3a7b4478940966b48bfef689db7afdbad8d97d2"
DEFAULT_SCIENTIFIC_PROTOCOL_LOCK = ROOT / "experiments/locks/dit_event_rich_confirmation_protocol_lock_v4_2_1"
DEFAULT_DYNAMIC_SOURCE_LOCK = ROOT / "experiments/locks/dit_scientific_v4_2_1_be_dynamic_source_lock_v1"
DYNAMIC_SOURCE_STATUS = (
    "SCIENTIFIC_V4_2_1_B_E_DYNAMIC_SOURCES_FROZEN_EXECUTION_NOT_READY"
)
DYNAMIC_SOURCE_ARTIFACT_KIND = "SCIENTIFIC_V4_2_1_B_E_DYNAMIC_SOURCE_LOCK_V1"

RNG_DOMAIN = "eqvae.dit.event-rich.endpoint.v1"
RNG_MODULUS = 1 << 63
CHECKPOINTS = (69, 79, 89, 99, 109, 119, 129, 139, 149)
INTERNAL_TIMESTEPS = tuple(249 - step for step in CHECKPOINTS)
CALIBRATION_SEEDS = tuple(range(1100, 1120))
CONFIRMATION_SEEDS = tuple(range(1200, 1328))
PHASE_SEEDS = {"calibration": CALIBRATION_SEEDS, "confirmation": CONFIRMATION_SEEDS}
B_CANDIDATE = "B_persistence"
E_CANDIDATE = "E_blur_gated_running_max_log"
CANDIDATES = (B_CANDIDATE, E_CANDIDATE)
B_SCORE = "B_persistence"
E_SCORE = "E_blur_gated_running_max_log"
E_ALERT = "E_blur_gated_alarm"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

FORBIDDEN_METHOD_TOKENS = (
    "label",
    "review",
    "consensus",
    "adjudicat",
    "endpoint",
    "inception",
    "dino",
    "fid",
    "clip",
    "embedding",
    "mahalanobis",
    "quality_posterior",
)


def canonical_sha256(value: Any) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


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


def require_hex64(value: Any, description: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimeError(f"{description} must be lowercase 64-hex SHA-256")
    return value


def require_regular(path: Path, description: str) -> Path:
    absolute = path.expanduser().absolute()
    if not absolute.is_file() or absolute.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {absolute}")
    return absolute.resolve()


def require_directory(path: Path, description: str) -> Path:
    absolute = path.expanduser().absolute()
    if not absolute.is_dir() or absolute.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {absolute}")
    return absolute.resolve()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
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
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        # Only the artifact's own root envelope is implicit.  A nested source
        # lock/receipt named manifest.json or completion.json is payload and
        # must therefore be hashed like every other nested member.
        if relative in {"manifest.json", "completion.json"}:
            continue
        rows.append(
            {
                "name": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def manifest_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("manifest file list is malformed")
    result = {str(row.get("name")): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("manifest repeats a filename")
    return result


def validate_manifest_tree(
    root: Path, *, expected_status: str = "complete"
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
    listed = manifest_map(manifest)
    expected_directories: set[str] = set()
    for name in listed:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or name in {"", ".", "manifest.json", "completion.json"}
        ):
            raise RuntimeError(f"manifest member name is unsafe or reserved: {name}")
        parent = relative.parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"artifact contains symlink: {path}")
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix()
        not in {"manifest.json", "completion.json"}
    }
    if set(listed) != observed:
        raise RuntimeError("manifest member set differs from exact disk tree")
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    if observed_directories != expected_directories:
        raise RuntimeError("artifact directory set differs from manifest-implied tree")
    for name, record in listed.items():
        path = require_regular(root / name, f"artifact member {name}")
        if set(record) != {"name", "bytes", "sha256"}:
            raise RuntimeError(f"manifest record schema changed: {name}")
        if record["bytes"] != path.stat().st_size or record["sha256"] != sha256_file(path):
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


def validate_method_lock(path: Path = DEFAULT_METHOD_LOCK) -> tuple[dict[str, Any], dict[str, Any]]:
    path = require_directory(path, "blur-focused method lock")
    manifest_path = require_regular(path / "manifest.json", "method manifest")
    completion_path = require_regular(path / "completion.json", "method completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("method lock file records are missing")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "bytes", "sha256"}:
            raise RuntimeError("method lock file record schema changed")
        relative = str(row["relative_path"])
        if relative in observed:
            raise RuntimeError("method lock repeats a file")
        observed.add(relative)
        member = require_regular(path / relative, f"method member {relative}")
        if row["bytes"] != member.stat().st_size or row["sha256"] != sha256_file(member):
            raise RuntimeError(f"method lock member changed: {relative}")
    actual = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
        and item.relative_to(path).as_posix()
        not in {"manifest.json", "completion.json"}
    }
    if observed != actual:
        raise RuntimeError("method lock exact tree changed")
    file_list_identity = canonical_sha256(rows)
    # The method locker uses this same canonical JSON identity convention.
    if (
        manifest.get("files_sha256") != file_list_identity
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
    ):
        raise RuntimeError("method manifest or file-list identity is invalid")
    protocol = load_json(require_regular(path / "protocol.json", "method protocol"))
    if (
        manifest.get("identity_sha256") != METHOD_LOCK_ID
        or completion.get("identity_sha256") != METHOD_LOCK_ID
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or sha256_file(path / "protocol.json")
        != next(row["sha256"] for row in rows if row["relative_path"] == "protocol.json")
        or protocol.get("protocol_name") != "dit_blur_latched_directional_eprocess_v2_2"
        or protocol.get("schema_version") != 2
        or protocol.get("status") != "METHOD_PROTOCOL_ONLY_NOT_EXECUTION_READY"
        or protocol.get("execution_ready") is not False
        or manifest.get("matched_q_power_gate_identity")
        != "ae284448a324349488ab1be3962502d5450d006a64722bb717f5199903c6e6b2"
        or manifest.get("adaptive_null_audit_identity")
        != "4b69c132d39a70e615fc60ec12709daff670f15409a61c4f12e543f43fb7162c"
    ):
        raise RuntimeError("wrong or modified blur-latched B/E method-v2.2 lock")
    if tuple(protocol["observation_window"]["sampling_steps"]) != CHECKPOINTS:
        raise RuntimeError("method observation checkpoints changed")
    if tuple(protocol["candidate_family"]["co_primary"][index]["id"] for index in range(2)) != CANDIDATES:
        raise RuntimeError("method co-primary family is not exactly B/E")
    mechanics = protocol.get("pre_label_confirmation_path_mechanics_gate")
    if (
        not isinstance(mechanics, dict)
        or mechanics.get("minimum_confirmation_paths") != 768
        or mechanics.get("minimum_qualifying_started_paths_per_scale") != 12
        or mechanics.get("minimum_qualifying_started_classes_per_scale") != 3
        or mechanics.get("minimum_complete_coverage_fraction_among_started_paths")
        != 1.0
        or mechanics.get(
            "maximum_last_valid_fallback_fraction_among_started_steps_per_scale"
        )
        != 0.01
    ):
        raise RuntimeError("method-v2.2 confirmation mechanics gate changed")
    forbidden = tuple(str(item).lower() for item in protocol.get("forbidden_method_inputs", ()))
    if not any("endpoint" in item for item in forbidden) or not any("label" in item for item in forbidden):
        raise RuntimeError("method lock lost its external-input prohibition")
    return manifest, protocol


def validate_scientific_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the prospective v4 envelope and its B/E compatibility fields.

    The v4 protocol owner controls the complete schema.  This consumer checks
    only fields that are scientific dependencies of the B/E dynamic pipeline,
    while exact-tree validation prevents silent edits elsewhere.
    """

    manifest, completion = validate_manifest_tree(path)
    protocol = load_json(require_regular(path / "protocol.json", "scientific-v4 protocol"))
    identity = require_hex64(protocol.get("identity_sha256"), "scientific protocol identity")
    if canonical_sha256(without_identity(protocol)) != identity:
        raise RuntimeError("scientific protocol identity mismatch")
    if (
        protocol.get("schema_version") != 4
        or protocol.get("status")
        != "SCIENTIFIC_V4_2_1_CLAIM_LIMITED_FROZEN_EXECUTION_NOT_READY"
        or identity != SCIENTIFIC_PROTOCOL_ID
        or manifest.get("protocol_identity_sha256") != identity
        or completion.get("protocol_identity_sha256") != identity
    ):
        raise RuntimeError("not the frozen scientific-v4 protocol")
    encoded = json.dumps(protocol, ensure_ascii=False, sort_keys=True).lower()
    if METHOD_LOCK_ID not in encoded:
        raise RuntimeError("scientific v4 does not bind the blur-focused method lock")
    if "b_persistence" not in encoded or "e_blur_gated_running_max_log" not in encoded:
        raise RuntimeError("scientific v4 does not declare exactly the B/E method family")
    supersession = protocol.get("supersession", {})
    if not isinstance(supersession, dict) or supersession.get(
        "v3_B_C_execution_under_this_pool_forbidden"
    ) is not True:
        raise RuntimeError("scientific v4 does not explicitly supersede B/C v3 execution")
    return manifest, protocol


def validate_dynamic_contract_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the source-only v4.2.1 contract; it cannot authorize execution."""

    identity = require_hex64(contract.get("identity_sha256"), "dynamic contract identity")
    method = contract.get("method_lock")
    scientific = contract.get("scientific_protocol")
    endpoint = contract.get("endpoint_sampling_source_lock")
    review = contract.get("review_pipeline_source_lock")
    firewall = contract.get("method_input_firewall")
    activation = contract.get("execution_activation_requirements")
    evidence = contract.get("evidence_access_audit")
    if (
        canonical_sha256(without_identity(contract)) != identity
        or contract.get("schema_version") != 1
        or contract.get("status") != DYNAMIC_SOURCE_STATUS
        or contract.get("artifact_kind") != DYNAMIC_SOURCE_ARTIFACT_KIND
        or contract.get("scientific_revision") != "v4.2.1"
        or contract.get("execution_ready") is not False
        or not isinstance(method, dict)
        or method.get("identity_sha256") != METHOD_LOCK_ID
        or not isinstance(scientific, dict)
        or scientific.get("identity_sha256") != SCIENTIFIC_PROTOCOL_ID
        or contract.get("scientific_protocol_validated_identity_sha256")
        != SCIENTIFIC_PROTOCOL_ID
        or not isinstance(endpoint, dict)
        or require_hex64(
            endpoint.get("sampling_protocol_identity_sha256"),
            "endpoint sampling protocol identity",
        )
        != endpoint.get("sampling_protocol_identity_sha256")
        or not isinstance(review, dict)
        or require_hex64(
            review.get("review_contract_identity_sha256"),
            "review contract identity",
        )
        != review.get("review_contract_identity_sha256")
        or not isinstance(firewall, dict)
        or firewall.get("external_visual_labels_used_for_evaluation_cohort_enrichment")
        is not True
        or firewall.get("external_representations_used_for_cohort_selection") is not False
        or firewall.get("external_inputs_used_by_B_or_E") is not False
        or firewall.get("FID_Inception_DINO_CLIP_embeddings_external_distances_forbidden")
        is not True
        or not isinstance(activation, dict)
        or activation.get("this_source_lock_alone_authorizes_sampling") is not False
        or activation.get("trace_plan_decision_go_alone_is_not_execution_authority")
        is not True
        or not isinstance(evidence, dict)
        or evidence.get("trace_plan_endpoint_trace_review_label_score_opened") is not False
        or evidence.get("FID_Inception_DINO_CLIP_embedding_or_external_distance_opened")
        is not False
        or evidence.get("real_GPU_sampling_or_evaluation_run") is not False
    ):
        raise RuntimeError("dynamic v4.2.1 source-only contract changed")
    return dict(contract)


def derive_pair_seed(global_seed: int, class_id: int) -> int:
    if type(global_seed) is not int or global_seed < 0:
        raise ValueError("global_seed must be a non-negative integer")
    if type(class_id) is not int or not 0 <= class_id < 1000:
        raise ValueError("class_id must lie in [0,999]")
    payload = f"{RNG_DOMAIN}\0{global_seed}\0{class_id}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % RNG_MODULUS


def validate_trace_plan(path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the post-anchor cohort plan without pretending labels were absent.

    Blind visual labels may enrich the *evaluation cohort*.  They are external
    evaluation design data, never an input to B, E, their calibration, alarm,
    or any eventual rollback rule.  Endpoint representations remain forbidden.
    """
    plan = load_json(require_regular(path, "post-anchor v4 trace plan"))
    identity = require_hex64(plan.get("identity_sha256"), "trace-plan identity")
    if canonical_sha256(without_identity(plan)) != identity:
        raise RuntimeError("trace-plan identity mismatch")
    required = {
        "schema_version",
        "artifact_kind",
        "status",
        "protocol_identity_sha256",
        "selection_identity_sha256",
        "anchor_consensus_file_sha256",
        "selected_classes",
        "aggregate_counts",
        "decision",
        "descriptive_only",
        "calibration_seeds",
        "confirmation_seeds",
        "calibration_trace_rows",
        "confirmation_trace_rows",
        "B_and_E_share_exact_selected_class_set",
        "external_visual_labels_used_only_for_cohort_enrichment_and_go",
        "method_score_threshold_intervention_or_external_representation_input_used",
        "identity_sha256",
    }
    if set(plan) != required:
        raise RuntimeError("v4 trace-plan schema changed")
    classes = plan.get("selected_classes")
    if (
        plan.get("schema_version") != 1
        or plan.get("artifact_kind") != "EVENT_RICH_BLUR_ANCHOR_PLAN_LOCK_V1"
        or plan.get("status") != "BLUR_ANCHOR_GO_DECISION_LOCKED_BEFORE_INTERNAL_TRACES"
        or plan.get("protocol_identity_sha256") != protocol.get("identity_sha256")
        or not isinstance(classes, list)
        or len(classes) != 6
        or any(type(item) is not int or not 0 <= item < 1000 for item in classes)
        or len(set(classes)) != len(classes)
        or tuple(plan.get("calibration_seeds", ())) != CALIBRATION_SEEDS
        or tuple(plan.get("confirmation_seeds", ())) != CONFIRMATION_SEEDS
        or plan.get("calibration_trace_rows") != len(classes) * len(CALIBRATION_SEEDS)
        or plan.get("confirmation_trace_rows") != len(classes) * len(CONFIRMATION_SEEDS)
        or plan.get("B_and_E_share_exact_selected_class_set") is not True
        or plan.get("external_visual_labels_used_only_for_cohort_enrichment_and_go") is not True
        or plan.get("method_score_threshold_intervention_or_external_representation_input_used")
        is not False
    ):
        raise RuntimeError("v4 trace plan is incompatible with the frozen B/E study")
    decision = plan.get("decision")
    if not isinstance(decision, dict) or decision.get("go") is not True:
        raise RuntimeError("anchor GO did not authorize internal B/E traces")
    return plan


def exact_pairs(
    plan: Mapping[str, Any], phases: Iterable[str] = ("calibration", "confirmation")
) -> tuple[tuple[str, int, int], ...]:
    classes = tuple(plan["selected_classes"])
    rows: list[tuple[str, int, int]] = []
    for phase in phases:
        if phase not in PHASE_SEEDS:
            raise ValueError(f"unknown phase: {phase}")
        rows.extend((phase, seed, class_id) for seed in PHASE_SEEDS[phase] for class_id in classes)
    return tuple(rows)


def fixed_no_touch_pair(plan: Mapping[str, Any]) -> tuple[str, int, int]:
    """Return the sole pre-registered observation no-touch audit pair.

    Fixing this before replay prevents choosing whichever in-axis pair happens
    to pass after seeing audit outcomes.
    """

    classes = tuple(plan["selected_classes"])
    if len(classes) != 6:
        raise ValueError("no-touch pair requires the validated six-class plan")
    return ("calibration", CALIBRATION_SEEDS[0], int(classes[0]))


def pair_relative_directory(phase: str, global_seed: int, class_id: int) -> str:
    if phase not in PHASE_SEEDS or global_seed not in PHASE_SEEDS[phase]:
        raise ValueError("pair phase/seed is outside the frozen v4 axis")
    derive_pair_seed(global_seed, class_id)
    return f"pairs/{phase}/seed{global_seed:04d}_class{class_id:04d}"


def reject_forbidden_method_name(value: str, description: str) -> None:
    lowered = value.lower()
    if any(token in lowered for token in FORBIDDEN_METHOD_TOKENS):
        raise RuntimeError(f"{description} contains a forbidden external/supervised token")


def require_exact_columns(
    columns: Sequence[str], expected: Sequence[str], description: str
) -> tuple[str, ...]:
    observed = tuple(columns)
    if observed != tuple(expected):
        raise RuntimeError(f"{description} columns changed: {observed}")
    return observed


def run_self_test() -> None:
    known = {
        (1000, 0): 3026363209052735318,
        (1000, 1): 3606479167075842380,
        (1011, 999): 8394018843514802193,
    }
    if {pair: derive_pair_seed(*pair) for pair in known} != known:
        raise AssertionError("pair-keyed RNG known answers changed")
    for poison in (
        "labels.csv",
        "endpoint.png",
        "inception_features.npz",
        "DINO_distance.json",
        "FID.json",
        "CLIP_embedding.npy",
    ):
        try:
            reject_forbidden_method_name(poison, "synthetic poison")
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"forbidden method input escaped: {poison}")
    require_exact_columns(
        ("phase", "global_seed", "class_id", B_SCORE, "B_alarm"),
        ("phase", "global_seed", "class_id", B_SCORE, "B_alarm"),
        "B product",
    )


if __name__ == "__main__":
    run_self_test()
    print("scientific-v4 B/E contract self-test passed")
