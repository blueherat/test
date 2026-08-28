#!/usr/bin/env python3
"""Two-stage, fail-closed evaluation of the frozen DiT third pool.

The source is frozen before any third-pool consensus or feature-product identity
exists.  Later, ``--bind-inputs`` may fill only absolute paths and exact manifest
identities; it exposes no scientific-analysis options.

Stage A opens only the bound consensus manifest, completion receipt, and
``aggregate_counts.json``.  It always stops after publishing an aggregate gate
receipt.  In particular, it never stats, hashes, or opens either feature-product
path, a consensus row table, a score CSV/NPZ, an image, or a screening result.

Stage B is a separate invocation.  It is authorized only by a complete Stage-A
receipt with both frozen event minima satisfied.  It then validates the complete
consensus and label-free products, performs the first score-label join, and emits
aggregate statistics only.  No row, score, rank, permutation draw, image, or
trajectory is ever emitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PHASE1_PROTOCOL_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_protocol_lock_v1"
)
PHASE1_THRESHOLD_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_threshold_lock_v1"
)
SAMPLING_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_sampling_source_lock_v1"
)
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_evaluation_source_lock_v5"
)

EXPECTED_PHASE1_PROTOCOL_IDENTITY = (
    "0788c7074adc55f1896dea4e0626f57c8b2b4899a5b600f38d2f982a87acfed5"
)
EXPECTED_PHASE1_PROTOCOL_MANIFEST_IDENTITY = (
    "931c928054da332dd83501e09a69793b5217596609e085b441b7cb993a325aa2"
)
EXPECTED_PHASE1_THRESHOLD_IDENTITY = (
    "c89fee87731968aa0c8a7ef086cb9a95a578dc3462149a6135bb71275bdbe43d"
)
EXPECTED_PHASE1_THRESHOLD_MANIFEST_IDENTITY = (
    "6b2c117b0dcc2eb3e2be71e1f4838ffe8b56206c8ffa0a557a6868692a732fb4"
)
EXPECTED_SAMPLING_PROTOCOL_IDENTITY = (
    "330661e87de7846e1f590660f03ecef6270fa45e2f39c4fc54d992e3260950d8"
)
EXPECTED_SAMPLING_MANIFEST_IDENTITY = (
    "eae86d48c1c1b9c732fbeea4838b2418b9b7261b61db0355fd7306469f5b6df3"
)

CLASSES = (207, 602, 795)
SEEDS = tuple(range(250, 850))
TRAJECTORY_COUNT = len(CLASSES) * len(SEEDS)
PRIMARY_FEATURE = (
    "pred_xstart_alpha_compensated_gradient_energy_c3__q2_max_positive_jump"
)
VISUAL_FEATURE = "decoded_local_blur_severity__mean"
PRIMARY_EXTRACTOR_SHA256 = (
    "acc348c7aa94ffe53d59ef4268c669fd948448de86f67202573bd04b69e9e129"
)
VISUAL_EXTRACTOR_SHA256 = (
    "452ae0e61fe36d027036e0d74c232fbcfbd7cb462d3749db92e062a104d0e398"
)

EVENT_MIN_BLUR = 15
EVENT_MIN_TOTAL_BAD = 30
PERMUTATION_DRAWS = 100_000
PERMUTATION_SEED = 2026082801
PERMUTATION_BATCH = 256
ALPHAS = ("alpha_0p10", "alpha_0p05")
CANDIDATES = ("B_blur_mean", "C_c3_low_jump")

CONSENSUS_EXPERIMENT = "dit_bad_good_third_pool_blind_consensus_v1"
CONSENSUS_AGGREGATE_NAME = "aggregate_counts.json"
CONSENSUS_ROWS_NAME = "consensus_rows.csv"
CONSENSUS_COLUMNS = (
    "sample_index",
    "global_seed",
    "class_slot",
    "class_id",
    "final_severity",
    "blur_component_consensus",
    "discrete_structure_component_consensus",
)
SEVERITIES = ("clean_good", "clear_bad", "mild_or_disputed")
COUNT_KEYS = (
    "trajectory_count",
    "clean_good",
    "clear_bad",
    "mild_or_disputed",
    "blur_or_soft_fusion_clear_bad",
    "mixed_blur_and_structure_clear_bad",
    "structural_non_blur_clear_bad",
    "phenotype_disputed_clear_bad",
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_CONSENSUS_MEMBER_TOKENS = (
    "score",
    "feature",
    "rank",
    "alert",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".npz",
    "screen",
)
FORBIDDEN_FEATURE_HEADER_TOKENS = (
    "severity",
    "clear_bad",
    "clean_good",
    "phenotype",
    "review",
    "consensus",
    "quality_label",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def require_regular(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_real_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def require_hex64(value: Any, description: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimeError(f"{description} must be a lowercase 64-hex SHA-256")
    return value


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"artifact contains a symlink: {path}")
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


def _manifest_map(manifest: Mapping[str, Any], description: str) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{description} manifest member list is malformed")
    result = {str(row.get("name")): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{description} manifest has duplicate member names")
    for name in result:
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"{description} manifest has unsafe member name: {name!r}")
    return result


def publish_record_lock(
    output: Path,
    *,
    artifact_kind: str,
    record_name: str,
    record: Mapping[str, Any],
    source_copies: Mapping[str, Path] | None = None,
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / record_name, record)
        for relative, source in (source_copies or {}).items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(require_regular(source, f"source copy {relative}"), destination)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "artifact_kind": artifact_kind,
            "primary_record_name": record_name,
            "primary_record_identity_sha256": record["identity_sha256"],
            "files": artifact_records(staging),
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "artifact_kind": artifact_kind,
            "primary_record_name": record_name,
            "primary_record_file_sha256": sha256_file(staging / record_name),
            "primary_record_identity_sha256": record["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
        }
        write_json(staging / "completion.json", completion)
        validate_record_lock(staging, artifact_kind=artifact_kind, record_name=record_name)
        os.replace(staging, output)
        validate_record_lock(output, artifact_kind=artifact_kind, record_name=record_name)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output.resolve()


def validate_record_lock(
    root: Path, *, artifact_kind: str, record_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_real_directory(root, artifact_kind)
    record_path = require_regular(root / record_name, f"{artifact_kind} primary record")
    manifest_path = require_regular(root / "manifest.json", f"{artifact_kind} manifest")
    completion_path = require_regular(
        root / "completion.json", f"{artifact_kind} completion"
    )
    record = load_json(record_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    record_identity = require_hex64(record.get("identity_sha256"), "record identity")
    manifest_identity = require_hex64(
        manifest.get("identity_sha256"), "manifest identity"
    )
    expected_completion = {
        "complete": True,
        "artifact_kind": artifact_kind,
        "primary_record_name": record_name,
        "primary_record_file_sha256": sha256_file(record_path),
        "primary_record_identity_sha256": record_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": manifest_identity,
    }
    if (
        canonical_sha256(without_identity(record)) != record_identity
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("artifact_kind") != artifact_kind
        or manifest.get("primary_record_name") != record_name
        or manifest.get("primary_record_identity_sha256") != record_identity
        or manifest.get("files") != artifact_records(root)
        or completion != expected_completion
    ):
        raise RuntimeError(f"{artifact_kind} lock validation failed: {root}")
    return record, manifest


def _validate_foreign_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = require_regular(root / "manifest.json", "foreign manifest")
    completion_path = require_regular(root / "completion.json", "foreign completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = require_hex64(manifest.get("identity_sha256"), "foreign manifest identity")
    if (
        canonical_sha256(without_identity(manifest)) != identity
        or manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or manifest.get("files") != artifact_records(root)
    ):
        raise RuntimeError(f"foreign lock manifest/completion failed: {root}")
    return manifest, completion


def validate_foundation_locks() -> dict[str, dict[str, Any]]:
    protocol_root = require_real_directory(PHASE1_PROTOCOL_LOCK, "phase-1 protocol lock")
    threshold_root = require_real_directory(
        PHASE1_THRESHOLD_LOCK, "phase-1 threshold lock"
    )
    sampling_root = require_real_directory(SAMPLING_SOURCE_LOCK, "sampling source lock")

    protocol = load_json(require_regular(protocol_root / "third_pool_protocol.json", "protocol"))
    p_manifest, _ = _validate_foreign_manifest(protocol_root)
    threshold = load_json(require_regular(threshold_root / "thresholds_locked.json", "thresholds"))
    t_manifest, _ = _validate_foreign_manifest(threshold_root)
    sampling = load_json(require_regular(sampling_root / "sampling_protocol.json", "sampling protocol"))
    s_manifest, _ = _validate_foreign_manifest(sampling_root)

    expected = (
        (
            protocol,
            p_manifest,
            EXPECTED_PHASE1_PROTOCOL_IDENTITY,
            EXPECTED_PHASE1_PROTOCOL_MANIFEST_IDENTITY,
            "phase-1 protocol",
        ),
        (
            threshold,
            t_manifest,
            EXPECTED_PHASE1_THRESHOLD_IDENTITY,
            EXPECTED_PHASE1_THRESHOLD_MANIFEST_IDENTITY,
            "phase-1 thresholds",
        ),
        (
            sampling,
            s_manifest,
            EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
            EXPECTED_SAMPLING_MANIFEST_IDENTITY,
            "sampling protocol",
        ),
    )
    for record, manifest, record_id, manifest_id, description in expected:
        if (
            canonical_sha256(without_identity(record)) != record.get("identity_sha256")
            or record.get("identity_sha256") != record_id
            or manifest.get("identity_sha256") != manifest_id
        ):
            raise RuntimeError(f"wrong pinned {description} identity")

    if (
        sampling.get("phase1_protocol", {}).get("identity_sha256")
        != EXPECTED_PHASE1_PROTOCOL_IDENTITY
        or sampling.get("phase1_protocol", {}).get("manifest_identity_sha256")
        != EXPECTED_PHASE1_PROTOCOL_MANIFEST_IDENTITY
        or sampling.get("phase1_thresholds", {}).get("identity_sha256")
        != EXPECTED_PHASE1_THRESHOLD_IDENTITY
        or sampling.get("phase1_thresholds", {}).get("manifest_identity_sha256")
        != EXPECTED_PHASE1_THRESHOLD_MANIFEST_IDENTITY
    ):
        raise RuntimeError("sampling lock is not bound to the exact phase-1 locks")
    frozen = sampling.get("frozen_statistics", {})
    if (
        frozen.get("candidate_combination_allowed") is not False
        or frozen.get("B_blur_mean", {}).get("primary_endpoint")
        != "blur_or_soft_fusion_clear_bad_vs_clean_good"
        or frozen.get("C_c3_low_jump", {}).get("primary_endpoint")
        != "all_clear_bad_vs_clean_good"
    ):
        raise RuntimeError("sampling lock frozen-statistics contract changed")
    return {"protocol": protocol, "thresholds": threshold, "sampling": sampling}


def scientific_contract() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_THIRD_POOL_CONSENSUS_OR_FEATURE_IDENTITIES",
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        "foundation_identity_pins": {
            "phase1_protocol_identity_sha256": EXPECTED_PHASE1_PROTOCOL_IDENTITY,
            "phase1_protocol_manifest_identity_sha256": EXPECTED_PHASE1_PROTOCOL_MANIFEST_IDENTITY,
            "phase1_threshold_identity_sha256": EXPECTED_PHASE1_THRESHOLD_IDENTITY,
            "phase1_threshold_manifest_identity_sha256": EXPECTED_PHASE1_THRESHOLD_MANIFEST_IDENTITY,
            "sampling_protocol_identity_sha256": EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
            "sampling_manifest_identity_sha256": EXPECTED_SAMPLING_MANIFEST_IDENTITY,
        },
        "cohort": {
            "classes_ordered": list(CLASSES),
            "global_seeds": list(SEEDS),
            "trajectory_count": TRAJECTORY_COUNT,
            "global_seed_block_size": len(CLASSES),
        },
        "two_stage_access_contract": {
            "stage_A": (
                "open only consensus manifest.json, completion.json, and aggregate_counts.json; "
                "always stop after an aggregate gate receipt"
            ),
            "stage_A_forbidden": [
                "consensus_rows.csv",
                "sampling-pool directory stat/hash/open",
                "feature-product directory stat/hash/open",
                "score CSV or NPZ",
                "image",
                "screen result",
            ],
            "stage_B_unlock": (
                "a separately completed stage-A receipt bound to the same input lock, "
                "with blur/fusion clear-bad >=15 and total clear-bad >=30"
            ),
        },
        "event_gate": {
            "minimum_blur_or_soft_fusion_clear_bad": EVENT_MIN_BLUR,
            "minimum_total_clear_bad": EVENT_MIN_TOTAL_BAD,
            "logical_rule": "both minima must hold",
            "failure_output": "aggregate label/phenotype counts only",
        },
        "candidates": {
            "B_blur_mean": {
                "feature": VISUAL_FEATURE,
                "orientation": "bad_high",
                "primary_endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                "guardrail_endpoint": "all_clear_bad_vs_clean_good",
                "gate": {
                    "pair_weighted_auc_at_least": 0.75,
                    "Holm_adjusted_p_strictly_below": 0.05,
                    "alpha_0p10_micro_TP_at_least": 3,
                    "alpha_0p10_micro_TPR_strictly_greater_than_micro_FPR": True,
                },
                "intervention_authorization": "blur-specific experiment only",
            },
            "C_c3_low_jump": {
                "feature": PRIMARY_FEATURE,
                "orientation": "bad_low",
                "primary_endpoint": "all_clear_bad_vs_clean_good",
                "guardrail_endpoint": "blur_or_soft_fusion_clear_bad_vs_clean_good",
                "gate": {
                    "pair_weighted_auc_at_least": 0.70,
                    "Holm_adjusted_p_strictly_below": 0.05,
                },
                "intervention_authorization": False,
            },
        },
        "statistics": {
            "primary_auc": (
                "sum over classes of oriented concordant bad-good pairs, ties=0.5, "
                "divided by sum over classes n_bad_class*n_clean_good_class; zero total pairs fail closed"
            ),
            "operating_points": (
                "class-specific locked raw thresholds with strict comparisons; TP, FP, TPR, "
                "and FPR are micro-pooled across all three classes; zero denominator fails closed"
            ),
            "threshold_levels": list(ALPHAS),
            "permutation": {
                "draws": PERMUTATION_DRAWS,
                "rng": f"numpy.default_rng(PCG64(seed={PERMUTATION_SEED}))",
                "unit": "one intact ordered three-class label/phenotype block per global seed",
                "same_seed_permutation_used_for_both_candidates": True,
                "p_value": "(1+count(permuted primary AUC >= observed))/(1+draws)",
            },
            "multiple_testing": {
                "method": "Holm step-down",
                "family": list(CANDIDATES),
                "family_size": 2,
                "strict_alpha": 0.05,
            },
            "small_guardrail_suppression": (
                "descriptive subtype/per-class AUC is numeric only when both positive and "
                "clean-good counts are at least 5"
            ),
        },
        "fixed_guardrails": [
            "B on all clear-bad versus clean-good",
            "C on blur/soft-fusion clear-bad versus clean-good",
            "per-class primary AUC and event counts",
            "B and C on mixed blur-plus-structure clear-bad versus clean-good",
            "B and C on structural-non-blur clear-bad versus clean-good",
        ],
        "consensus_schema": {
            "experiment": CONSENSUS_EXPERIMENT,
            "aggregate_member": CONSENSUS_AGGREGATE_NAME,
            "row_member": CONSENSUS_ROWS_NAME,
            "root_members_exact": [
                "manifest.json",
                "completion.json",
                CONSENSUS_AGGREGATE_NAME,
                CONSENSUS_ROWS_NAME,
            ],
            "row_columns_exact": list(CONSENSUS_COLUMNS),
            "missing_or_extra_row_cells": "fail_closed",
            "three_independent_reviewers": True,
            "reviewers_score_threshold_alert_trajectory_blind": True,
        },
        "future_input_binding": {
            "mutable_fields_only": [
                "completed sampling-pool absolute path and pool-manifest identity",
                "consensus absolute path and manifest identity",
                "primary label-free product absolute path and manifest identity",
                "visual label-free product absolute path and manifest identity",
            ],
            "scientific_overrides_allowed": False,
        },
        "output_contract": {
            "aggregate_only": True,
            "forbidden": [
                "individual rows",
                "individual scores",
                "ranks",
                "permutation draws",
                "images",
                "trajectory arrays",
            ],
        },
    }


def freeze_source_lock(output: Path) -> Path:
    validate_foundation_locks()
    contract = scientific_contract()
    contract["identity_sha256"] = canonical_sha256(contract)
    return publish_record_lock(
        output,
        artifact_kind="dit_bad_good_third_pool_evaluation_source_lock_v1",
        record_name="scientific_contract.json",
        record=contract,
        source_copies={"evaluator_source.py": Path(__file__).resolve()},
    )


def validate_source_lock(root: Path, *, require_live_source: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_foundation_locks()
    contract, manifest = validate_record_lock(
        root,
        artifact_kind="dit_bad_good_third_pool_evaluation_source_lock_v1",
        record_name="scientific_contract.json",
    )
    expected = scientific_contract()
    expected["identity_sha256"] = canonical_sha256(expected)
    if contract != expected:
        raise RuntimeError("evaluation scientific contract differs from frozen source")
    by_name = _manifest_map(manifest, "evaluation source lock")
    if set(by_name) != {"evaluator_source.py", "scientific_contract.json"}:
        raise RuntimeError("evaluation source lock member set changed")
    source = require_regular(Path(root) / "evaluator_source.py", "frozen evaluator source")
    if sha256_file(source) != contract["implementation_source_sha256"]:
        raise RuntimeError("frozen evaluator source hash differs from scientific contract")
    if require_live_source and sha256_file(Path(__file__).resolve()) != sha256_file(source):
        raise RuntimeError("live evaluator source differs from frozen evaluator source")
    return contract, manifest


def _absolute_uninspected(path: Path) -> str:
    """Normalize text without exists/stat/resolve: bind mode must not inspect inputs."""

    return str(path.expanduser().absolute())


def bind_inputs(
    *,
    source_lock: Path,
    sampling_pool_path: Path,
    sampling_pool_manifest_identity: str,
    consensus_path: Path,
    consensus_manifest_identity: str,
    primary_path: Path,
    primary_manifest_identity: str,
    visual_path: Path,
    visual_manifest_identity: str,
    output: Path,
) -> Path:
    contract, source_manifest = validate_source_lock(source_lock)
    refs = {
        "sampling_pool": {
            "path": _absolute_uninspected(sampling_pool_path),
            "manifest_identity_sha256": require_hex64(
                sampling_pool_manifest_identity, "sampling-pool manifest identity"
            ),
        },
        "consensus": {
            "path": _absolute_uninspected(consensus_path),
            "manifest_identity_sha256": require_hex64(
                consensus_manifest_identity, "consensus manifest identity"
            ),
        },
        "primary_label_free_product": {
            "path": _absolute_uninspected(primary_path),
            "manifest_identity_sha256": require_hex64(
                primary_manifest_identity, "primary product manifest identity"
            ),
        },
        "visual_label_free_product": {
            "path": _absolute_uninspected(visual_path),
            "manifest_identity_sha256": require_hex64(
                visual_manifest_identity, "visual product manifest identity"
            ),
        },
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "BOUND_FUTURE_IDENTITIES_WITHOUT_OPENING_INPUT_PATHS",
        "scientific_contract_identity_sha256": contract["identity_sha256"],
        "evaluation_source_lock": {
            "path": str(Path(source_lock).expanduser().absolute()),
            "manifest_identity_sha256": source_manifest["identity_sha256"],
            "evaluator_source_sha256": contract["implementation_source_sha256"],
        },
        "foundation_identity_pins": contract["foundation_identity_pins"],
        "inputs": refs,
        "access_audit": {
            "sampling_pool_path_opened_or_statted": False,
            "consensus_path_opened_or_statted": False,
            "primary_product_path_opened_or_statted": False,
            "visual_product_path_opened_or_statted": False,
            "only_paths_and_caller_supplied_manifest_identities_bound": True,
            "scientific_field_override_interface_exists": False,
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    record["identity_sha256"] = canonical_sha256(record)
    return publish_record_lock(
        output,
        artifact_kind="dit_bad_good_third_pool_evaluation_input_binding_v1",
        record_name="input_binding.json",
        record=record,
        source_copies={"evaluator_source.py": Path(__file__).resolve()},
    )


def validate_input_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record, manifest = validate_record_lock(
        root,
        artifact_kind="dit_bad_good_third_pool_evaluation_input_binding_v1",
        record_name="input_binding.json",
    )
    if record.get("status") != "BOUND_FUTURE_IDENTITIES_WITHOUT_OPENING_INPUT_PATHS":
        raise RuntimeError("input-binding status changed")
    source_ref = record.get("evaluation_source_lock", {})
    source_lock = Path(str(source_ref.get("path", "")))
    contract, source_manifest = validate_source_lock(source_lock)
    if (
        record.get("scientific_contract_identity_sha256") != contract["identity_sha256"]
        or record.get("foundation_identity_pins") != contract["foundation_identity_pins"]
        or source_ref.get("manifest_identity_sha256") != source_manifest["identity_sha256"]
        or source_ref.get("evaluator_source_sha256")
        != contract["implementation_source_sha256"]
        or record.get("implementation_source_sha256")
        != contract["implementation_source_sha256"]
    ):
        raise RuntimeError("input binding differs from evaluation source lock")
    inputs = record.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "sampling_pool",
        "consensus",
        "primary_label_free_product",
        "visual_label_free_product",
    }:
        raise RuntimeError("input-binding member set changed")
    for name, item in inputs.items():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise RuntimeError(f"malformed bound input: {name}")
        require_hex64(item.get("manifest_identity_sha256"), f"{name} manifest identity")
        if not Path(item["path"]).is_absolute():
            raise RuntimeError(f"bound path is not absolute: {name}")
    audit = record.get("access_audit", {})
    if audit != {
        "sampling_pool_path_opened_or_statted": False,
        "consensus_path_opened_or_statted": False,
        "primary_product_path_opened_or_statted": False,
        "visual_product_path_opened_or_statted": False,
        "only_paths_and_caller_supplied_manifest_identities_bound": True,
        "scientific_field_override_interface_exists": False,
    }:
        raise RuntimeError("input-binding access audit changed")
    return record, manifest


def _validate_count_row(row: Mapping[str, Any], expected_total: int, description: str) -> dict[str, int]:
    if set(row) != set(COUNT_KEYS):
        raise RuntimeError(f"{description} count keys changed")
    result: dict[str, int] = {}
    for key in COUNT_KEYS:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"invalid {description} count: {key}")
        result[key] = value
    if result["trajectory_count"] != expected_total:
        raise RuntimeError(f"wrong {description} trajectory count")
    if (
        result["clean_good"]
        + result["clear_bad"]
        + result["mild_or_disputed"]
        != expected_total
    ):
        raise RuntimeError(f"{description} severity counts do not partition the cohort")
    if (
        result["blur_or_soft_fusion_clear_bad"]
        + result["structural_non_blur_clear_bad"]
        + result["phenotype_disputed_clear_bad"]
        != result["clear_bad"]
        or result["mixed_blur_and_structure_clear_bad"]
        > result["blur_or_soft_fusion_clear_bad"]
    ):
        raise RuntimeError(f"{description} phenotype counts do not partition clear-bad")
    return result


def validate_aggregate_counts(counts: Any) -> dict[str, Any]:
    if not isinstance(counts, dict) or set(counts) != {"overall", "per_class"}:
        raise RuntimeError("aggregate counts schema changed")
    overall = _validate_count_row(counts["overall"], TRAJECTORY_COUNT, "overall")
    per_class = counts["per_class"]
    if not isinstance(per_class, list) or len(per_class) != len(CLASSES):
        raise RuntimeError("per-class count rows changed")
    observed: dict[int, dict[str, int]] = {}
    for item in per_class:
        if not isinstance(item, dict) or set(item) != {"class_id", *COUNT_KEYS}:
            raise RuntimeError("malformed per-class count row")
        class_id = item["class_id"]
        if class_id in observed or class_id not in CLASSES:
            raise RuntimeError("invalid/duplicate class in aggregate counts")
        observed[class_id] = _validate_count_row(
            {key: item[key] for key in COUNT_KEYS}, len(SEEDS), f"class {class_id}"
        )
    if tuple(observed) != CLASSES:
        raise RuntimeError("per-class count order changed")
    for key in COUNT_KEYS:
        if key == "trajectory_count":
            continue
        if sum(observed[class_id][key] for class_id in CLASSES) != overall[key]:
            raise RuntimeError(f"per-class {key} does not add to overall")
    return {"overall": overall, "per_class": per_class}


def _consensus_member_names_safe(by_name: Mapping[str, Any]) -> None:
    required = {CONSENSUS_AGGREGATE_NAME, CONSENSUS_ROWS_NAME}
    if not required.issubset(by_name):
        raise RuntimeError("consensus manifest lacks aggregate or row member")
    for name in by_name:
        lower = name.lower()
        if any(token in lower for token in FORBIDDEN_CONSENSUS_MEMBER_TOKENS):
            raise RuntimeError(f"consensus lock contains forbidden evidence member: {name}")


def _validate_consensus_exact_tree(root: Path) -> None:
    """Reject extra consensus entries before opening any non-metadata payload."""

    expected = {
        "manifest.json",
        "completion.json",
        CONSENSUS_AGGREGATE_NAME,
        CONSENSUS_ROWS_NAME,
    }
    observed: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"consensus lock contains a non-regular entry: {path.name}")
        observed.add(path.name)
    if observed != expected:
        raise RuntimeError("consensus lock must contain exactly the frozen four files")


def load_consensus_aggregate_only(
    ref: Mapping[str, Any], sampling_pool_ref: Mapping[str, Any]
) -> dict[str, Any]:
    """Stage-A-only loader. Do not add any access to the row member or products."""

    root = require_real_directory(Path(ref["path"]), "bound consensus lock")
    _validate_consensus_exact_tree(root)
    manifest_path = require_regular(root / "manifest.json", "consensus manifest")
    completion_path = require_regular(root / "completion.json", "consensus completion")
    aggregate_path = require_regular(
        root / CONSENSUS_AGGREGATE_NAME, "consensus aggregate counts"
    )
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    aggregate = load_json(aggregate_path)
    manifest_identity = require_hex64(
        manifest.get("identity_sha256"), "consensus manifest identity"
    )
    aggregate_identity = require_hex64(
        aggregate.get("identity_sha256"), "aggregate-count identity"
    )
    if (
        canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest_identity != ref.get("manifest_identity_sha256")
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != CONSENSUS_EXPERIMENT
        or canonical_sha256(without_identity(aggregate)) != aggregate_identity
        or manifest.get("aggregate_counts_identity_sha256") != aggregate_identity
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("aggregate_counts_file_sha256") != sha256_file(aggregate_path)
        or completion.get("aggregate_counts_identity_sha256") != aggregate_identity
    ):
        raise RuntimeError("consensus aggregate-only lock validation failed")
    by_name = _manifest_map(manifest, "consensus")
    _consensus_member_names_safe(by_name)
    aggregate_member = by_name[CONSENSUS_AGGREGATE_NAME]
    if (
        aggregate_member.get("bytes") != aggregate_path.stat().st_size
        or aggregate_member.get("sha256") != sha256_file(aggregate_path)
    ):
        raise RuntimeError("consensus aggregate member changed")
    if (
        aggregate.get("status") != "FROZEN_THIRD_POOL_BLIND_CONSENSUS_COUNTS"
        or aggregate.get("experiment") != CONSENSUS_EXPERIMENT
        or aggregate.get("phase1_protocol_identity_sha256")
        != EXPECTED_PHASE1_PROTOCOL_IDENTITY
        or aggregate.get("sampling_protocol_identity_sha256")
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or aggregate.get("sampling_pool_identity_sha256")
        != sampling_pool_ref.get("manifest_identity_sha256")
        or tuple(aggregate.get("classes_ordered", ())) != CLASSES
        or tuple(aggregate.get("global_seeds", ())) != SEEDS
        or aggregate.get("trajectory_count") != TRAJECTORY_COUNT
        or aggregate.get("labels_and_phenotypes_immutable") is not True
        or aggregate.get("three_independent_endpoint_only_reviewers") is not True
        or aggregate.get("reviewers_score_threshold_alert_trajectory_blind") is not True
        or aggregate.get("consensus_rows_member") != CONSENSUS_ROWS_NAME
        or aggregate.get("consensus_rows_file_sha256")
        != by_name[CONSENSUS_ROWS_NAME].get("sha256")
    ):
        raise RuntimeError("consensus aggregate scientific contract changed")
    counts = validate_aggregate_counts(aggregate.get("counts"))
    return {
        "root": str(root),
        "manifest_identity_sha256": manifest_identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "aggregate_identity_sha256": aggregate_identity,
        "aggregate_file_sha256": sha256_file(aggregate_path),
        "row_member_declared_sha256": by_name[CONSENSUS_ROWS_NAME]["sha256"],
        "counts": counts,
    }


def run_stage_a(*, input_lock: Path, output: Path) -> Path:
    binding, binding_manifest = validate_input_binding(input_lock)
    consensus = load_consensus_aggregate_only(
        binding["inputs"]["consensus"], binding["inputs"]["sampling_pool"]
    )
    overall = consensus["counts"]["overall"]
    blur_ok = overall["blur_or_soft_fusion_clear_bad"] >= EVENT_MIN_BLUR
    total_ok = overall["clear_bad"] >= EVENT_MIN_TOTAL_BAD
    passed = bool(blur_ok and total_ok)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "EVENT_GATE_PASSED_SCORES_STILL_UNOPENED"
            if passed
            else "EVENT_GATE_FAILED_NO_SCORE_ACCESS"
        ),
        "input_binding_identity_sha256": binding["identity_sha256"],
        "input_binding_manifest_identity_sha256": binding_manifest["identity_sha256"],
        "scientific_contract_identity_sha256": binding[
            "scientific_contract_identity_sha256"
        ],
        "consensus_receipt": {
            key: consensus[key]
            for key in (
                "manifest_identity_sha256",
                "manifest_file_sha256",
                "aggregate_identity_sha256",
                "aggregate_file_sha256",
                "row_member_declared_sha256",
            )
        },
        "aggregate_counts": consensus["counts"],
        "event_gate": {
            "minimum_blur_or_soft_fusion_clear_bad": EVENT_MIN_BLUR,
            "observed_blur_or_soft_fusion_clear_bad": overall[
                "blur_or_soft_fusion_clear_bad"
            ],
            "blur_minimum_met": blur_ok,
            "minimum_total_clear_bad": EVENT_MIN_TOTAL_BAD,
            "observed_total_clear_bad": overall["clear_bad"],
            "total_bad_minimum_met": total_ok,
            "both_minima_met": passed,
            "stage_B_authorized": passed,
        },
        "access_audit": {
            "consensus_manifest_opened": True,
            "consensus_completion_opened": True,
            "consensus_aggregate_counts_opened": True,
            "consensus_rows_opened_or_hashed": False,
            "sampling_pool_path_opened_statted_or_hashed": False,
            "primary_feature_product_path_opened_statted_or_hashed": False,
            "visual_feature_product_path_opened_statted_or_hashed": False,
            "score_csv_or_npz_opened": False,
            "image_opened": False,
            "screen_result_opened": False,
            "stage_B_invoked_in_same_process": False,
        },
        "output_scope": "aggregate label/phenotype counts and gate decision only",
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    receipt["identity_sha256"] = canonical_sha256(receipt)
    return publish_record_lock(
        output,
        artifact_kind="dit_bad_good_third_pool_stage_a_event_gate_v1",
        record_name="stage_a_gate_receipt.json",
        record=receipt,
        source_copies={"evaluator_source.py": Path(__file__).resolve()},
    )


def validate_stage_a_receipt(
    root: Path, binding: Mapping[str, Any], binding_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, manifest = validate_record_lock(
        root,
        artifact_kind="dit_bad_good_third_pool_stage_a_event_gate_v1",
        record_name="stage_a_gate_receipt.json",
    )
    if (
        receipt.get("input_binding_identity_sha256") != binding["identity_sha256"]
        or receipt.get("input_binding_manifest_identity_sha256")
        != binding_manifest["identity_sha256"]
        or receipt.get("scientific_contract_identity_sha256")
        != binding["scientific_contract_identity_sha256"]
        or receipt.get("implementation_source_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("stage-A receipt lineage differs from bound evaluator")
    counts = validate_aggregate_counts(receipt.get("aggregate_counts"))
    overall = counts["overall"]
    blur_ok = overall["blur_or_soft_fusion_clear_bad"] >= EVENT_MIN_BLUR
    total_ok = overall["clear_bad"] >= EVENT_MIN_TOTAL_BAD
    expected_gate = {
        "minimum_blur_or_soft_fusion_clear_bad": EVENT_MIN_BLUR,
        "observed_blur_or_soft_fusion_clear_bad": overall[
            "blur_or_soft_fusion_clear_bad"
        ],
        "blur_minimum_met": blur_ok,
        "minimum_total_clear_bad": EVENT_MIN_TOTAL_BAD,
        "observed_total_clear_bad": overall["clear_bad"],
        "total_bad_minimum_met": total_ok,
        "both_minima_met": bool(blur_ok and total_ok),
        "stage_B_authorized": bool(blur_ok and total_ok),
    }
    if receipt.get("event_gate") != expected_gate:
        raise RuntimeError("stage-A event gate does not replay aggregate counts")
    if (
        receipt.get("status") != "EVENT_GATE_PASSED_SCORES_STILL_UNOPENED"
        or not expected_gate["stage_B_authorized"]
    ):
        raise RuntimeError("stage B is not authorized by the event gate")
    expected_audit = {
        "consensus_manifest_opened": True,
        "consensus_completion_opened": True,
        "consensus_aggregate_counts_opened": True,
        "consensus_rows_opened_or_hashed": False,
        "sampling_pool_path_opened_statted_or_hashed": False,
        "primary_feature_product_path_opened_statted_or_hashed": False,
        "visual_feature_product_path_opened_statted_or_hashed": False,
        "score_csv_or_npz_opened": False,
        "image_opened": False,
        "screen_result_opened": False,
        "stage_B_invoked_in_same_process": False,
    }
    if receipt.get("access_audit") != expected_audit:
        raise RuntimeError("stage-A no-score access audit changed")
    return receipt, manifest


def _parse_bool(value: str, description: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise RuntimeError(f"{description} must be literal true or false")


def _expected_keys() -> set[tuple[int, int, int]]:
    return {
        (seed, slot, class_id)
        for seed in SEEDS
        for slot, class_id in enumerate(CLASSES)
    }


def _count_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def one(subset: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        clear = [row for row in subset if row["final_severity"] == "clear_bad"]
        blur = [row for row in clear if row["blur_component_consensus"]]
        mixed = [
            row
            for row in clear
            if row["blur_component_consensus"]
            and row["discrete_structure_component_consensus"]
        ]
        structural = [
            row
            for row in clear
            if not row["blur_component_consensus"]
            and row["discrete_structure_component_consensus"]
        ]
        disputed = [
            row
            for row in clear
            if not row["blur_component_consensus"]
            and not row["discrete_structure_component_consensus"]
        ]
        return {
            "trajectory_count": len(subset),
            "clean_good": sum(row["final_severity"] == "clean_good" for row in subset),
            "clear_bad": len(clear),
            "mild_or_disputed": sum(
                row["final_severity"] == "mild_or_disputed" for row in subset
            ),
            "blur_or_soft_fusion_clear_bad": len(blur),
            "mixed_blur_and_structure_clear_bad": len(mixed),
            "structural_non_blur_clear_bad": len(structural),
            "phenotype_disputed_clear_bad": len(disputed),
        }

    return {
        "overall": one(rows),
        "per_class": [
            {"class_id": class_id, **one([r for r in rows if r["class_id"] == class_id])}
            for class_id in CLASSES
        ],
    }


def load_full_consensus(
    ref: Mapping[str, Any], aggregate_receipt: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = require_real_directory(Path(ref["path"]), "bound consensus lock")
    _validate_consensus_exact_tree(root)
    manifest_path = require_regular(root / "manifest.json", "consensus manifest")
    completion_path = require_regular(root / "completion.json", "consensus completion")
    aggregate_path = require_regular(root / CONSENSUS_AGGREGATE_NAME, "aggregate counts")
    rows_path = require_regular(root / CONSENSUS_ROWS_NAME, "consensus rows")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    aggregate = load_json(aggregate_path)
    by_name = _manifest_map(manifest, "consensus")
    _consensus_member_names_safe(by_name)
    if (
        manifest.get("identity_sha256") != ref.get("manifest_identity_sha256")
        or manifest.get("identity_sha256")
        != aggregate_receipt.get("manifest_identity_sha256")
        or canonical_sha256(without_identity(manifest)) != manifest.get("identity_sha256")
        or canonical_sha256(without_identity(aggregate)) != aggregate.get("identity_sha256")
        or aggregate.get("identity_sha256")
        != aggregate_receipt.get("aggregate_identity_sha256")
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or completion.get("aggregate_counts_file_sha256") != sha256_file(aggregate_path)
        or completion.get("aggregate_counts_identity_sha256")
        != aggregate.get("identity_sha256")
        or manifest.get("files") != artifact_records(root)
        or sha256_file(rows_path) != aggregate_receipt.get("row_member_declared_sha256")
    ):
        raise RuntimeError("full consensus lock differs from stage-A receipt")

    rows: list[dict[str, Any]] = []
    observed: set[tuple[int, int, int]] = set()
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CONSENSUS_COLUMNS:
            raise RuntimeError("consensus row columns/order changed")
        for raw in reader:
            if None in raw or set(raw) != set(CONSENSUS_COLUMNS):
                raise RuntimeError("consensus row has missing or extra cells")
            try:
                sample_index = int(raw["sample_index"])
                seed = int(raw["global_seed"])
                slot = int(raw["class_slot"])
                class_id = int(raw["class_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("invalid consensus identifier") from exc
            key = (seed, slot, class_id)
            if key in observed:
                raise RuntimeError(f"duplicate consensus key: {key}")
            observed.add(key)
            if (
                seed not in SEEDS
                or slot < 0
                or slot >= len(CLASSES)
                or class_id != CLASSES[slot]
                or sample_index != (seed - SEEDS[0]) * len(CLASSES) + slot
                or raw["final_severity"] not in SEVERITIES
            ):
                raise RuntimeError(f"consensus row contract failed: {key}")
            rows.append(
                {
                    "sample_index": sample_index,
                    "global_seed": seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "final_severity": raw["final_severity"],
                    "blur_component_consensus": _parse_bool(
                        raw["blur_component_consensus"], "blur component"
                    ),
                    "discrete_structure_component_consensus": _parse_bool(
                        raw["discrete_structure_component_consensus"],
                        "structure component",
                    ),
                }
            )
    if observed != _expected_keys() or len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError("consensus rows are not exact third-pool Cartesian cohort")
    recomputed = validate_aggregate_counts(_count_from_rows(rows))
    if recomputed != aggregate.get("counts") or recomputed != aggregate_receipt.get(
        "counts"
    ):
        raise RuntimeError("consensus row counts do not replay stage-A aggregate counts")
    return rows, {
        "manifest_identity_sha256": manifest["identity_sha256"],
        "aggregate_identity_sha256": aggregate["identity_sha256"],
        "rows_file_sha256": sha256_file(rows_path),
    }


def validate_sampling_pool(
    ref: Mapping[str, Any]
) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """Validate only the aggregate pool receipts; endpoint payloads stay unopened."""

    root = require_real_directory(Path(ref["path"]), "completed third-pool root")
    manifest_path = require_regular(root / "pool_manifest.json", "pool manifest")
    completion_path = require_regular(root / "pool_completion.json", "pool completion")
    plan_path = require_regular(root / "execution_plan.json", "pool execution plan")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    identity = require_hex64(manifest.get("identity_sha256"), "sampling-pool identity")
    if (
        canonical_sha256(without_identity(manifest)) != identity
        or identity != ref.get("manifest_identity_sha256")
        or manifest.get("status") != "complete"
        or manifest.get("sampling_protocol_identity_sha256")
        != EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or manifest.get("phase1_protocol_identity_sha256")
        != EXPECTED_PHASE1_PROTOCOL_IDENTITY
        or manifest.get("phase1_threshold_identity_sha256")
        != EXPECTED_PHASE1_THRESHOLD_IDENTITY
        or manifest.get("seed_count") != len(SEEDS)
        or manifest.get("trajectory_count") != TRAJECTORY_COUNT
        or manifest.get("observation_only") is not True
        or manifest.get("labels_reviews_screen_results_or_sample_scores_read") is not False
        or manifest.get("score_label_join_performed") is not False
        or completion.get("complete") is not True
        or completion.get("pool_identity_sha256") != identity
        or completion.get("pool_manifest_sha256") != sha256_file(manifest_path)
        or completion.get("execution_plan_sha256") != sha256_file(plan_path)
        or completion.get("seed_count") != len(SEEDS)
        or completion.get("trajectory_count") != TRAJECTORY_COUNT
    ):
        raise RuntimeError("completed sampling-pool receipt failed")
    outputs = manifest.get("seed_outputs")
    if not isinstance(outputs, list) or len(outputs) != len(SEEDS):
        raise RuntimeError("sampling-pool seed receipt count changed")
    by_seed: dict[int, dict[str, str]] = {}
    for expected_seed, item in zip(SEEDS, outputs, strict=True):
        if not isinstance(item, dict) or item.get("seed") != expected_seed:
            raise RuntimeError("sampling-pool seed receipt order changed")
        expected_relative = f"third_pool_v1_seed{expected_seed:03d}"
        if item.get("relative_output") != expected_relative:
            raise RuntimeError("sampling-pool relative seed path changed")
        fields = {
            "identity_sha256": item.get("identity_sha256"),
            "manifest_sha256": item.get("manifest_sha256"),
            "completion_sha256": item.get("completion_sha256"),
            "trace_sha256": item.get("trace_npz_sha256"),
        }
        for name, digest in fields.items():
            require_hex64(digest, f"sampling-pool seed {expected_seed} {name}")
        by_seed[expected_seed] = fields
    return by_seed, {
        "path": str(root),
        "pool_identity_sha256": identity,
        "pool_manifest_file_sha256": sha256_file(manifest_path),
        "pool_completion_file_sha256": sha256_file(completion_path),
        "execution_plan_file_sha256": sha256_file(plan_path),
        "seed_receipt_lineage_sha256": canonical_sha256(
            [{"seed": seed, **by_seed[seed]} for seed in SEEDS]
        ),
        "seed_count": len(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
    }


def _catalog_feature_row(path: Path, feature: str) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        for row in reader:
            if None in row or set(row) != fields:
                raise RuntimeError("feature-catalog row has missing or extra cells")
            if row.get("feature") == feature:
                matches.append(dict(row))
    if len(matches) != 1:
        raise RuntimeError(f"expected one feature-catalog row for {feature}")
    return matches[0]


def load_feature_product(
    ref: Mapping[str, Any], *, candidate: str
) -> tuple[
    dict[tuple[int, int, int], float],
    dict[str, Any],
    dict[int, dict[str, str]],
]:
    if candidate == "B_blur_mean":
        feature = VISUAL_FEATURE
        source_sha = VISUAL_EXTRACTOR_SHA256
        experiment = "dit_predxstart_preterminal_visual_tracks_label_free"
        status = "COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION"
        supervision_field = "labels_read_or_emitted"
    elif candidate == "C_c3_low_jump":
        feature = PRIMARY_FEATURE
        source_sha = PRIMARY_EXTRACTOR_SHA256
        experiment = "dit_bad_good_custom_trace_metric_discovery"
        status = "DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER"
        supervision_field = "labels_joined"
    else:
        raise RuntimeError(f"unknown candidate product: {candidate}")
    root = require_real_directory(Path(ref["path"]), f"{candidate} label-free product")
    manifest_path = require_regular(root / "manifest.json", "product manifest")
    completion_path = require_regular(root / "completion.json", "product completion")
    summary_path = require_regular(root / "summary.json", "product summary")
    catalog_path = require_regular(root / "feature_catalog.csv", "feature catalog")
    score_path = require_regular(root / "sample_features.csv", "sample feature CSV")
    inventory_path = require_regular(root / "source_inventory.json", "source inventory")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    summary = load_json(summary_path)
    inventory = load_json(inventory_path)
    identity = require_hex64(manifest.get("identity_sha256"), "product manifest identity")
    by_name = _manifest_map(manifest, "label-free product")
    required = {
        "analysis_source.py",
        "feature_catalog.csv",
        "sample_features.csv",
        "source_inventory.json",
        "summary.json",
    }
    if (
        not required.issubset(by_name)
        or identity != ref.get("manifest_identity_sha256")
        or canonical_sha256(without_identity(manifest)) != identity
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != experiment
        or manifest.get("analysis_source_sha256") != source_sha
        or by_name["analysis_source.py"].get("sha256") != source_sha
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != identity
        or manifest.get("files") != artifact_records(root)
        or summary.get("status") != status
        or summary.get(supervision_field) is not False
        or tuple(summary.get("ordered_classes", ())) != CLASSES
        or tuple(summary.get("ordered_seeds", ())) != SEEDS
        or summary.get("sample_count") != TRAJECTORY_COUNT
        or tuple(inventory.get("ordered_classes", ())) != CLASSES
        or tuple(inventory.get("ordered_seeds", ())) != SEEDS
        or inventory.get("analysis_source", {}).get("sha256") != source_sha
    ):
        raise RuntimeError(f"{candidate} label-free product contract failed")
    catalog = _catalog_feature_row(catalog_path, feature)
    if (
        catalog.get("latest_required_sampling_step") != "149"
        or catalog.get("latest_required_internal_timestep") != "100"
        or catalog.get("preterminal_actionable") != "True"
        or catalog.get("uses_realized_innovation") != "False"
    ):
        raise RuntimeError(f"{candidate} preterminal timing contract changed")

    values: dict[tuple[int, int, int], float] = {}
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        required_fields = {
            "sample_index",
            "run_index",
            "global_seed",
            "class_slot",
            "class_id",
            feature,
        }
        if not required_fields.issubset(fields):
            raise RuntimeError(f"{candidate} score CSV lacks required columns")
        for field in fields:
            if field == feature:
                # Candidate B's frozen feature name contains the diagnostic word
                # ``severity``; it is a label-free image statistic, not a label.
                continue
            lower = field.lower()
            if any(token in lower for token in FORBIDDEN_FEATURE_HEADER_TOKENS):
                raise RuntimeError(f"label-like column in label-free product: {field}")
        for raw in reader:
            if None in raw or set(raw) != set(fields):
                raise RuntimeError(f"{candidate} score row has missing or extra cells")
            try:
                sample_index = int(raw["sample_index"])
                run_index = int(raw["run_index"])
                seed = int(raw["global_seed"])
                slot = int(raw["class_slot"])
                class_id = int(raw["class_id"])
                value = float(raw[feature])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"invalid {candidate} score row") from exc
            key = (seed, slot, class_id)
            if (
                key in values
                or seed not in SEEDS
                or slot < 0
                or slot >= len(CLASSES)
                or class_id != CLASSES[slot]
                or run_index != seed - SEEDS[0]
                or sample_index != (seed - SEEDS[0]) * len(CLASSES) + slot
                or not math.isfinite(value)
            ):
                raise RuntimeError(f"{candidate} score row contract failed: {key}")
            values[key] = value
    if set(values) != _expected_keys() or len(values) != TRAJECTORY_COUNT:
        raise RuntimeError(f"{candidate} score cohort is incomplete")
    trace_runs = inventory.get("trace_runs")
    if not isinstance(trace_runs, list) or len(trace_runs) != len(SEEDS):
        raise RuntimeError(f"{candidate} trace lineage count changed")
    trace_lineage: dict[int, dict[str, str]] = {}
    for expected_seed, item in zip(SEEDS, trace_runs, strict=True):
        if (
            not isinstance(item, dict)
            or item.get("global_seed") != expected_seed
            or tuple(item.get("classes", ())) != CLASSES
        ):
            raise RuntimeError(f"{candidate} trace lineage order changed")
        fields = {
            "identity_sha256": item.get("identity_sha256"),
            "manifest_sha256": item.get("manifest_sha256"),
            "completion_sha256": item.get("completion_sha256"),
            "trace_sha256": item.get("trace_sha256"),
        }
        for name, digest in fields.items():
            require_hex64(digest, f"{candidate} seed {expected_seed} {name}")
        trace_lineage[expected_seed] = fields
    lineage_hash = canonical_sha256(
        [{"seed": seed, **trace_lineage[seed]} for seed in SEEDS]
    )
    return values, {
        "path": str(root),
        "manifest_identity_sha256": identity,
        "manifest_file_sha256": sha256_file(manifest_path),
        "sample_features_file_sha256": sha256_file(score_path),
        "feature_catalog_file_sha256": sha256_file(catalog_path),
        "analysis_source_sha256": source_sha,
        "feature": feature,
        "trace_run_lineage_sha256": lineage_hash,
    }, trace_lineage


def _positive(row: Mapping[str, Any], endpoint: str) -> bool:
    if row["final_severity"] != "clear_bad":
        return False
    if endpoint == "all_clear_bad":
        return True
    if endpoint == "blur_or_soft_fusion_clear_bad":
        return bool(row["blur_component_consensus"])
    if endpoint == "mixed_blur_and_structure_clear_bad":
        return bool(
            row["blur_component_consensus"]
            and row["discrete_structure_component_consensus"]
        )
    if endpoint == "structural_non_blur_clear_bad":
        return bool(
            not row["blur_component_consensus"]
            and row["discrete_structure_component_consensus"]
        )
    if endpoint == "phenotype_disputed_clear_bad":
        return bool(
            not row["blur_component_consensus"]
            and not row["discrete_structure_component_consensus"]
        )
    raise RuntimeError(f"unknown endpoint: {endpoint}")


def _oriented_score(row: Mapping[str, Any], candidate: str) -> float:
    raw = float(row["B_score"] if candidate == "B_blur_mean" else row["C_score"])
    return raw if candidate == "B_blur_mean" else -raw


def auc_summary(
    rows: Sequence[Mapping[str, Any]], *, candidate: str, endpoint: str
) -> dict[str, Any]:
    numerator = 0.0
    denominator = 0
    per_class: list[dict[str, Any]] = []
    for class_id in CLASSES:
        positives = np.asarray(
            [
                _oriented_score(row, candidate)
                for row in rows
                if row["class_id"] == class_id and _positive(row, endpoint)
            ],
            dtype=np.float64,
        )
        negatives = np.asarray(
            [
                _oriented_score(row, candidate)
                for row in rows
                if row["class_id"] == class_id
                and row["final_severity"] == "clean_good"
            ],
            dtype=np.float64,
        )
        pairs = int(len(positives) * len(negatives))
        concordant = 0.0
        if pairs:
            delta = positives[:, None] - negatives[None, :]
            concordant = float(np.sum(delta > 0.0) + 0.5 * np.sum(delta == 0.0))
        numerator += concordant
        denominator += pairs
        reportable = len(positives) >= 5 and len(negatives) >= 5
        per_class.append(
            {
                "class_id": class_id,
                "positive_count": int(len(positives)),
                "clean_good_count": int(len(negatives)),
                "pair_count": pairs,
                "auc": float(concordant / pairs) if pairs and reportable else None,
                "numeric_auc_suppressed": bool(pairs and not reportable),
            }
        )
    return {
        "candidate": candidate,
        "endpoint": f"{endpoint}_vs_clean_good",
        "positive_count": sum(row["positive_count"] for row in per_class),
        "clean_good_count": sum(row["clean_good_count"] for row in per_class),
        "pair_count": denominator,
        "concordant_pair_credit": numerator,
        "auc": float(numerator / denominator) if denominator else None,
        "zero_total_pair_denominator": denominator == 0,
        "per_class": per_class,
    }


def _tie_group_starts(oriented_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(oriented_scores, kind="stable")
    sorted_scores = oriented_scores[order]
    starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1)
    )
    return order, starts


def _batch_concordance(
    permuted_codes: np.ndarray, order: np.ndarray, starts: np.ndarray
) -> np.ndarray:
    ordered = permuted_codes[:, order]
    positives = ordered == 1
    negatives = ordered == 0
    pos_group = np.add.reduceat(positives, starts, axis=1).astype(np.float64)
    neg_group = np.add.reduceat(negatives, starts, axis=1).astype(np.float64)
    neg_before = np.cumsum(neg_group, axis=1) - neg_group
    return np.sum(pos_group * (neg_before + 0.5 * neg_group), axis=1)


def permutation_p_values(
    rows: Sequence[Mapping[str, Any]], *, draws: int = PERMUTATION_DRAWS
) -> dict[str, dict[str, Any]]:
    if draws <= 0:
        raise ValueError("permutation draws must be positive")
    ordered = sorted(rows, key=lambda row: (row["global_seed"], row["class_slot"]))
    if len(ordered) != TRAJECTORY_COUNT:
        raise RuntimeError("permutation cohort size changed")
    matrices: dict[str, dict[str, Any]] = {}
    endpoints = {
        "B_blur_mean": "blur_or_soft_fusion_clear_bad",
        "C_c3_low_jump": "all_clear_bad",
    }
    for candidate in CANDIDATES:
        codes = np.full((len(SEEDS), len(CLASSES)), -1, dtype=np.int8)
        scores = np.empty((len(SEEDS), len(CLASSES)), dtype=np.float64)
        for row in ordered:
            seed_index = row["global_seed"] - SEEDS[0]
            slot = row["class_slot"]
            if row["final_severity"] == "clean_good":
                codes[seed_index, slot] = 0
            elif _positive(row, endpoints[candidate]):
                codes[seed_index, slot] = 1
            scores[seed_index, slot] = _oriented_score(row, candidate)
        orders: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        denominator = 0
        observed = 0.0
        for slot in range(len(CLASSES)):
            order, start = _tie_group_starts(scores[:, slot])
            orders.append(order)
            starts.append(start)
            code = codes[:, slot][None, :]
            observed += float(_batch_concordance(code, order, start)[0])
            denominator += int(np.sum(code == 1) * np.sum(code == 0))
        if denominator == 0:
            raise RuntimeError(f"zero primary pair denominator: {candidate}")
        matrices[candidate] = {
            "codes": codes,
            "orders": orders,
            "starts": starts,
            "observed_concordance": observed,
            "pair_count": denominator,
        }

    rng = np.random.default_rng(PERMUTATION_SEED)
    exceedances = {candidate: 0 for candidate in CANDIDATES}
    remaining = draws
    while remaining:
        size = min(PERMUTATION_BATCH, remaining)
        permutations = np.empty((size, len(SEEDS)), dtype=np.int32)
        for index in range(size):
            permutations[index] = rng.permutation(len(SEEDS))
        for candidate in CANDIDATES:
            data = matrices[candidate]
            total = np.zeros(size, dtype=np.float64)
            for slot in range(len(CLASSES)):
                permuted = data["codes"][permutations, slot]
                total += _batch_concordance(
                    permuted, data["orders"][slot], data["starts"][slot]
                )
            exceedances[candidate] += int(
                np.sum(total >= data["observed_concordance"])
            )
        remaining -= size
    return {
        candidate: {
            "observed_auc": float(
                matrices[candidate]["observed_concordance"]
                / matrices[candidate]["pair_count"]
            ),
            "pair_count": matrices[candidate]["pair_count"],
            "draws": draws,
            "exceedances": exceedances[candidate],
            "raw_p_value": float((1 + exceedances[candidate]) / (1 + draws)),
        }
        for candidate in CANDIDATES
    }


def holm_adjust(raw: Mapping[str, float]) -> dict[str, float]:
    if set(raw) != set(CANDIDATES):
        raise RuntimeError("Holm family must contain exactly B and C")
    ordered = sorted(CANDIDATES, key=lambda name: (raw[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    family = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (family - index) * float(raw[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def operating_point(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    candidate: str,
    alpha: str,
    endpoint: str,
) -> dict[str, Any]:
    tp = fp = alert_count = excluded_alert = positives = negatives = 0
    per_class: list[dict[str, int]] = []
    for class_id in CLASSES:
        class_tp = class_fp = class_alert = 0
        threshold = float(thresholds[candidate]["classes"][str(class_id)][alpha]["threshold"])
        for row in rows:
            if row["class_id"] != class_id:
                continue
            raw = float(row["B_score"] if candidate == "B_blur_mean" else row["C_score"])
            alert = raw > threshold if candidate == "B_blur_mean" else raw < threshold
            positive = _positive(row, endpoint)
            negative = row["final_severity"] == "clean_good"
            positives += int(positive)
            negatives += int(negative)
            if alert:
                alert_count += 1
                class_alert += 1
                if positive:
                    tp += 1
                    class_tp += 1
                elif negative:
                    fp += 1
                    class_fp += 1
                else:
                    excluded_alert += 1
        per_class.append(
            {
                "class_id": class_id,
                "alert_count": class_alert,
                "true_positive_count": class_tp,
                "false_positive_count": class_fp,
            }
        )
    if positives == 0 or negatives == 0:
        raise RuntimeError(f"zero TPR/FPR denominator: {candidate}/{alpha}")
    return {
        "candidate": candidate,
        "alpha": alpha,
        "comparison": (
            "raw_score > class-specific threshold"
            if candidate == "B_blur_mean"
            else "raw_score < class-specific threshold"
        ),
        "positive_endpoint": f"{endpoint}_vs_clean_good",
        "positive_count": positives,
        "clean_good_count": negatives,
        "alert_count_all_trajectories": alert_count,
        "alert_rate_all_trajectories": float(alert_count / len(rows)),
        "excluded_mild_or_nonendpoint_alert_count": excluded_alert,
        "true_positive_count": tp,
        "false_positive_count": fp,
        "micro_TPR": float(tp / positives),
        "micro_FPR": float(fp / negatives),
        "per_class_counts": per_class,
    }


def _suppressed_guardrail(summary: dict[str, Any]) -> dict[str, Any]:
    result = dict(summary)
    result.pop("concordant_pair_credit", None)
    if result["positive_count"] < 5 or result["clean_good_count"] < 5:
        result["auc"] = None
        result["numeric_auc_suppressed"] = True
    else:
        result["numeric_auc_suppressed"] = False
    return result


def evaluate_joined(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    *,
    permutation_draws: int = PERMUTATION_DRAWS,
) -> dict[str, Any]:
    primary = {
        "B_blur_mean": auc_summary(
            rows,
            candidate="B_blur_mean",
            endpoint="blur_or_soft_fusion_clear_bad",
        ),
        "C_c3_low_jump": auc_summary(
            rows, candidate="C_c3_low_jump", endpoint="all_clear_bad"
        ),
    }
    for candidate in CANDIDATES:
        if primary[candidate]["auc"] is None:
            raise RuntimeError(f"primary AUC undefined: {candidate}")
    permutation = permutation_p_values(rows, draws=permutation_draws)
    for candidate in CANDIDATES:
        if not math.isclose(
            float(primary[candidate]["auc"]),
            float(permutation[candidate]["observed_auc"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"independent observed-AUC replay failed: {candidate}")
    raw = {candidate: permutation[candidate]["raw_p_value"] for candidate in CANDIDATES}
    adjusted = holm_adjust(raw)
    for candidate in CANDIDATES:
        permutation[candidate]["holm_adjusted_p_value"] = adjusted[candidate]

    operating = {
        candidate: {
            alpha: operating_point(
                rows,
                thresholds,
                candidate=candidate,
                alpha=alpha,
                endpoint=(
                    "blur_or_soft_fusion_clear_bad"
                    if candidate == "B_blur_mean"
                    else "all_clear_bad"
                ),
            )
            for alpha in ALPHAS
        }
        for candidate in CANDIDATES
    }
    b010 = operating["B_blur_mean"]["alpha_0p10"]
    b_pass = bool(
        primary["B_blur_mean"]["auc"] >= 0.75
        and adjusted["B_blur_mean"] < 0.05
        and b010["true_positive_count"] >= 3
        and b010["micro_TPR"] > b010["micro_FPR"]
    )
    c_pass = bool(
        primary["C_c3_low_jump"]["auc"] >= 0.70
        and adjusted["C_c3_low_jump"] < 0.05
    )
    guardrails: dict[str, Any] = {
        "B_all_clear_bad": _suppressed_guardrail(
            auc_summary(rows, candidate="B_blur_mean", endpoint="all_clear_bad")
        ),
        "C_blur_or_soft_fusion_clear_bad": _suppressed_guardrail(
            auc_summary(
                rows,
                candidate="C_c3_low_jump",
                endpoint="blur_or_soft_fusion_clear_bad",
            )
        ),
    }
    for candidate in CANDIDATES:
        for endpoint in (
            "mixed_blur_and_structure_clear_bad",
            "structural_non_blur_clear_bad",
        ):
            guardrails[f"{candidate}_{endpoint}"] = _suppressed_guardrail(
                auc_summary(rows, candidate=candidate, endpoint=endpoint)
            )
    return {
        "primary_auc": primary,
        "permutation": permutation,
        "holm_family": list(CANDIDATES),
        "operating_points": operating,
        "guardrails": guardrails,
        "candidate_decisions": {
            "B_blur_mean": {
                "all_frozen_gates_pass": b_pass,
                "blur_specific_intervention_experiment_authorized": b_pass,
                "universal_artifact_claim_authorized": False,
            },
            "C_c3_low_jump": {
                "all_frozen_gates_pass": c_pass,
                "mechanism_support": c_pass,
                "intervention_experiment_authorized": False,
            },
        },
        "candidate_combination_performed": False,
    }


def run_stage_b(
    *,
    input_lock: Path,
    stage_a_receipt: Path,
    output: Path,
    _self_test_permutation_draws: int | None = None,
) -> Path:
    binding, binding_manifest = validate_input_binding(input_lock)
    stage_a, stage_a_manifest = validate_stage_a_receipt(
        stage_a_receipt, binding, binding_manifest
    )
    pool_trace_lineage, pool_lineage = validate_sampling_pool(
        binding["inputs"]["sampling_pool"]
    )
    if (
        pool_lineage["pool_identity_sha256"]
        != binding["inputs"]["sampling_pool"]["manifest_identity_sha256"]
    ):
        raise RuntimeError("sampling-pool identity differs from input binding")
    consensus_rows, consensus_lineage = load_full_consensus(
        binding["inputs"]["consensus"],
        {
            **stage_a["consensus_receipt"],
            "counts": stage_a["aggregate_counts"],
        },
    )
    c_values, primary_lineage, primary_trace_lineage = load_feature_product(
        binding["inputs"]["primary_label_free_product"], candidate="C_c3_low_jump"
    )
    b_values, visual_lineage, visual_trace_lineage = load_feature_product(
        binding["inputs"]["visual_label_free_product"], candidate="B_blur_mean"
    )
    if (
        primary_trace_lineage != pool_trace_lineage
        or visual_trace_lineage != pool_trace_lineage
    ):
        raise RuntimeError(
            "primary/visual label-free products are not derived from the exact bound sampling pool"
        )
    joined: list[dict[str, Any]] = []
    for row in consensus_rows:
        key = (row["global_seed"], row["class_slot"], row["class_id"])
        joined.append({**row, "B_score": b_values[key], "C_score": c_values[key]})
    if len(joined) != TRAJECTORY_COUNT:
        raise RuntimeError("score-label join lost or multiplied rows")

    foundations = validate_foundation_locks()
    thresholds = foundations["thresholds"].get("thresholds", {})
    if set(thresholds) != set(CANDIDATES):
        raise RuntimeError("pinned threshold candidate family changed")
    draws = (
        PERMUTATION_DRAWS
        if _self_test_permutation_draws is None
        else _self_test_permutation_draws
    )
    if _self_test_permutation_draws is not None and os.environ.get(
        "EQVAE_ALLOW_SYNTHETIC_SELF_TEST_DRAWS"
    ) != "1":
        raise RuntimeError("reduced draws are reserved for the internal synthetic self-test")
    statistics = evaluate_joined(joined, thresholds, permutation_draws=draws)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": (
            "COMPLETE_SYNTHETIC_SELF_TEST"
            if _self_test_permutation_draws is not None
            else "COMPLETE_FROZEN_THIRD_POOL_CONFIRMATION"
        ),
        "scientific_contract_identity_sha256": binding[
            "scientific_contract_identity_sha256"
        ],
        "input_binding_identity_sha256": binding["identity_sha256"],
        "input_binding_manifest_identity_sha256": binding_manifest["identity_sha256"],
        "stage_a_receipt_identity_sha256": stage_a["identity_sha256"],
        "stage_a_manifest_identity_sha256": stage_a_manifest["identity_sha256"],
        "foundation_identity_pins": binding["foundation_identity_pins"],
        "input_lineage": {
            "sampling_pool": pool_lineage,
            "consensus": consensus_lineage,
            "primary_label_free_product": primary_lineage,
            "visual_label_free_product": visual_lineage,
        },
        "aggregate_counts": stage_a["aggregate_counts"],
        "event_gate": stage_a["event_gate"],
        "statistics": statistics,
        "permutation_contract": {
            "draws": draws,
            "production_draws_frozen": PERMUTATION_DRAWS,
            "rng_seed": PERMUTATION_SEED,
            "complete_seed_blocks_permuted": True,
            "same_permutation_used_for_B_and_C": True,
        },
        "access_audit": {
            "stage_A_was_separate_completed_process": True,
            "stage_A_feature_paths_unopened": True,
            "stage_B_opened_consensus_rows_after_gate": True,
            "stage_B_opened_only_two_bound_label_free_feature_products": True,
            "old_labels_images_or_screen_results_opened": False,
            "third_pool_endpoint_images_opened": False,
            "individual_rows_scores_ranks_or_permutations_emitted": False,
        },
        "output_scope": "aggregate-only; no sampling or intervention performed",
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["identity_sha256"] = canonical_sha256(result)
    return publish_record_lock(
        output,
        artifact_kind="dit_bad_good_third_pool_confirmation_result_v1",
        record_name="confirmation_result.json",
        record=result,
        source_copies={"evaluator_source.py": Path(__file__).resolve()},
    )


def _synthetic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for slot, class_id in enumerate(CLASSES):
            index = (seed - SEEDS[0]) * len(CLASSES) + slot
            local = seed - SEEDS[0]
            severity = "clean_good"
            blur = False
            structure = False
            if slot == 0 and local < 15:
                severity = "clear_bad"
                blur = True
                structure = local < 5
            elif slot == 1 and local < 10:
                severity = "clear_bad"
                structure = True
            elif slot == 2 and local < 5:
                severity = "clear_bad"
            elif local in range(20, 25):
                severity = "mild_or_disputed"
            rows.append(
                {
                    "sample_index": index,
                    "global_seed": seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "final_severity": severity,
                    "blur_component_consensus": blur,
                    "discrete_structure_component_consensus": structure,
                    "B_score": float((local % 37) / 37.0 + (2.0 if blur else 0.0)),
                    "C_score": float((local % 41) / 41.0 - (1.0 if severity == "clear_bad" else 0.0)),
                }
            )
    return rows


def _write_synthetic_consensus_lock(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    sampling_pool_identity_sha256: str,
    append_extra_cell: bool = False,
) -> str:
    """Self-test fixture matching the future aggregate/row separation contract."""

    root.mkdir()
    rows_path = root / CONSENSUS_ROWS_NAME
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONSENSUS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_index": row["sample_index"],
                    "global_seed": row["global_seed"],
                    "class_slot": row["class_slot"],
                    "class_id": row["class_id"],
                    "final_severity": row["final_severity"],
                    "blur_component_consensus": str(
                        bool(row["blur_component_consensus"])
                    ).lower(),
                    "discrete_structure_component_consensus": str(
                        bool(row["discrete_structure_component_consensus"])
                    ).lower(),
                }
            )
    if append_extra_cell:
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        lines[1] += ",unexpected"
        rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    counts = validate_aggregate_counts(_count_from_rows(rows))
    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_THIRD_POOL_BLIND_CONSENSUS_COUNTS",
        "experiment": CONSENSUS_EXPERIMENT,
        "phase1_protocol_identity_sha256": EXPECTED_PHASE1_PROTOCOL_IDENTITY,
        "sampling_protocol_identity_sha256": EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
        "sampling_pool_identity_sha256": sampling_pool_identity_sha256,
        "classes_ordered": list(CLASSES),
        "global_seeds": list(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
        "labels_and_phenotypes_immutable": True,
        "three_independent_endpoint_only_reviewers": True,
        "reviewers_score_threshold_alert_trajectory_blind": True,
        "consensus_rows_member": CONSENSUS_ROWS_NAME,
        "consensus_rows_file_sha256": sha256_file(rows_path),
        "counts": counts,
    }
    aggregate["identity_sha256"] = canonical_sha256(aggregate)
    aggregate_path = root / CONSENSUS_AGGREGATE_NAME
    write_json(aggregate_path, aggregate)
    files = []
    for path in (aggregate_path, rows_path):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "experiment": CONSENSUS_EXPERIMENT,
        "aggregate_counts_identity_sha256": aggregate["identity_sha256"],
        "files": sorted(files, key=lambda row: row["name"]),
    }
    manifest["identity_sha256"] = canonical_sha256(manifest)
    manifest_path = root / "manifest.json"
    write_json(manifest_path, manifest)
    write_json(
        root / "completion.json",
        {
            "complete": True,
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "aggregate_counts_file_sha256": sha256_file(aggregate_path),
            "aggregate_counts_identity_sha256": aggregate["identity_sha256"],
        },
    )
    return manifest["identity_sha256"]


def self_test() -> None:
    foundations = validate_foundation_locks()
    contract = scientific_contract()
    assert contract["foundation_identity_pins"]["sampling_protocol_identity_sha256"] == (
        EXPECTED_SAMPLING_PROTOCOL_IDENTITY
    )
    codes = np.asarray([[0, 1, -1, 0, 1]], dtype=np.int8)
    order, starts = _tie_group_starts(np.asarray([0.0, 2.0, 1.0, 3.0, 2.0]))
    assert _batch_concordance(codes, order, starts).shape == (1,)
    assert holm_adjust({"B_blur_mean": 0.01, "C_c3_low_jump": 0.03}) == {
        "B_blur_mean": 0.02,
        "C_c3_low_jump": 0.03,
    }
    rows = _synthetic_rows()
    counts = validate_aggregate_counts(_count_from_rows(rows))
    assert counts["overall"]["blur_or_soft_fusion_clear_bad"] == 15
    assert counts["overall"]["clear_bad"] == 30
    thresholds = foundations["thresholds"]["thresholds"]
    stats = evaluate_joined(rows, thresholds, permutation_draws=31)
    assert set(stats["permutation"]) == set(CANDIDATES)
    assert stats["candidate_combination_performed"] is False
    assert scientific_contract()["statistics"]["permutation"]["draws"] == 100_000
    # Exercise the actual Stage-A boundary with deliberately nonexistent feature
    # roots.  A below-gate receipt must succeed, and Stage B must reject it before
    # it can touch either poisoned path.
    with tempfile.TemporaryDirectory(prefix="third-pool-evaluator-selftest-") as tmp:
        temporary = Path(tmp)
        source_lock = freeze_source_lock(temporary / "source_lock")
        failed_rows = [dict(row) for row in rows]
        first_blur = next(
            row
            for row in failed_rows
            if row["final_severity"] == "clear_bad"
            and row["blur_component_consensus"]
        )
        first_blur["final_severity"] = "clean_good"
        first_blur["blur_component_consensus"] = False
        first_blur["discrete_structure_component_consensus"] = False
        consensus_root = temporary / "consensus"
        pool_identity = "c" * 64
        consensus_id = _write_synthetic_consensus_lock(
            consensus_root,
            failed_rows,
            sampling_pool_identity_sha256=pool_identity,
        )
        poison_pool = temporary / "MUST_NOT_EXIST_pool"
        poison_primary = temporary / "MUST_NOT_EXIST_primary"
        poison_visual = temporary / "MUST_NOT_EXIST_visual"
        binding_root = bind_inputs(
            source_lock=source_lock,
            sampling_pool_path=poison_pool,
            sampling_pool_manifest_identity=pool_identity,
            consensus_path=consensus_root,
            consensus_manifest_identity=consensus_id,
            primary_path=poison_primary,
            primary_manifest_identity="a" * 64,
            visual_path=poison_visual,
            visual_manifest_identity="b" * 64,
            output=temporary / "binding",
        )
        assert (
            not poison_pool.exists()
            and not poison_primary.exists()
            and not poison_visual.exists()
        )
        stage_a_root = run_stage_a(
            input_lock=binding_root, output=temporary / "stage_a"
        )
        receipt, _ = validate_record_lock(
            stage_a_root,
            artifact_kind="dit_bad_good_third_pool_stage_a_event_gate_v1",
            record_name="stage_a_gate_receipt.json",
        )
        assert receipt["status"] == "EVENT_GATE_FAILED_NO_SCORE_ACCESS"
        assert receipt["access_audit"]["score_csv_or_npz_opened"] is False
        forged = dict(receipt)
        forged["status"] = "EVENT_GATE_PASSED_SCORES_STILL_UNOPENED"
        forged["event_gate"] = {
            **forged["event_gate"],
            "observed_blur_or_soft_fusion_clear_bad": EVENT_MIN_BLUR,
            "blur_minimum_met": True,
            "observed_total_clear_bad": EVENT_MIN_TOTAL_BAD,
            "total_bad_minimum_met": True,
            "both_minima_met": True,
            "stage_B_authorized": True,
        }
        forged.pop("identity_sha256")
        forged["identity_sha256"] = canonical_sha256(forged)
        forged_root = publish_record_lock(
            temporary / "forged_stage_a",
            artifact_kind="dit_bad_good_third_pool_stage_a_event_gate_v1",
            record_name="stage_a_gate_receipt.json",
            record=forged,
            source_copies={"evaluator_source.py": Path(__file__).resolve()},
        )
        binding, binding_manifest = validate_input_binding(binding_root)
        try:
            validate_stage_a_receipt(forged_root, binding, binding_manifest)
        except RuntimeError as exc:
            assert "does not replay aggregate counts" in str(exc)
        else:
            raise AssertionError("forged Stage-A pass fields bypassed count replay")
        try:
            run_stage_b(
                input_lock=binding_root,
                stage_a_receipt=stage_a_root,
                output=temporary / "must_not_publish",
            )
        except RuntimeError as exc:
            assert "not authorized" in str(exc)
        else:
            raise AssertionError("below-gate Stage B was not rejected")
        assert (
            not poison_pool.exists()
            and not poison_primary.exists()
            and not poison_visual.exists()
        )
        (consensus_root / "unexpected.txt").write_text(
            "must be rejected before payload validation\n", encoding="utf-8"
        )
        try:
            load_consensus_aggregate_only(
                {
                    "path": str(consensus_root),
                    "manifest_identity_sha256": consensus_id,
                },
                {"manifest_identity_sha256": pool_identity},
            )
        except RuntimeError as exc:
            assert "exactly the frozen four files" in str(exc)
        else:
            raise AssertionError("extra consensus member was accepted")
        (consensus_root / "unexpected.txt").unlink()

        malformed_root = temporary / "malformed_consensus"
        malformed_id = _write_synthetic_consensus_lock(
            malformed_root,
            rows,
            sampling_pool_identity_sha256=pool_identity,
            append_extra_cell=True,
        )
        malformed_receipt = load_consensus_aggregate_only(
            {
                "path": str(malformed_root),
                "manifest_identity_sha256": malformed_id,
            },
            {"manifest_identity_sha256": pool_identity},
        )
        try:
            load_full_consensus(
                {
                    "path": str(malformed_root),
                    "manifest_identity_sha256": malformed_id,
                },
                {
                    **malformed_receipt,
                    "counts": malformed_receipt["counts"],
                },
            )
        except RuntimeError as exc:
            assert "missing or extra cells" in str(exc)
        else:
            raise AssertionError("extra consensus CSV cell was accepted")
    print(
        "self-test passed: exact foundation pins, two-stage contract, count partition, "
        "tie-aware pair AUC, intact-block permutation core, Holm family, strict locked "
        "operating points, poisoned-path Stage-A non-access, forged-gate count replay, "
        "below-gate Stage-B rejection, exact consensus tree/row width, "
        "and aggregate-only decisions"
    )


def _print_lock(path: Path, artifact_kind: str, record_name: str) -> None:
    record, manifest = validate_record_lock(
        path, artifact_kind=artifact_kind, record_name=record_name
    )
    print(
        json.dumps(
            {
                "output": str(Path(path).expanduser().absolute()),
                "record_identity_sha256": record["identity_sha256"],
                "manifest_identity_sha256": manifest["identity_sha256"],
                "status": record["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--freeze-source-lock", action="store_true")
    modes.add_argument("--validate-source-lock", action="store_true")
    modes.add_argument("--bind-inputs", action="store_true")
    modes.add_argument("--stage-a", action="store_true")
    modes.add_argument("--stage-b", action="store_true")
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--input-lock", type=Path)
    parser.add_argument("--stage-a-receipt", type=Path)
    parser.add_argument("--sampling-pool-path", type=Path)
    parser.add_argument("--sampling-pool-manifest-identity")
    parser.add_argument("--consensus-path", type=Path)
    parser.add_argument("--consensus-manifest-identity")
    parser.add_argument("--primary-path", type=Path)
    parser.add_argument("--primary-manifest-identity")
    parser.add_argument("--visual-path", type=Path)
    parser.add_argument("--visual-manifest-identity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.freeze_source_lock:
        if args.output is None:
            parser.error("--freeze-source-lock requires --output")
        path = freeze_source_lock(args.output)
        _print_lock(
            path,
            "dit_bad_good_third_pool_evaluation_source_lock_v1",
            "scientific_contract.json",
        )
        return 0
    if args.validate_source_lock:
        contract, manifest = validate_source_lock(args.source_lock)
        print(
            json.dumps(
                {
                    "output": str(args.source_lock.expanduser().absolute()),
                    "scientific_contract_identity_sha256": contract["identity_sha256"],
                    "manifest_identity_sha256": manifest["identity_sha256"],
                    "status": "valid",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.bind_inputs:
        required = {
            "--sampling-pool-path": args.sampling_pool_path,
            "--sampling-pool-manifest-identity": args.sampling_pool_manifest_identity,
            "--consensus-path": args.consensus_path,
            "--consensus-manifest-identity": args.consensus_manifest_identity,
            "--primary-path": args.primary_path,
            "--primary-manifest-identity": args.primary_manifest_identity,
            "--visual-path": args.visual_path,
            "--visual-manifest-identity": args.visual_manifest_identity,
            "--output": args.output,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("--bind-inputs missing: " + ", ".join(missing))
        path = bind_inputs(
            source_lock=args.source_lock,
            sampling_pool_path=args.sampling_pool_path,
            sampling_pool_manifest_identity=args.sampling_pool_manifest_identity,
            consensus_path=args.consensus_path,
            consensus_manifest_identity=args.consensus_manifest_identity,
            primary_path=args.primary_path,
            primary_manifest_identity=args.primary_manifest_identity,
            visual_path=args.visual_path,
            visual_manifest_identity=args.visual_manifest_identity,
            output=args.output,
        )
        _print_lock(
            path,
            "dit_bad_good_third_pool_evaluation_input_binding_v1",
            "input_binding.json",
        )
        return 0
    if args.stage_a:
        if args.input_lock is None or args.output is None:
            parser.error("--stage-a requires --input-lock and --output")
        path = run_stage_a(input_lock=args.input_lock, output=args.output)
        _print_lock(
            path,
            "dit_bad_good_third_pool_stage_a_event_gate_v1",
            "stage_a_gate_receipt.json",
        )
        return 0
    if args.stage_b:
        if args.input_lock is None or args.stage_a_receipt is None or args.output is None:
            parser.error(
                "--stage-b requires --input-lock, --stage-a-receipt, and --output"
            )
        path = run_stage_b(
            input_lock=args.input_lock,
            stage_a_receipt=args.stage_a_receipt,
            output=args.output,
        )
        _print_lock(
            path,
            "dit_bad_good_third_pool_confirmation_result_v1",
            "confirmation_result.json",
        )
        return 0
    raise AssertionError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
