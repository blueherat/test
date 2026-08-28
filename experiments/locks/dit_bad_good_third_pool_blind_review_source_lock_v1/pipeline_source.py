#!/usr/bin/env python3
"""Frozen endpoint-only blind-review pipeline for the DiT third pool.

Phase 0 freezes this source before a third-pool sampling identity, endpoint image,
review, feature product, score, or label is opened.  Runtime modes form a strict
chain:

1. build one endpoint-only review pack from completed pool receipts, per-seed
   manifests, and the three terminal PNGs only;
2. lock three independent reviewer sheets separately;
3. derive a raw-majority-clear-bad-only adjudication pack;
4. lock retain/downgrade decisions (promotion is structurally impossible); and
5. publish the evaluator-v5 consensus CSV and aggregate-count lock.

No post-freeze pack/review/adjudication/consensus mode opens trace.npz, feature
products, score tables, thresholds, alerts, previous labels, or screening
results.  Phase-0 source freezing alone verifies the already-frozen upstream
locks.  The self-test uses a complete synthetic 3 x 600 cohort and deliberately
omits every forbidden payload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo

try:
    from . import evaluate_dit_bad_good_third_pool_confirmation as evaluation
except ImportError:  # pragma: no cover - direct CLI execution.
    import evaluate_dit_bad_good_third_pool_confirmation as evaluation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_blind_review_source_lock_v1"
)
EVALUATOR_SOURCE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_evaluation_source_lock_v5"
)

EXPECTED_EVALUATOR_CONTRACT_IDENTITY = (
    "6638f75eef792fa313fa14ebb0b6c65a696dab881c193f2bf8fa83615e1475e2"
)
EXPECTED_EVALUATOR_MANIFEST_IDENTITY = (
    "d7467275fab416a5eddadf528fd24b98ffb6bfeed499711c3e8ba6b6f72cd6e8"
)
EXPECTED_EVALUATOR_SOURCE_SHA256 = (
    "006a9337295d1a3f27ad8626fdff21d227038cf11f291a769dde1af8c41aff5c"
)

EXPECTED_PHENOTYPE_CONTRACT = {
    "adjudication": {
        "allowed_actions": ["retain_clear_bad", "downgrade_to_mild"],
        "candidate_hypothesis_blind": True,
        "promotion_allowed": False,
        "scope": "raw majority clear-bad only",
    },
    "blur_or_soft_fusion_positive": (
        "final retained clear-bad and blur-component consensus present; mixed "
        "blur-plus-discrete-structure failures are included and also reported separately"
    ),
    "component_consensus": (
        "For each broad component group, presence requires at least two of three "
        "reviewers to mark any member of that group."
    ),
    "frozen_before_third_pool_images_are_reviewed": True,
    "model_relative_quality_rule": (
        "Severity 2/3 is reserved for a clear defect materially below the normal "
        "quality of this frozen model/class pool; ordinary model limitations, slight "
        "softness, distant subjects, and merely imperfect details are not clear-bad."
    ),
    "phenotype_disputed": (
        "final retained clear-bad without the required component consensus; retained "
        "in candidate C's all-bad endpoint but excluded from subtype-only cuts"
    ),
    "review_components": {
        "additional_components": ["texture_break", "other", "none"],
        "blur_components": [
            "global_blur",
            "local_blur",
            "soft_fusion_or_melting",
        ],
        "discrete_structure_components": [
            "discrete_duplication_or_extra_part",
            "detachment_or_floating_part",
            "topology_or_attachment_error",
            "limb_or_object_misalignment",
        ],
    },
    "reviewer_candidate_hypothesis_blind": True,
    "reviewer_metric_score_threshold_alert_and_trajectory_blind": True,
    "severity_consensus": {
        "clean_good": "at least two of three independent severity scores are 0",
        "clear_bad": "at least two of three independent severity scores are 2 or 3",
        "mild_or_disputed": "neither majority; excluded from binary endpoints",
    },
    "structural_non_blur": (
        "final retained clear-bad, no blur-component consensus, and "
        "discrete-structure-component consensus present"
    ),
    "three_independent_endpoint_only_reviewers": True,
}

CLASSES = evaluation.CLASSES
SEEDS = evaluation.SEEDS
TRAJECTORY_COUNT = evaluation.TRAJECTORY_COUNT
REVIEWERS = ("reviewer_1", "reviewer_2", "reviewer_3")
TILE_SIZE = 256
GRID_COLUMNS = 5
GRID_ROWS = 4
GRID_BLOCK_SIZE = GRID_COLUMNS * GRID_ROWS
GRID_LABEL_HEIGHT = 24

COMPONENT_FLAGS = (
    "global_blur",
    "local_blur",
    "soft_fusion_or_melting",
    "discrete_duplication_or_extra_part",
    "detachment_or_floating_part",
    "topology_or_attachment_error",
    "limb_or_object_misalignment",
    "texture_break",
    "other",
)
NONE_SENTINEL = "none"
BLUR_COMPONENTS = frozenset(COMPONENT_FLAGS[:3])
DISCRETE_STRUCTURE_COMPONENTS = frozenset(COMPONENT_FLAGS[3:7])
REVIEW_COLUMNS = ("review_id", "severity", *COMPONENT_FLAGS, NONE_SENTINEL)
REVIEW_ORDER_COLUMNS = (
    "reviewer_id",
    "review_order_index",
    "review_id",
    "class_id",
    "native_image_relative_path",
    "grid_relative_path",
    "grid_cell_index",
)
SAMPLE_CATALOG_COLUMNS = (
    "sample_index",
    "global_seed",
    "class_slot",
    "class_id",
    "native_image_relative_path",
    "source_seed_relative_path",
    "source_endpoint_relative_path",
    "source_seed_identity_sha256",
    "source_seed_manifest_sha256",
    "source_png_bytes",
    "source_png_sha256",
    "source_pixel_sha256",
)
ADJUDICATION_CASE_COLUMNS = (
    "case_id",
    "class_id",
    "native_image_relative_path",
)
ADJUDICATION_COLUMNS = ("case_id", "action")
ADJUDICATION_ACTIONS = ("retain_clear_bad", "downgrade_to_mild")
CONSENSUS_COLUMNS = evaluation.CONSENSUS_COLUMNS

SEVERITY_RUBRIC = {
    "0": (
        "Acceptable/normal for this frozen model and class pool: no clear defect "
        "materially below its ordinary sampling quality."
    ),
    "1": (
        "Mild or ambiguous imperfection: visible on inspection but plausibly ordinary "
        "model softness, crop, pose, occlusion, distance, or imperfect detail."
    ),
    "2": (
        "Clear material defect below the frozen model/class norm, usually localized, "
        "while the intended subject remains recognizable."
    ),
    "3": (
        "Severe/gross defect below the frozen model/class norm that dominates the image "
        "or substantially breaks subject/object coherence."
    ),
}
REVIEWER_INSTRUCTIONS = """# Frozen endpoint-only review rubric

Review each endpoint independently at native 256x256 resolution. Class-specific
20-image grids provide only the frozen model/class quality reference band; use the
individual native file whenever a grid cell is ambiguous.

Severity:

- 0: Acceptable/normal for this frozen model and class pool; no clear defect
  materially below ordinary sampling quality.
- 1: Mild or ambiguous imperfection, including ordinary softness, crop, pose,
  occlusion, distance, or merely imperfect detail.
- 2: Clear material defect below the model/class norm, usually localized, while
  the intended subject remains recognizable.
- 3: Severe/gross defect that dominates the image or substantially breaks
  subject/object coherence.

Set every applicable defect component to 1. Set `none=1` exactly when all nine
defect components are 0; otherwise set `none=0`. Do not infer candidate metrics,
trajectory behavior, or intervention suitability. Review independently without
seeing another reviewer's rows or summary.
"""
ADJUDICATION_INSTRUCTIONS = """# Frozen conservative adjudication rubric

Only raw-majority clear-bad cases appear here. For every case choose exactly one:

- `retain_clear_bad`: the endpoint has a clear defect materially below the normal
  frozen model/class quality band.
- `downgrade_to_mild`: crop, motion, pose, occlusion, distance, ordinary model
  texture/softness, or ambiguity makes clear-bad unsupported.

Promotion is impossible: no non-raw-clear-bad case is exposed, and no action can
create a new clear-bad label. Do not inspect metrics, scores, thresholds, alerts,
trajectories, feature products, old labels, or screening results.
"""

EXPECTED_SEED_OUTPUTS = {
    "custom_baseline_helper.py",
    "images/00_class0207.png",
    "images/01_class0602.png",
    "images/02_class0795.png",
    "runner_source.py",
    "sample.png",
    "strict_reproduction_helper.py",
    "trace.npz",
}
ENDPOINT_RELATIVE = {
    207: "images/00_class0207.png",
    602: "images/01_class0602.png",
    795: "images/02_class0795.png",
}
HEX64 = evaluation.HEX64


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return evaluation.canonical_sha256(value)


def without_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("identity_sha256", None)
    return result


def write_json(path: Path, value: Any) -> None:
    evaluation.write_json(path, value)


def load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {description}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be a JSON object: {path}")
    return value


def require_hex64(value: Any, description: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise RuntimeError(f"{description} must be a lowercase 64-hex SHA-256")
    return value


def require_directory(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real non-symlink directory: {path}")
    return path.resolve()


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().absolute()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a regular non-symlink file: {path}")
    return path.resolve()


def require_output_target(path: Path) -> Path:
    output = path.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if set(row) != set(columns):
                raise RuntimeError(f"CSV row does not match frozen schema: {path}")
            writer.writerow({column: row[column] for column in columns})


def read_csv(path: Path, columns: Sequence[str], description: str) -> list[dict[str, str]]:
    path = require_file(path, description)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise RuntimeError(f"{description} columns/order changed")
        rows = []
        for row in reader:
            if None in row or set(row) != set(columns) or any(
                value is None for value in row.values()
            ):
                raise RuntimeError(f"{description} row has missing or extra cells")
            rows.append(dict(row))
    return rows


def _artifact_records(root: Path) -> list[dict[str, Any]]:
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


def _safe_manifest_map(manifest: Mapping[str, Any], description: str) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{description} file list is malformed")
    result = {str(row.get("name")): dict(row) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{description} has duplicate member names")
    for name, row in result.items():
        relative = Path(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or set(row) != {"name", "bytes", "sha256"}
            or isinstance(row.get("bytes"), bool)
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] < 0
        ):
            raise RuntimeError(f"unsafe {description} member: {name!r}")
        require_hex64(row.get("sha256"), f"{description} member hash")
    return result


def _preflight_manifest_members(root: Path, by_name: Mapping[str, Any], description: str) -> None:
    """Reject unknown members before opening/hashing any non-metadata member."""

    expected = set(by_name) | {"manifest.json", "completion.json"}
    observed: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"{description} contains a symlink: {relative}")
        if path.is_file():
            observed.add(relative)
        elif not path.is_dir():
            raise RuntimeError(f"{description} contains a special entry: {relative}")
    if observed != expected:
        raise RuntimeError(f"{description} member set differs from its manifest")


def _preflight_exact_names(
    root: Path, expected_files: set[str], description: str
) -> Path:
    """Name-only allowlist check performed before any payload member is opened."""

    root = require_directory(root, description)
    expected_dirs = {
        parent.as_posix()
        for name in expected_files
        for parent in Path(name).parents
        if parent != Path(".")
    }
    observed_files: set[str] = set()
    observed_dirs: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError(f"{description} contains a symlink: {relative}")
        if path.is_file():
            observed_files.add(relative)
        elif path.is_dir():
            observed_dirs.add(relative)
        else:
            raise RuntimeError(f"{description} contains a special entry: {relative}")
    if observed_files != expected_files or observed_dirs != expected_dirs:
        raise RuntimeError(f"{description} differs from its frozen exact tree")
    return root


def finalize_record_staging(
    staging: Path,
    output: Path,
    *,
    artifact_kind: str,
    record_name: str,
    record: Mapping[str, Any],
) -> Path:
    write_json(staging / record_name, record)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "artifact_kind": artifact_kind,
        "primary_record_name": record_name,
        "primary_record_identity_sha256": record["identity_sha256"],
        "files": _artifact_records(staging),
    }
    manifest["identity_sha256"] = canonical_sha256(manifest)
    write_json(staging / "manifest.json", manifest)
    write_json(
        staging / "completion.json",
        {
            "complete": True,
            "artifact_kind": artifact_kind,
            "primary_record_name": record_name,
            "primary_record_file_sha256": sha256_file(staging / record_name),
            "primary_record_identity_sha256": record["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
        },
    )
    os.replace(staging, output)
    validate_record_lock(
        output,
        artifact_kind=artifact_kind,
        record_name=record_name,
    )
    return output.resolve()


def validate_record_lock(
    root: Path,
    *,
    artifact_kind: str,
    record_name: str,
    expected_manifest_identity: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = require_directory(root, artifact_kind)
    manifest_path = require_file(root / "manifest.json", f"{artifact_kind} manifest")
    completion_path = require_file(
        root / "completion.json", f"{artifact_kind} completion"
    )
    manifest = load_json(manifest_path, f"{artifact_kind} manifest")
    manifest_identity = require_hex64(
        manifest.get("identity_sha256"), f"{artifact_kind} manifest identity"
    )
    if expected_manifest_identity is not None and manifest_identity != require_hex64(
        expected_manifest_identity, f"expected {artifact_kind} manifest identity"
    ):
        raise RuntimeError(f"wrong bound {artifact_kind} manifest identity")
    if canonical_sha256(without_identity(manifest)) != manifest_identity:
        raise RuntimeError(f"{artifact_kind} manifest canonical identity failed")
    by_name = _safe_manifest_map(manifest, artifact_kind)
    _preflight_manifest_members(root, by_name, artifact_kind)
    record_path = require_file(root / record_name, f"{artifact_kind} primary record")
    record = load_json(record_path, f"{artifact_kind} primary record")
    completion = load_json(completion_path, f"{artifact_kind} completion")
    record_identity = require_hex64(
        record.get("identity_sha256"), f"{artifact_kind} record identity"
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
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("artifact_kind") != artifact_kind
        or manifest.get("primary_record_name") != record_name
        or manifest.get("primary_record_identity_sha256") != record_identity
        or manifest.get("files") != _artifact_records(root)
        or completion != expected_completion
    ):
        raise RuntimeError(f"{artifact_kind} record lock validation failed")
    return record, manifest


def validate_foundations() -> tuple[dict[str, Any], dict[str, Any]]:
    foundations = evaluation.validate_foundation_locks()
    evaluator_contract, evaluator_manifest = evaluation.validate_source_lock(
        EVALUATOR_SOURCE_LOCK
    )
    if (
        evaluator_contract.get("identity_sha256")
        != EXPECTED_EVALUATOR_CONTRACT_IDENTITY
        or evaluator_manifest.get("identity_sha256")
        != EXPECTED_EVALUATOR_MANIFEST_IDENTITY
        or evaluator_contract.get("implementation_source_sha256")
        != EXPECTED_EVALUATOR_SOURCE_SHA256
    ):
        raise RuntimeError("wrong pinned evaluator-v5 identity")
    phenotype = foundations["protocol"].get("phenotype_contract")
    expected_components = phenotype.get("review_components", {}) if isinstance(phenotype, dict) else {}
    if (
        phenotype != EXPECTED_PHENOTYPE_CONTRACT
        or not isinstance(phenotype, dict)
        or tuple(expected_components.get("blur_components", ()))
        != tuple(COMPONENT_FLAGS[:3])
        or tuple(expected_components.get("discrete_structure_components", ()))
        != tuple(COMPONENT_FLAGS[3:7])
        or tuple(expected_components.get("additional_components", ()))
        != (*COMPONENT_FLAGS[7:], NONE_SENTINEL)
        or phenotype.get("adjudication", {}).get("promotion_allowed") is not False
    ):
        raise RuntimeError("phase-1 phenotype contract differs from implemented rubric")
    return foundations, {
        "contract": evaluator_contract,
        "manifest": evaluator_manifest,
    }


def scientific_contract() -> dict[str, Any]:
    phenotype = EXPECTED_PHENOTYPE_CONTRACT
    return {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_THIRD_POOL_ENDPOINT_OR_REVIEW_ACCESS",
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        "foundation_identity_pins": {
            "phase1_protocol_identity_sha256": evaluation.EXPECTED_PHASE1_PROTOCOL_IDENTITY,
            "phase1_protocol_manifest_identity_sha256": evaluation.EXPECTED_PHASE1_PROTOCOL_MANIFEST_IDENTITY,
            "phase1_threshold_identity_sha256": evaluation.EXPECTED_PHASE1_THRESHOLD_IDENTITY,
            "phase1_threshold_manifest_identity_sha256": evaluation.EXPECTED_PHASE1_THRESHOLD_MANIFEST_IDENTITY,
            "sampling_protocol_identity_sha256": evaluation.EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
            "sampling_manifest_identity_sha256": evaluation.EXPECTED_SAMPLING_MANIFEST_IDENTITY,
            "evaluator_contract_identity_sha256": EXPECTED_EVALUATOR_CONTRACT_IDENTITY,
            "evaluator_manifest_identity_sha256": EXPECTED_EVALUATOR_MANIFEST_IDENTITY,
            "evaluator_source_sha256": EXPECTED_EVALUATOR_SOURCE_SHA256,
        },
        "cohort": {
            "classes_ordered": list(CLASSES),
            "global_seeds": list(SEEDS),
            "trajectory_count": TRAJECTORY_COUNT,
        },
        "phenotype_contract": phenotype,
        "phenotype_contract_identity_sha256": canonical_sha256(phenotype),
        "review_schema": {
            "severity_allowed": [0, 1, 2, 3],
            "severity_rubric": SEVERITY_RUBRIC,
            "component_flags_exact": list(COMPONENT_FLAGS),
            "none_sentinel": NONE_SENTINEL,
            "none_rule": "none=1 iff all nine component flags are 0",
            "columns_exact": list(REVIEW_COLUMNS),
            "three_independent_reviewers": list(REVIEWERS),
        },
        "majority_rules": {
            "clean_good": "at least two severity votes equal 0",
            "raw_clear_bad": "at least two severity votes are 2 or 3",
            "mild_or_disputed": "neither severity majority",
            "broad_component": (
                "for each reviewer take any member of the broad group, then require "
                "at least two of three reviewer-level presences"
            ),
        },
        "adjudication": {
            "scope": "raw majority clear-bad only",
            "allowed_actions": list(ADJUDICATION_ACTIONS),
            "promotion_allowed": False,
            "non_raw-clear-bad rows_exposed_to_adjudicator": False,
            "case_columns_exact": list(ADJUDICATION_CASE_COLUMNS),
            "individual_reviewer_votes_or_component_counts_exposed": False,
        },
        "review_pack": {
            "native_endpoint_size": [TILE_SIZE, TILE_SIZE],
            "native_endpoint_copy": "byte-identical and individually viewable",
            "reviewer_specific_order": (
                "SHA-256 sort by frozen domain, sampling-pool identity, reviewer, "
                "class, and sample index"
            ),
            "grid": {
                "one_class_per_grid": True,
                "images_per_grid": GRID_BLOCK_SIZE,
                "columns": GRID_COLUMNS,
                "rows": GRID_ROWS,
                "tile_pixels_unresampled": [TILE_SIZE, TILE_SIZE],
                "grids_per_class_per_reviewer": len(SEEDS) // GRID_BLOCK_SIZE,
            },
        },
        "runtime_input_access": {
            "phase0_source_freeze_only": (
                "verifies already-frozen phase1 protocol/threshold, sampling-source, "
                "and evaluator locks before any third-pool evidence exists"
            ),
            "post_freeze_source_validation_reopens_foundation_locks": False,
            "review_pack_builder_allowed": [
                "pool_manifest.json",
                "pool_completion.json",
                "execution_plan.json",
                "600 per-seed manifest.json files",
                "600 per-seed completion.json files",
                "1800 terminal class PNG files",
            ],
            "always_forbidden": [
                "trace.npz bytes",
                "feature products or feature CSV/NPZ",
                "candidate scores, thresholds, ranks, or alerts",
                "old or screening labels/reviews",
                "non-terminal sample.png",
            ],
        },
        "final_consensus": {
            "experiment": evaluation.CONSENSUS_EXPERIMENT,
            "row_member": evaluation.CONSENSUS_ROWS_NAME,
            "row_columns_exact": list(CONSENSUS_COLUMNS),
            "aggregate_member": evaluation.CONSENSUS_AGGREGATE_NAME,
            "file_members_exact": [
                evaluation.CONSENSUS_AGGREGATE_NAME,
                evaluation.CONSENSUS_ROWS_NAME,
            ],
            "evaluator_v5_stage_A_compatible": True,
        },
    }


SOURCE_ARTIFACT_KIND = "dit_bad_good_third_pool_blind_review_source_lock_v1"
SOURCE_RECORD_NAME = "blind_review_contract.json"


def freeze_source_lock(output: Path) -> Path:
    validate_foundations()
    output = require_output_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        contract = scientific_contract()
        contract["identity_sha256"] = canonical_sha256(contract)
        shutil.copy2(Path(__file__).resolve(), staging / "pipeline_source.py")
        return finalize_record_staging(
            staging,
            output,
            artifact_kind=SOURCE_ARTIFACT_KIND,
            record_name=SOURCE_RECORD_NAME,
            record=contract,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_source_lock(
    root: Path, *, require_live_source: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    _preflight_exact_names(
        root,
        {
            SOURCE_RECORD_NAME,
            "pipeline_source.py",
            "manifest.json",
            "completion.json",
        },
        "blind-review source lock",
    )
    contract, manifest = validate_record_lock(
        root,
        artifact_kind=SOURCE_ARTIFACT_KIND,
        record_name=SOURCE_RECORD_NAME,
    )
    expected = scientific_contract()
    expected["identity_sha256"] = canonical_sha256(expected)
    if contract != expected:
        raise RuntimeError("blind-review source contract differs from live frozen source")
    by_name = _safe_manifest_map(manifest, "blind-review source lock")
    if set(by_name) != {SOURCE_RECORD_NAME, "pipeline_source.py"}:
        raise RuntimeError("blind-review source lock member set changed")
    frozen_source = require_file(root / "pipeline_source.py", "frozen pipeline source")
    if sha256_file(frozen_source) != contract["implementation_source_sha256"]:
        raise RuntimeError("frozen pipeline source hash differs from contract")
    if require_live_source and sha256_file(Path(__file__).resolve()) != sha256_file(
        frozen_source
    ):
        raise RuntimeError("live blind-review source differs from frozen source")
    return contract, manifest


def _inspect_png(path: Path) -> tuple[int, str, str]:
    path = require_file(path, "terminal endpoint PNG")
    byte_count = path.stat().st_size
    file_hash = sha256_file(path)
    with Image.open(path) as image:
        if (
            image.format != "PNG"
            or image.mode != "RGB"
            or image.size != (TILE_SIZE, TILE_SIZE)
            or getattr(image, "n_frames", 1) != 1
        ):
            raise RuntimeError(f"endpoint is not one native RGB 256x256 PNG: {path}")
        image.load()
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return byte_count, file_hash, pixel_hash


def _seed_output_map(manifest: Mapping[str, Any], seed: int) -> dict[str, dict[str, Any]]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not all(isinstance(row, dict) for row in outputs):
        raise RuntimeError(f"seed {seed} output records are malformed")
    by_name = {str(row.get("relative_path")): dict(row) for row in outputs}
    if (
        len(by_name) != len(outputs)
        or set(by_name) != EXPECTED_SEED_OUTPUTS
        or canonical_sha256(outputs) != manifest.get("outputs_sha256")
    ):
        raise RuntimeError(f"seed {seed} declared payload set changed")
    return by_name


def load_pool_endpoints(
    root: Path, expected_pool_identity: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open only pool receipts/plan, seed manifests/completions, and terminal PNGs."""

    root = require_directory(root, "completed third-pool root")
    pool_manifest_path = require_file(root / "pool_manifest.json", "pool manifest")
    pool_completion_path = require_file(root / "pool_completion.json", "pool completion")
    execution_plan_path = require_file(root / "execution_plan.json", "pool execution plan")
    pool_manifest = load_json(pool_manifest_path, "pool manifest")
    pool_completion = load_json(pool_completion_path, "pool completion")
    execution_plan = load_json(execution_plan_path, "pool execution plan")
    pool_identity = require_hex64(pool_manifest.get("identity_sha256"), "pool identity")
    expected_pool_identity = require_hex64(expected_pool_identity, "bound pool identity")
    expected_pool_manifest_keys = {
        "schema_version",
        "status",
        "sampling_protocol_identity_sha256",
        "phase1_protocol_identity_sha256",
        "phase1_threshold_identity_sha256",
        "execution_plan_sha256",
        "seed_count",
        "trajectory_count",
        "seed_outputs",
        "runner_logs",
        "observation_only",
        "labels_reviews_screen_results_or_sample_scores_read",
        "score_label_join_performed",
        "identity_sha256",
    }
    expected_plan_keys = {
        "schema_version",
        "status",
        "sampling_source_lock",
        "sampling_protocol_identity_sha256",
        "sampling_manifest_identity_sha256",
        "phase1_protocol_identity_sha256",
        "phase1_threshold_identity_sha256",
        "trace_source",
        "trace_source_sha256",
        "launcher_source_sha256",
        "classes_ordered",
        "global_seeds",
        "global_seed_count",
        "trajectory_count",
        "gpus_ordered",
        "assignment",
        "assignment_kind",
        "output_root",
        "dit_root",
        "checkpoint",
        "vae_snapshot",
        "required_trace_arrays",
        "observation_only",
        "labels_reviews_screen_results_or_sample_scores_read",
        "score_label_join_performed",
    }
    expected_pool_completion = {
        "complete": True,
        "pool_identity_sha256": pool_identity,
        "pool_manifest_sha256": sha256_file(pool_manifest_path),
        "execution_plan_sha256": sha256_file(execution_plan_path),
        "seed_count": len(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
    }
    if (
        set(pool_manifest) != expected_pool_manifest_keys
        or set(execution_plan) != expected_plan_keys
        or pool_identity != expected_pool_identity
        or canonical_sha256(without_identity(pool_manifest)) != pool_identity
        or pool_manifest.get("status") != "complete"
        or pool_manifest.get("sampling_protocol_identity_sha256")
        != evaluation.EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or pool_manifest.get("phase1_protocol_identity_sha256")
        != evaluation.EXPECTED_PHASE1_PROTOCOL_IDENTITY
        or pool_manifest.get("phase1_threshold_identity_sha256")
        != evaluation.EXPECTED_PHASE1_THRESHOLD_IDENTITY
        or pool_manifest.get("seed_count") != len(SEEDS)
        or pool_manifest.get("trajectory_count") != TRAJECTORY_COUNT
        or pool_manifest.get("observation_only") is not True
        or pool_manifest.get("labels_reviews_screen_results_or_sample_scores_read")
        is not False
        or pool_manifest.get("score_label_join_performed") is not False
        or pool_completion != expected_pool_completion
        or pool_manifest.get("execution_plan_sha256")
        != expected_pool_completion["execution_plan_sha256"]
        or execution_plan.get("status")
        != "FROZEN_FOUR_GPU_CONTIGUOUS_EXECUTION_PLAN"
        or execution_plan.get("sampling_protocol_identity_sha256")
        != evaluation.EXPECTED_SAMPLING_PROTOCOL_IDENTITY
        or execution_plan.get("sampling_manifest_identity_sha256")
        != evaluation.EXPECTED_SAMPLING_MANIFEST_IDENTITY
        or execution_plan.get("phase1_protocol_identity_sha256")
        != evaluation.EXPECTED_PHASE1_PROTOCOL_IDENTITY
        or execution_plan.get("phase1_threshold_identity_sha256")
        != evaluation.EXPECTED_PHASE1_THRESHOLD_IDENTITY
        or tuple(execution_plan.get("classes_ordered", ())) != CLASSES
        or tuple(execution_plan.get("global_seeds", ())) != SEEDS
        or execution_plan.get("global_seed_count") != len(SEEDS)
        or execution_plan.get("trajectory_count") != TRAJECTORY_COUNT
        or execution_plan.get("observation_only") is not True
        or execution_plan.get("labels_reviews_screen_results_or_sample_scores_read")
        is not False
        or execution_plan.get("score_label_join_performed") is not False
    ):
        raise RuntimeError("completed sampling-pool receipts failed")
    seed_outputs = pool_manifest.get("seed_outputs")
    if not isinstance(seed_outputs, list) or len(seed_outputs) != len(SEEDS):
        raise RuntimeError("pool seed receipt count changed")
    endpoints: list[dict[str, Any]] = []
    seed_lineage: list[dict[str, Any]] = []
    expected_seed_keys = {
        "seed",
        "relative_output",
        "identity_sha256",
        "manifest_sha256",
        "completion_sha256",
        "outputs_sha256",
        "output_count",
        "trace_npz_sha256",
    }
    for expected_seed, seed_receipt in zip(SEEDS, seed_outputs, strict=True):
        if (
            not isinstance(seed_receipt, dict)
            or set(seed_receipt) != expected_seed_keys
            or seed_receipt.get("seed") != expected_seed
            or seed_receipt.get("relative_output")
            != f"third_pool_v1_seed{expected_seed:03d}"
            or seed_receipt.get("output_count") != len(EXPECTED_SEED_OUTPUTS)
        ):
            raise RuntimeError(f"pool seed receipt changed at seed {expected_seed}")
        for name in (
            "identity_sha256",
            "manifest_sha256",
            "completion_sha256",
            "outputs_sha256",
            "trace_npz_sha256",
        ):
            require_hex64(seed_receipt.get(name), f"seed {expected_seed} {name}")
        seed_root = require_directory(
            root / seed_receipt["relative_output"], f"seed {expected_seed} directory"
        )
        require_directory(seed_root / "images", f"seed {expected_seed} image directory")
        seed_manifest_path = require_file(
            seed_root / "manifest.json", f"seed {expected_seed} manifest"
        )
        seed_completion_path = require_file(
            seed_root / "completion.json", f"seed {expected_seed} completion"
        )
        if sha256_file(seed_manifest_path) != seed_receipt["manifest_sha256"]:
            raise RuntimeError(f"seed {expected_seed} manifest hash differs from pool")
        seed_manifest = load_json(seed_manifest_path, f"seed {expected_seed} manifest")
        seed_completion = load_json(
            seed_completion_path, f"seed {expected_seed} completion"
        )
        identity = seed_manifest.get("identity")
        if not isinstance(identity, dict):
            raise RuntimeError(f"seed {expected_seed} lacks scientific identity")
        seed_identity = canonical_sha256(identity)
        protocol = identity.get("protocol", {})
        if (
            seed_manifest.get("status") != "complete"
            or seed_manifest.get("identity_sha256") != seed_identity
            or seed_identity != seed_receipt["identity_sha256"]
            or seed_manifest.get("outputs_sha256") != seed_receipt["outputs_sha256"]
            or identity.get("runner") != "trace_dit_imagenet256_custom_batch"
            or identity.get("observation_only") is not True
            or identity.get("quality_score") is not None
            or identity.get("selection") is not None
            or identity.get("intervention") is not None
            or tuple(protocol.get("class_ids_ordered", ())) != CLASSES
            or protocol.get("global_torch_seed") != expected_seed
            or protocol.get("sampling_steps") != 250
            or protocol.get("image_size") != TILE_SIZE
        ):
            raise RuntimeError(f"seed {expected_seed} scientific identity changed")
        outputs = _seed_output_map(seed_manifest, expected_seed)
        if seed_completion != {
            "schema": 1,
            "identity_sha256": seed_identity,
            "manifest_sha256": seed_receipt["manifest_sha256"],
            "outputs_sha256": seed_receipt["outputs_sha256"],
            "output_count": len(EXPECTED_SEED_OUTPUTS),
        } or sha256_file(seed_completion_path) != seed_receipt["completion_sha256"]:
            raise RuntimeError(f"seed {expected_seed} completion differs from pool")
        trace_record = outputs["trace.npz"]
        if trace_record.get("sha256") != seed_receipt["trace_npz_sha256"]:
            raise RuntimeError(f"seed {expected_seed} declared trace hash differs from pool")
        seed_lineage.append(
            {
                "seed": expected_seed,
                "relative_output": seed_receipt["relative_output"],
                "identity_sha256": seed_identity,
                "manifest_sha256": seed_receipt["manifest_sha256"],
                "completion_sha256": seed_receipt["completion_sha256"],
                "outputs_sha256": seed_receipt["outputs_sha256"],
            }
        )
        for slot, class_id in enumerate(CLASSES):
            relative = ENDPOINT_RELATIVE[class_id]
            output_record = outputs[relative]
            png_path = require_file(
                seed_root / relative,
                f"seed {expected_seed} class {class_id} terminal PNG",
            )
            byte_count, file_hash, pixel_hash = _inspect_png(png_path)
            if (
                output_record.get("bytes") != byte_count
                or output_record.get("sha256") != file_hash
                or output_record.get("pixel_sha256") != pixel_hash
                or output_record.get("mode") != "RGB"
                or output_record.get("size") != [TILE_SIZE, TILE_SIZE]
            ):
                raise RuntimeError(
                    f"seed {expected_seed} class {class_id} terminal PNG differs from manifest"
                )
            sample_index = (expected_seed - SEEDS[0]) * len(CLASSES) + slot
            endpoints.append(
                {
                    "sample_index": sample_index,
                    "global_seed": expected_seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "source_seed_relative_path": seed_receipt["relative_output"],
                    "source_endpoint_relative_path": relative,
                    "source_path": str(png_path),
                    "source_seed_identity_sha256": seed_identity,
                    "source_seed_manifest_sha256": seed_receipt["manifest_sha256"],
                    "source_png_bytes": byte_count,
                    "source_png_sha256": file_hash,
                    "source_pixel_sha256": pixel_hash,
                }
            )
    if [row["sample_index"] for row in endpoints] != list(range(TRAJECTORY_COUNT)):
        raise RuntimeError("terminal endpoints are not the exact ordered Cartesian cohort")
    return endpoints, {
        "path": str(root),
        "manifest_identity_sha256": pool_identity,
        "manifest_file_sha256": sha256_file(pool_manifest_path),
        "completion_file_sha256": sha256_file(pool_completion_path),
        "execution_plan_file_sha256": sha256_file(execution_plan_path),
        "seed_lineage_sha256": canonical_sha256(seed_lineage),
        "seed_count": len(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
        "access_audit": {
            "pool_manifest_opened": True,
            "pool_completion_opened": True,
            "execution_plan_opened": True,
            "per_seed_manifests_opened": len(SEEDS),
            "per_seed_completion_files_opened": len(SEEDS),
            "terminal_pngs_opened": TRAJECTORY_COUNT,
            "trace_npz_opened_or_statted": False,
            "sample_png_opened_or_statted": False,
            "feature_score_threshold_alert_or_label_file_opened": False,
        },
    }


PACK_ARTIFACT_KIND = "dit_bad_good_third_pool_endpoint_blind_review_pack_v1"
PACK_RECORD_NAME = "review_pack.json"


def native_relative(sample_index: int) -> str:
    return f"native/endpoint_{sample_index:04d}.png"


def reviewer_order_relative(reviewer: str) -> str:
    return f"{reviewer}/review_order.csv"


def reviewer_template_relative(reviewer: str) -> str:
    return f"{reviewer}/review_template.csv"


def reviewer_grid_relative(reviewer: str, class_id: int, grid_index: int) -> str:
    return f"{reviewer}/grids/class{class_id:04d}_grid{grid_index:02d}.png"


def review_id(pool_identity: str, reviewer: str, sample_index: int) -> str:
    payload = (
        f"dit-third-pool-review-id-v1|{pool_identity}|{reviewer}|{sample_index}"
    ).encode("utf-8")
    return f"{reviewer[-1]}_{hashlib.sha256(payload).hexdigest()[:20]}"


def review_sort_key(
    pool_identity: str, reviewer: str, class_id: int, sample_index: int
) -> str:
    payload = (
        "dit-third-pool-review-order-v1|"
        f"{pool_identity}|{reviewer}|{class_id}|{sample_index}"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def expected_reviewer_order(
    catalog: Sequence[Mapping[str, Any]], pool_identity: str, reviewer: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order_index = 0
    for class_id in CLASSES:
        class_rows = sorted(
            (row for row in catalog if int(row["class_id"]) == class_id),
            key=lambda row: (
                review_sort_key(
                    pool_identity, reviewer, class_id, int(row["sample_index"])
                ),
                int(row["sample_index"]),
            ),
        )
        if len(class_rows) != len(SEEDS):
            raise RuntimeError(f"wrong endpoint count for class {class_id}")
        for class_offset, row in enumerate(class_rows):
            grid_index = class_offset // GRID_BLOCK_SIZE
            grid_cell = class_offset % GRID_BLOCK_SIZE
            sample_index = int(row["sample_index"])
            rows.append(
                {
                    "reviewer_id": reviewer,
                    "review_order_index": order_index,
                    "review_id": review_id(pool_identity, reviewer, sample_index),
                    "class_id": class_id,
                    "native_image_relative_path": native_relative(sample_index),
                    "grid_relative_path": reviewer_grid_relative(
                        reviewer, class_id, grid_index
                    ),
                    "grid_cell_index": grid_cell,
                }
            )
            order_index += 1
    return rows


def _make_grid(
    root: Path, rows: Sequence[Mapping[str, Any]], destination: Path
) -> None:
    if len(rows) != GRID_BLOCK_SIZE:
        raise RuntimeError("each frozen review grid must contain exactly 20 images")
    cell_height = TILE_SIZE + GRID_LABEL_HEIGHT
    canvas = Image.new(
        "RGB", (GRID_COLUMNS * TILE_SIZE, GRID_ROWS * cell_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for cell, row in enumerate(rows):
        image_path = require_file(
            root / str(row["native_image_relative_path"]), "copied native endpoint"
        )
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError("native endpoint changed before grid assembly")
            image.load()
            tile = image.copy()
        column = cell % GRID_COLUMNS
        grid_row = cell // GRID_COLUMNS
        x = column * TILE_SIZE
        y = grid_row * cell_height
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y + TILE_SIZE, x + TILE_SIZE, y + cell_height), fill="white")
        draw.text(
            (x + 4, y + TILE_SIZE + 5),
            str(row["review_id"]),
            fill="black",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", pnginfo=PngInfo(), compress_level=9)


def build_review_pack(
    *,
    source_lock: Path,
    sampling_pool: Path,
    sampling_pool_manifest_identity: str,
    output: Path,
) -> Path:
    source_contract, source_manifest = validate_source_lock(source_lock)
    endpoints, pool_lineage = load_pool_endpoints(
        sampling_pool, sampling_pool_manifest_identity
    )
    output = require_output_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        catalog_rows: list[dict[str, Any]] = []
        for endpoint in endpoints:
            sample_index = int(endpoint["sample_index"])
            destination = staging / native_relative(sample_index)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(endpoint["source_path"], destination)
            if (
                destination.stat().st_size != endpoint["source_png_bytes"]
                or sha256_file(destination) != endpoint["source_png_sha256"]
            ):
                raise RuntimeError("native endpoint copy is not byte-identical")
            catalog_rows.append(
                {
                    "sample_index": sample_index,
                    "global_seed": endpoint["global_seed"],
                    "class_slot": endpoint["class_slot"],
                    "class_id": endpoint["class_id"],
                    "native_image_relative_path": native_relative(sample_index),
                    "source_seed_relative_path": endpoint[
                        "source_seed_relative_path"
                    ],
                    "source_endpoint_relative_path": endpoint[
                        "source_endpoint_relative_path"
                    ],
                    "source_seed_identity_sha256": endpoint[
                        "source_seed_identity_sha256"
                    ],
                    "source_seed_manifest_sha256": endpoint[
                        "source_seed_manifest_sha256"
                    ],
                    "source_png_bytes": endpoint["source_png_bytes"],
                    "source_png_sha256": endpoint["source_png_sha256"],
                    "source_pixel_sha256": endpoint["source_pixel_sha256"],
                }
            )
        catalog_path = staging / "sample_catalog.csv"
        write_csv(catalog_path, SAMPLE_CATALOG_COLUMNS, catalog_rows)
        (staging / "reviewer_instructions.md").write_text(
            REVIEWER_INSTRUCTIONS, encoding="utf-8"
        )

        order_hashes: dict[str, str] = {}
        template_hashes: dict[str, str] = {}
        class_orders: dict[tuple[str, int], tuple[int, ...]] = {}
        for reviewer in REVIEWERS:
            order_rows = expected_reviewer_order(
                catalog_rows, pool_lineage["manifest_identity_sha256"], reviewer
            )
            order_path = staging / reviewer_order_relative(reviewer)
            order_path.parent.mkdir(parents=True, exist_ok=True)
            write_csv(order_path, REVIEW_ORDER_COLUMNS, order_rows)
            template_path = staging / reviewer_template_relative(reviewer)
            write_csv(
                template_path,
                REVIEW_COLUMNS,
                [
                    {
                        "review_id": row["review_id"],
                        "severity": "",
                        **{flag: "" for flag in COMPONENT_FLAGS},
                        NONE_SENTINEL: "",
                    }
                    for row in order_rows
                ],
            )
            order_hashes[reviewer] = sha256_file(order_path)
            template_hashes[reviewer] = sha256_file(template_path)
            for class_id in CLASSES:
                class_rows = [row for row in order_rows if row["class_id"] == class_id]
                class_orders[(reviewer, class_id)] = tuple(
                    int(Path(str(row["native_image_relative_path"])).stem.split("_")[-1])
                    for row in class_rows
                )
                for offset in range(0, len(class_rows), GRID_BLOCK_SIZE):
                    block = class_rows[offset : offset + GRID_BLOCK_SIZE]
                    _make_grid(staging, block, staging / block[0]["grid_relative_path"])
        for class_id in CLASSES:
            if len({class_orders[(reviewer, class_id)] for reviewer in REVIEWERS}) != len(
                REVIEWERS
            ):
                raise RuntimeError("reviewer-specific class grid orders are not distinct")

        shutil.copy2(Path(__file__).resolve(), staging / "pipeline_source.py")
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE_ENDPOINT_ONLY_BLIND_REVIEW_PACK",
            "blind_review_source_contract_identity_sha256": source_contract[
                "identity_sha256"
            ],
            "blind_review_source_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "sampling_pool": pool_lineage,
            "cohort": {
                "classes_ordered": list(CLASSES),
                "global_seeds": list(SEEDS),
                "trajectory_count": TRAJECTORY_COUNT,
            },
            "sample_catalog": {
                "member": "sample_catalog.csv",
                "columns_exact": list(SAMPLE_CATALOG_COLUMNS),
                "rows": TRAJECTORY_COUNT,
                "file_sha256": sha256_file(catalog_path),
            },
            "reviewer_instructions": {
                "member": "reviewer_instructions.md",
                "file_sha256": sha256_file(staging / "reviewer_instructions.md"),
            },
            "reviewers": {
                reviewer: {
                    "review_order_member": reviewer_order_relative(reviewer),
                    "review_order_file_sha256": order_hashes[reviewer],
                    "review_template_member": reviewer_template_relative(reviewer),
                    "review_template_file_sha256": template_hashes[reviewer],
                    "endpoint_count": TRAJECTORY_COUNT,
                    "grid_count": len(CLASSES) * len(SEEDS) // GRID_BLOCK_SIZE,
                    "different_frozen_order_from_other_reviewers": True,
                }
                for reviewer in REVIEWERS
            },
            "image_contract": {
                "native_endpoint_count": TRAJECTORY_COUNT,
                "native_endpoint_size": [TILE_SIZE, TILE_SIZE],
                "native_copy_byte_identical": True,
                "each_native_image_individually_viewable": True,
                "one_class_per_grid": True,
                "images_per_grid": GRID_BLOCK_SIZE,
                "tile_resampling": "none",
                "grid_count_total": len(REVIEWERS)
                * len(CLASSES)
                * len(SEEDS)
                // GRID_BLOCK_SIZE,
            },
            "review_schema": source_contract["review_schema"],
            "access_audit": {
                **pool_lineage["access_audit"],
                "only_terminal_endpoint_pngs_copied_or_decoded": True,
                "trace_feature_score_threshold_alert_old_label_or_screen_opened": False,
            },
            "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        record["identity_sha256"] = canonical_sha256(record)
        return finalize_record_staging(
            staging,
            output,
            artifact_kind=PACK_ARTIFACT_KIND,
            record_name=PACK_RECORD_NAME,
            record=record,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_review_pack(
    root: Path,
    *,
    expected_manifest_identity: str,
    source_lock: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    source_contract, source_manifest = validate_source_lock(source_lock)
    expected_files = {
        PACK_RECORD_NAME,
        "pipeline_source.py",
        "sample_catalog.csv",
        "reviewer_instructions.md",
        "manifest.json",
        "completion.json",
        *(native_relative(index) for index in range(TRAJECTORY_COUNT)),
        *(
            reviewer_order_relative(reviewer)
            for reviewer in REVIEWERS
        ),
        *(
            reviewer_template_relative(reviewer)
            for reviewer in REVIEWERS
        ),
        *(
            reviewer_grid_relative(reviewer, class_id, grid_index)
            for reviewer in REVIEWERS
            for class_id in CLASSES
            for grid_index in range(len(SEEDS) // GRID_BLOCK_SIZE)
        ),
    }
    _preflight_exact_names(root, expected_files, "endpoint blind-review pack")
    record, manifest = validate_record_lock(
        root,
        artifact_kind=PACK_ARTIFACT_KIND,
        record_name=PACK_RECORD_NAME,
        expected_manifest_identity=expected_manifest_identity,
    )
    pool = record.get("sampling_pool", {})
    if (
        record.get("status") != "COMPLETE_ENDPOINT_ONLY_BLIND_REVIEW_PACK"
        or record.get("blind_review_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
        or record.get("blind_review_source_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or record.get("implementation_source_sha256")
        != source_contract["implementation_source_sha256"]
        or tuple(record.get("cohort", {}).get("classes_ordered", ())) != CLASSES
        or tuple(record.get("cohort", {}).get("global_seeds", ())) != SEEDS
        or record.get("cohort", {}).get("trajectory_count") != TRAJECTORY_COUNT
        or record.get("review_schema") != source_contract["review_schema"]
        or pool.get("trajectory_count") != TRAJECTORY_COUNT
    ):
        raise RuntimeError("review-pack scientific contract changed")
    require_hex64(pool.get("manifest_identity_sha256"), "review-pack pool identity")
    catalog_path = root / "sample_catalog.csv"
    catalog_raw = read_csv(catalog_path, SAMPLE_CATALOG_COLUMNS, "sample catalog")
    if (
        len(catalog_raw) != TRAJECTORY_COUNT
        or sha256_file(catalog_path)
        != record.get("sample_catalog", {}).get("file_sha256")
    ):
        raise RuntimeError("review-pack sample catalog changed")
    instructions_path = require_file(
        root / "reviewer_instructions.md", "frozen reviewer instructions"
    )
    if (
        instructions_path.read_text(encoding="utf-8") != REVIEWER_INSTRUCTIONS
        or sha256_file(instructions_path)
        != record.get("reviewer_instructions", {}).get("file_sha256")
    ):
        raise RuntimeError("reviewer instructions changed")
    catalog: list[dict[str, Any]] = []
    for expected_index, raw in enumerate(catalog_raw):
        try:
            sample_index = int(raw["sample_index"])
            seed = int(raw["global_seed"])
            slot = int(raw["class_slot"])
            class_id = int(raw["class_id"])
            source_bytes = int(raw["source_png_bytes"])
        except ValueError as exc:
            raise RuntimeError("invalid sample-catalog identifier") from exc
        image_path = require_file(
            root / raw["native_image_relative_path"], "review-pack native endpoint"
        )
        copied_bytes, copied_hash, copied_pixel_hash = _inspect_png(image_path)
        if (
            sample_index != expected_index
            or seed != SEEDS[0] + sample_index // len(CLASSES)
            or slot != sample_index % len(CLASSES)
            or class_id != CLASSES[slot]
            or raw["native_image_relative_path"] != native_relative(sample_index)
            or source_bytes != copied_bytes
            or copied_hash != raw["source_png_sha256"]
            or copied_pixel_hash != raw["source_pixel_sha256"]
        ):
            raise RuntimeError(f"sample-catalog row failed: {expected_index}")
        require_hex64(raw["source_seed_identity_sha256"], "seed identity")
        require_hex64(raw["source_seed_manifest_sha256"], "seed manifest hash")
        require_hex64(raw["source_png_sha256"], "source PNG hash")
        require_hex64(raw["source_pixel_sha256"], "source pixel hash")
        catalog.append(
            {
                **raw,
                "sample_index": sample_index,
                "global_seed": seed,
                "class_slot": slot,
                "class_id": class_id,
                "source_png_bytes": source_bytes,
            }
        )
    orders: dict[str, list[dict[str, Any]]] = {}
    for reviewer in REVIEWERS:
        order_path = root / reviewer_order_relative(reviewer)
        template_path = root / reviewer_template_relative(reviewer)
        order_raw = read_csv(order_path, REVIEW_ORDER_COLUMNS, f"{reviewer} order")
        template = read_csv(template_path, REVIEW_COLUMNS, f"{reviewer} template")
        expected_order = expected_reviewer_order(catalog, pool["manifest_identity_sha256"], reviewer)
        expected_as_text = [
            {column: str(row[column]) for column in REVIEW_ORDER_COLUMNS}
            for row in expected_order
        ]
        if (
            order_raw != expected_as_text
            or len(template) != TRAJECTORY_COUNT
            or [row["review_id"] for row in template]
            != [row["review_id"] for row in order_raw]
            or any(any(row[column] for column in REVIEW_COLUMNS[1:]) for row in template)
            or sha256_file(order_path)
            != record.get("reviewers", {}).get(reviewer, {}).get(
                "review_order_file_sha256"
            )
            or sha256_file(template_path)
            != record.get("reviewers", {}).get(reviewer, {}).get(
                "review_template_file_sha256"
            )
        ):
            raise RuntimeError(f"{reviewer} frozen order/template changed")
        orders[reviewer] = expected_order
    for class_id in CLASSES:
        class_orders = {
            tuple(
                row["native_image_relative_path"]
                for row in orders[reviewer]
                if row["class_id"] == class_id
            )
            for reviewer in REVIEWERS
        }
        if len(class_orders) != len(REVIEWERS):
            raise RuntimeError("reviewer grid order collision")
    return record, manifest, catalog, orders


REVIEW_ARTIFACT_KIND = "dit_bad_good_third_pool_independent_review_lock_v1"
REVIEW_RECORD_NAME = "review_record.json"


def parse_completed_review(
    path: Path,
    *,
    reviewer: str,
    expected_order: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = read_csv(path, REVIEW_COLUMNS, f"completed {reviewer} review")
    if len(rows) != TRAJECTORY_COUNT:
        raise RuntimeError(f"{reviewer} review does not cover all trajectories")
    expected_ids = [str(row["review_id"]) for row in expected_order]
    if [row["review_id"] for row in rows] != expected_ids:
        raise RuntimeError(f"{reviewer} review IDs/order differ from its frozen template")
    parsed: list[dict[str, Any]] = []
    for row in rows:
        if row["severity"] not in {"0", "1", "2", "3"}:
            raise RuntimeError(f"invalid severity in {reviewer}: {row['review_id']}")
        try:
            severity = int(row["severity"])
        except ValueError as exc:
            raise RuntimeError(f"invalid severity in {reviewer}: {row['review_id']}") from exc
        flags: dict[str, bool] = {}
        for flag in (*COMPONENT_FLAGS, NONE_SENTINEL):
            value = row[flag]
            if value not in {"0", "1"}:
                raise RuntimeError(
                    f"{reviewer} flag must be literal 0/1: {row['review_id']}/{flag}"
                )
            flags[flag] = value == "1"
        any_component = any(flags[flag] for flag in COMPONENT_FLAGS)
        if severity not in range(4) or flags[NONE_SENTINEL] == any_component:
            raise RuntimeError(
                f"{reviewer} severity/none contract failed: {row['review_id']}"
            )
        parsed.append(
            {
                "review_id": row["review_id"],
                "severity": severity,
                **flags,
            }
        )
    return parsed


def lock_review(
    *,
    source_lock: Path,
    review_pack: Path,
    review_pack_manifest_identity: str,
    reviewer: str,
    completed_review_csv: Path,
    attest_blind: bool,
    output: Path,
) -> Path:
    if reviewer not in REVIEWERS:
        raise RuntimeError(f"reviewer must be one of {REVIEWERS}")
    if not attest_blind:
        raise RuntimeError("reviewer must explicitly attest the frozen blindness contract")
    source_contract, source_manifest = validate_source_lock(source_lock)
    pack_record, pack_manifest, _, orders = validate_review_pack(
        review_pack,
        expected_manifest_identity=review_pack_manifest_identity,
        source_lock=source_lock,
    )
    parsed = parse_completed_review(
        completed_review_csv,
        reviewer=reviewer,
        expected_order=orders[reviewer],
    )
    output = require_output_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        normalized_path = staging / "review_rows.csv"
        write_csv(
            normalized_path,
            REVIEW_COLUMNS,
            [
                {
                    "review_id": row["review_id"],
                    "severity": row["severity"],
                    **{
                        flag: int(bool(row[flag]))
                        for flag in (*COMPONENT_FLAGS, NONE_SENTINEL)
                    },
                }
                for row in parsed
            ],
        )
        shutil.copy2(Path(__file__).resolve(), staging / "pipeline_source.py")
        counts = {str(level): sum(row["severity"] == level for row in parsed) for level in range(4)}
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "LOCKED_INDEPENDENT_ENDPOINT_ONLY_REVIEW",
            "reviewer_id": reviewer,
            "blind_review_source_contract_identity_sha256": source_contract[
                "identity_sha256"
            ],
            "blind_review_source_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "review_pack": {
                "path": str(Path(review_pack).expanduser().absolute()),
                "record_identity_sha256": pack_record["identity_sha256"],
                "manifest_identity_sha256": pack_manifest["identity_sha256"],
                "sampling_pool_identity_sha256": pack_record["sampling_pool"][
                    "manifest_identity_sha256"
                ],
                "review_order_file_sha256": pack_record["reviewers"][reviewer][
                    "review_order_file_sha256"
                ],
                "review_template_file_sha256": pack_record["reviewers"][reviewer][
                    "review_template_file_sha256"
                ],
            },
            "review_rows": {
                "member": "review_rows.csv",
                "columns_exact": list(REVIEW_COLUMNS),
                "row_count": TRAJECTORY_COUNT,
                "file_sha256": sha256_file(normalized_path),
                "severity_counts": counts,
            },
            "reviewer_attestation": {
                "independent_review": True,
                "endpoint_images_only": True,
                "native_images_or_frozen_grids_only": True,
                "candidate_hypothesis_seen": False,
                "metric_score_threshold_rank_or_alert_seen": False,
                "trajectory_or_intermediate_state_seen": False,
                "feature_product_seen": False,
                "old_label_or_screen_result_seen": False,
                "other_reviewer_rows_or_summary_seen": False,
            },
            "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        record["identity_sha256"] = canonical_sha256(record)
        return finalize_record_staging(
            staging,
            output,
            artifact_kind=REVIEW_ARTIFACT_KIND,
            record_name=REVIEW_RECORD_NAME,
            record=record,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_review_lock(
    root: Path,
    *,
    reviewer: str,
    expected_manifest_identity: str,
    source_contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pack_record: Mapping[str, Any],
    pack_manifest: Mapping[str, Any],
    expected_order: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    _preflight_exact_names(
        root,
        {
            REVIEW_RECORD_NAME,
            "review_rows.csv",
            "pipeline_source.py",
            "manifest.json",
            "completion.json",
        },
        f"{reviewer} review lock",
    )
    record, manifest = validate_record_lock(
        root,
        artifact_kind=REVIEW_ARTIFACT_KIND,
        record_name=REVIEW_RECORD_NAME,
        expected_manifest_identity=expected_manifest_identity,
    )
    by_name = _safe_manifest_map(manifest, f"{reviewer} review lock")
    if set(by_name) != {REVIEW_RECORD_NAME, "review_rows.csv", "pipeline_source.py"}:
        raise RuntimeError(f"{reviewer} review-lock member set changed")
    expected_attestation = {
        "independent_review": True,
        "endpoint_images_only": True,
        "native_images_or_frozen_grids_only": True,
        "candidate_hypothesis_seen": False,
        "metric_score_threshold_rank_or_alert_seen": False,
        "trajectory_or_intermediate_state_seen": False,
        "feature_product_seen": False,
        "old_label_or_screen_result_seen": False,
        "other_reviewer_rows_or_summary_seen": False,
    }
    pack_ref = record.get("review_pack", {})
    if (
        record.get("status") != "LOCKED_INDEPENDENT_ENDPOINT_ONLY_REVIEW"
        or record.get("reviewer_id") != reviewer
        or record.get("blind_review_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
        or record.get("blind_review_source_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or record.get("implementation_source_sha256")
        != source_contract["implementation_source_sha256"]
        or record.get("reviewer_attestation") != expected_attestation
        or pack_ref.get("record_identity_sha256") != pack_record["identity_sha256"]
        or pack_ref.get("manifest_identity_sha256") != pack_manifest["identity_sha256"]
        or pack_ref.get("sampling_pool_identity_sha256")
        != pack_record["sampling_pool"]["manifest_identity_sha256"]
        or pack_ref.get("review_order_file_sha256")
        != pack_record["reviewers"][reviewer]["review_order_file_sha256"]
        or pack_ref.get("review_template_file_sha256")
        != pack_record["reviewers"][reviewer]["review_template_file_sha256"]
    ):
        raise RuntimeError(f"{reviewer} review-lock scientific lineage changed")
    rows_path = root / "review_rows.csv"
    parsed = parse_completed_review(
        rows_path,
        reviewer=reviewer,
        expected_order=expected_order,
    )
    counts = {str(level): sum(row["severity"] == level for row in parsed) for level in range(4)}
    if record.get("review_rows") != {
        "member": "review_rows.csv",
        "columns_exact": list(REVIEW_COLUMNS),
        "row_count": TRAJECTORY_COUNT,
        "file_sha256": sha256_file(rows_path),
        "severity_counts": counts,
    }:
        raise RuntimeError(f"{reviewer} review-row receipt changed")
    return record, manifest, parsed


def raw_majority_rows(
    catalog: Sequence[Mapping[str, Any]],
    orders: Mapping[str, Sequence[Mapping[str, Any]]],
    reviews: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[int, Mapping[str, Any]]] = {}
    for reviewer in REVIEWERS:
        review_by_id = {row["review_id"]: row for row in reviews[reviewer]}
        mapped: dict[int, Mapping[str, Any]] = {}
        for order_row in orders[reviewer]:
            sample_index = int(
                Path(str(order_row["native_image_relative_path"])).stem.split("_")[-1]
            )
            review_row = review_by_id.get(str(order_row["review_id"]))
            if sample_index in mapped or review_row is None:
                raise RuntimeError(f"{reviewer} review-to-pack join failed")
            mapped[sample_index] = review_row
        if set(mapped) != set(range(TRAJECTORY_COUNT)):
            raise RuntimeError(f"{reviewer} review-to-pack coverage failed")
        by_sample[reviewer] = mapped

    result: list[dict[str, Any]] = []
    for catalog_row in catalog:
        sample_index = int(catalog_row["sample_index"])
        rows = [by_sample[reviewer][sample_index] for reviewer in REVIEWERS]
        severities = [int(row["severity"]) for row in rows]
        clear_votes = sum(value in {2, 3} for value in severities)
        clean_votes = sum(value == 0 for value in severities)
        if clear_votes >= 2:
            raw_severity = "clear_bad"
        elif clean_votes >= 2:
            raw_severity = "clean_good"
        else:
            raw_severity = "mild_or_disputed"
        blur_votes = sum(
            any(bool(row[flag]) for flag in BLUR_COMPONENTS) for row in rows
        )
        structure_votes = sum(
            any(bool(row[flag]) for flag in DISCRETE_STRUCTURE_COMPONENTS)
            for row in rows
        )
        result.append(
            {
                "sample_index": sample_index,
                "global_seed": int(catalog_row["global_seed"]),
                "class_slot": int(catalog_row["class_slot"]),
                "class_id": int(catalog_row["class_id"]),
                "native_image_relative_path": catalog_row[
                    "native_image_relative_path"
                ],
                "source_png_sha256": catalog_row["source_png_sha256"],
                "severity_votes": severities,
                "raw_severity": raw_severity,
                "clear_bad_vote_count": clear_votes,
                "clean_good_vote_count": clean_votes,
                "blur_group_vote_count": blur_votes,
                "discrete_structure_group_vote_count": structure_votes,
                "blur_component_consensus": blur_votes >= 2,
                "discrete_structure_component_consensus": structure_votes >= 2,
            }
        )
    if len(result) != TRAJECTORY_COUNT:
        raise RuntimeError("raw majority lost or multiplied trajectories")
    return result


def load_review_chain(
    *,
    source_lock: Path,
    review_pack: Path,
    review_pack_manifest_identity: str,
    review_locks: Mapping[str, Path],
    review_lock_manifest_identities: Mapping[str, str],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    if set(review_locks) != set(REVIEWERS):
        raise RuntimeError("review chain must contain exactly the three frozen reviewers")
    if set(review_lock_manifest_identities) != set(REVIEWERS):
        raise RuntimeError("review chain must pin all three review-lock identities")
    source_contract, source_manifest = validate_source_lock(source_lock)
    pack_record, pack_manifest, catalog, orders = validate_review_pack(
        review_pack,
        expected_manifest_identity=review_pack_manifest_identity,
        source_lock=source_lock,
    )
    reviews: dict[str, list[dict[str, Any]]] = {}
    lineage: dict[str, dict[str, Any]] = {}
    seen_record_ids: set[str] = set()
    seen_manifest_ids: set[str] = set()
    for reviewer in REVIEWERS:
        review_record, review_manifest, rows = validate_review_lock(
            review_locks[reviewer],
            reviewer=reviewer,
            expected_manifest_identity=review_lock_manifest_identities[reviewer],
            source_contract=source_contract,
            source_manifest=source_manifest,
            pack_record=pack_record,
            pack_manifest=pack_manifest,
            expected_order=orders[reviewer],
        )
        if (
            review_record["identity_sha256"] in seen_record_ids
            or review_manifest["identity_sha256"] in seen_manifest_ids
        ):
            raise RuntimeError("three reviewer locks are not distinct")
        seen_record_ids.add(review_record["identity_sha256"])
        seen_manifest_ids.add(review_manifest["identity_sha256"])
        reviews[reviewer] = rows
        lineage[reviewer] = {
            "path": str(Path(review_locks[reviewer]).expanduser().absolute()),
            "record_identity_sha256": review_record["identity_sha256"],
            "manifest_identity_sha256": review_manifest["identity_sha256"],
            "review_rows_file_sha256": review_record["review_rows"]["file_sha256"],
        }
    raw = raw_majority_rows(catalog, orders, reviews)
    return pack_record, pack_manifest, catalog, orders, lineage, raw


ADJUDICATION_PACK_ARTIFACT_KIND = (
    "dit_bad_good_third_pool_raw_clear_bad_adjudication_pack_v1"
)
ADJUDICATION_PACK_RECORD_NAME = "adjudication_pack.json"


def adjudication_case_id(pack_manifest_identity: str, sample_index: int) -> str:
    payload = (
        "dit-third-pool-adjudication-case-v1|"
        f"{pack_manifest_identity}|{sample_index}"
    ).encode("utf-8")
    return f"a_{hashlib.sha256(payload).hexdigest()[:20]}"


def build_adjudication_pack(
    *,
    source_lock: Path,
    review_pack: Path,
    review_pack_manifest_identity: str,
    review_locks: Mapping[str, Path],
    review_lock_manifest_identities: Mapping[str, str],
    output: Path,
) -> Path:
    source_contract, source_manifest = validate_source_lock(source_lock)
    pack_record, pack_manifest, _, _, review_lineage, raw = load_review_chain(
        source_lock=source_lock,
        review_pack=review_pack,
        review_pack_manifest_identity=review_pack_manifest_identity,
        review_locks=review_locks,
        review_lock_manifest_identities=review_lock_manifest_identities,
    )
    raw_bad = [row for row in raw if row["raw_severity"] == "clear_bad"]
    output = require_output_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        case_rows: list[dict[str, Any]] = []
        for row in raw_bad:
            case_id = adjudication_case_id(
                pack_manifest["identity_sha256"], row["sample_index"]
            )
            destination_relative = f"images/{case_id}.png"
            destination = staging / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = require_file(
                review_pack / row["native_image_relative_path"],
                "raw-clear-bad native endpoint",
            )
            shutil.copyfile(source, destination)
            if sha256_file(destination) != row["source_png_sha256"]:
                raise RuntimeError("adjudication endpoint copy differs from reviewed PNG")
            case_rows.append(
                {
                    "case_id": case_id,
                    "class_id": row["class_id"],
                    "native_image_relative_path": destination_relative,
                }
            )
        cases_path = staging / "adjudication_cases.csv"
        template_path = staging / "adjudication_template.csv"
        instructions_path = staging / "adjudication_instructions.md"
        write_csv(cases_path, ADJUDICATION_CASE_COLUMNS, case_rows)
        write_csv(
            template_path,
            ADJUDICATION_COLUMNS,
            [{"case_id": row["case_id"], "action": ""} for row in case_rows],
        )
        instructions_path.write_text(ADJUDICATION_INSTRUCTIONS, encoding="utf-8")
        shutil.copy2(Path(__file__).resolve(), staging / "pipeline_source.py")
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "COMPLETE_RAW_MAJORITY_CLEAR_BAD_ONLY_ADJUDICATION_PACK",
            "blind_review_source_contract_identity_sha256": source_contract[
                "identity_sha256"
            ],
            "blind_review_source_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "review_pack": {
                "path": str(Path(review_pack).expanduser().absolute()),
                "record_identity_sha256": pack_record["identity_sha256"],
                "manifest_identity_sha256": pack_manifest["identity_sha256"],
                "sampling_pool_identity_sha256": pack_record["sampling_pool"][
                    "manifest_identity_sha256"
                ],
            },
            "review_locks": review_lineage,
            "raw_majority": {
                "trajectory_count": TRAJECTORY_COUNT,
                "raw_clear_bad_count": len(raw_bad),
                "clean_good_count": sum(
                    row["raw_severity"] == "clean_good" for row in raw
                ),
                "mild_or_disputed_count": sum(
                    row["raw_severity"] == "mild_or_disputed" for row in raw
                ),
                "raw_rows_identity_sha256": canonical_sha256(raw),
            },
            "adjudication_scope": {
                "input_population": "raw majority clear-bad only",
                "allowed_actions": list(ADJUDICATION_ACTIONS),
                "promotion_allowed": False,
                "non_raw_clear_bad_exposed": False,
            },
            "files": {
                "cases_member": "adjudication_cases.csv",
                "cases_file_sha256": sha256_file(cases_path),
                "template_member": "adjudication_template.csv",
                "template_file_sha256": sha256_file(template_path),
                "instructions_member": "adjudication_instructions.md",
                "instructions_file_sha256": sha256_file(instructions_path),
                "endpoint_image_count": len(raw_bad),
            },
            "access_audit": {
                "raw_three_review_locks_opened": True,
                "review_pack_lineage_validation_hashes_all_pack_members": True,
                "only_raw_majority_clear_bad_endpoint_images_copied": True,
                "individual_reviewer_votes_or_component_counts_exposed": False,
                "non_raw_clear_bad_endpoint_images_included_or_exposed_to_adjudicator": False,
                "trace_feature_score_threshold_rank_alert_old_label_or_screen_opened": False,
            },
            "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        record["identity_sha256"] = canonical_sha256(record)
        return finalize_record_staging(
            staging,
            output,
            artifact_kind=ADJUDICATION_PACK_ARTIFACT_KIND,
            record_name=ADJUDICATION_PACK_RECORD_NAME,
            record=record,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_adjudication_pack(
    root: Path,
    *,
    expected_manifest_identity: str,
    source_contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pack_record: Mapping[str, Any],
    pack_manifest: Mapping[str, Any],
    review_lineage: Mapping[str, Any],
    raw: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    raw_bad_for_tree = [row for row in raw if row["raw_severity"] == "clear_bad"]
    expected_files = {
        ADJUDICATION_PACK_RECORD_NAME,
        "adjudication_cases.csv",
        "adjudication_template.csv",
        "adjudication_instructions.md",
        "pipeline_source.py",
        "manifest.json",
        "completion.json",
        *(
            "images/"
            + adjudication_case_id(
                pack_manifest["identity_sha256"], int(row["sample_index"])
            )
            + ".png"
            for row in raw_bad_for_tree
        ),
    }
    _preflight_exact_names(root, expected_files, "adjudication pack")
    record, manifest = validate_record_lock(
        root,
        artifact_kind=ADJUDICATION_PACK_ARTIFACT_KIND,
        record_name=ADJUDICATION_PACK_RECORD_NAME,
        expected_manifest_identity=expected_manifest_identity,
    )
    raw_bad = [row for row in raw if row["raw_severity"] == "clear_bad"]
    pack_ref = record.get("review_pack", {})
    expected_scope = {
        "input_population": "raw majority clear-bad only",
        "allowed_actions": list(ADJUDICATION_ACTIONS),
        "promotion_allowed": False,
        "non_raw_clear_bad_exposed": False,
    }
    expected_access_audit = {
        "raw_three_review_locks_opened": True,
        "review_pack_lineage_validation_hashes_all_pack_members": True,
        "only_raw_majority_clear_bad_endpoint_images_copied": True,
        "individual_reviewer_votes_or_component_counts_exposed": False,
        "non_raw_clear_bad_endpoint_images_included_or_exposed_to_adjudicator": False,
        "trace_feature_score_threshold_rank_alert_old_label_or_screen_opened": False,
    }
    if (
        record.get("status")
        != "COMPLETE_RAW_MAJORITY_CLEAR_BAD_ONLY_ADJUDICATION_PACK"
        or record.get("blind_review_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
        or record.get("blind_review_source_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or record.get("implementation_source_sha256")
        != source_contract["implementation_source_sha256"]
        or pack_ref.get("record_identity_sha256") != pack_record["identity_sha256"]
        or pack_ref.get("manifest_identity_sha256") != pack_manifest["identity_sha256"]
        or pack_ref.get("sampling_pool_identity_sha256")
        != pack_record["sampling_pool"]["manifest_identity_sha256"]
        or record.get("review_locks") != review_lineage
        or record.get("adjudication_scope") != expected_scope
        or record.get("access_audit") != expected_access_audit
        or record.get("raw_majority", {}).get("trajectory_count") != TRAJECTORY_COUNT
        or record.get("raw_majority", {}).get("raw_clear_bad_count") != len(raw_bad)
        or record.get("raw_majority", {}).get("raw_rows_identity_sha256")
        != canonical_sha256(raw)
    ):
        raise RuntimeError("adjudication-pack scientific lineage changed")
    cases_path = root / "adjudication_cases.csv"
    template_path = root / "adjudication_template.csv"
    instructions_path = require_file(
        root / "adjudication_instructions.md", "adjudication instructions"
    )
    cases = read_csv(cases_path, ADJUDICATION_CASE_COLUMNS, "adjudication cases")
    template = read_csv(template_path, ADJUDICATION_COLUMNS, "adjudication template")
    expected_cases: list[dict[str, str]] = []
    for row in raw_bad:
        case_id = adjudication_case_id(
            pack_manifest["identity_sha256"], int(row["sample_index"])
        )
        image_relative = f"images/{case_id}.png"
        image_path = require_file(root / image_relative, "adjudication endpoint image")
        if sha256_file(image_path) != row["source_png_sha256"]:
            raise RuntimeError("adjudication image is not the reviewed endpoint")
        expected_cases.append(
            {
                "case_id": case_id,
                "class_id": str(row["class_id"]),
                "native_image_relative_path": image_relative,
            }
        )
    if (
        cases != expected_cases
        or template
        != [{"case_id": row["case_id"], "action": ""} for row in expected_cases]
        or instructions_path.read_text(encoding="utf-8") != ADJUDICATION_INSTRUCTIONS
        or record.get("files")
        != {
            "cases_member": "adjudication_cases.csv",
            "cases_file_sha256": sha256_file(cases_path),
            "template_member": "adjudication_template.csv",
            "template_file_sha256": sha256_file(template_path),
            "instructions_member": "adjudication_instructions.md",
            "instructions_file_sha256": sha256_file(instructions_path),
            "endpoint_image_count": len(raw_bad),
        }
    ):
        raise RuntimeError("adjudication cases/template changed")
    return record, manifest, expected_cases


ADJUDICATION_LOCK_ARTIFACT_KIND = "dit_bad_good_third_pool_adjudication_lock_v1"
ADJUDICATION_LOCK_RECORD_NAME = "adjudication_record.json"


def parse_adjudication_decisions(
    path: Path, expected_cases: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    rows = read_csv(path, ADJUDICATION_COLUMNS, "completed adjudication decisions")
    expected_ids = [str(row["case_id"]) for row in expected_cases]
    if len(rows) != len(expected_ids) or [row["case_id"] for row in rows] != expected_ids:
        raise RuntimeError("adjudication decisions do not cover the exact raw-clear-bad cases")
    for row in rows:
        if row["action"] not in ADJUDICATION_ACTIONS:
            raise RuntimeError(
                "adjudication action must retain clear-bad or downgrade to mild; "
                "promotion is forbidden"
            )
    return rows


def lock_adjudication(
    *,
    source_lock: Path,
    review_pack: Path,
    review_pack_manifest_identity: str,
    review_locks: Mapping[str, Path],
    review_lock_manifest_identities: Mapping[str, str],
    adjudication_pack: Path,
    adjudication_pack_manifest_identity: str,
    completed_adjudication_csv: Path,
    attest_blind: bool,
    output: Path,
) -> Path:
    if not attest_blind:
        raise RuntimeError("adjudicator must attest the frozen blindness contract")
    source_contract, source_manifest = validate_source_lock(source_lock)
    pack_record, pack_manifest, _, _, review_lineage, raw = load_review_chain(
        source_lock=source_lock,
        review_pack=review_pack,
        review_pack_manifest_identity=review_pack_manifest_identity,
        review_locks=review_locks,
        review_lock_manifest_identities=review_lock_manifest_identities,
    )
    adjudication_record, adjudication_manifest, cases = validate_adjudication_pack(
        adjudication_pack,
        expected_manifest_identity=adjudication_pack_manifest_identity,
        source_contract=source_contract,
        source_manifest=source_manifest,
        pack_record=pack_record,
        pack_manifest=pack_manifest,
        review_lineage=review_lineage,
        raw=raw,
    )
    decisions = parse_adjudication_decisions(completed_adjudication_csv, cases)
    output = require_output_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        decision_path = staging / "adjudication_decisions.csv"
        write_csv(decision_path, ADJUDICATION_COLUMNS, decisions)
        shutil.copy2(Path(__file__).resolve(), staging / "pipeline_source.py")
        action_counts = {
            action: sum(row["action"] == action for row in decisions)
            for action in ADJUDICATION_ACTIONS
        }
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "LOCKED_CONSERVATIVE_RAW_CLEAR_BAD_ADJUDICATION",
            "blind_review_source_contract_identity_sha256": source_contract[
                "identity_sha256"
            ],
            "blind_review_source_manifest_identity_sha256": source_manifest[
                "identity_sha256"
            ],
            "review_pack_record_identity_sha256": pack_record["identity_sha256"],
            "review_pack_manifest_identity_sha256": pack_manifest["identity_sha256"],
            "review_locks": review_lineage,
            "adjudication_pack": {
                "path": str(Path(adjudication_pack).expanduser().absolute()),
                "record_identity_sha256": adjudication_record["identity_sha256"],
                "manifest_identity_sha256": adjudication_manifest["identity_sha256"],
            },
            "decisions": {
                "member": "adjudication_decisions.csv",
                "columns_exact": list(ADJUDICATION_COLUMNS),
                "row_count": len(decisions),
                "file_sha256": sha256_file(decision_path),
                "action_counts": action_counts,
            },
            "adjudicator_attestation": {
                "visual_endpoint_only": True,
                "raw_majority_clear_bad_only": True,
                "candidate_hypothesis_seen": False,
                "metric_score_threshold_rank_or_alert_seen": False,
                "trajectory_or_intermediate_state_seen": False,
                "feature_product_seen": False,
                "old_label_or_screen_result_seen": False,
                "promotion_allowed": False,
            },
            "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        }
        record["identity_sha256"] = canonical_sha256(record)
        return finalize_record_staging(
            staging,
            output,
            artifact_kind=ADJUDICATION_LOCK_ARTIFACT_KIND,
            record_name=ADJUDICATION_LOCK_RECORD_NAME,
            record=record,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_adjudication_lock(
    root: Path,
    *,
    expected_manifest_identity: str | None = None,
    source_contract: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    pack_record: Mapping[str, Any],
    pack_manifest: Mapping[str, Any],
    review_lineage: Mapping[str, Any],
    adjudication_record: Mapping[str, Any],
    adjudication_manifest: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    _preflight_exact_names(
        root,
        {
            ADJUDICATION_LOCK_RECORD_NAME,
            "adjudication_decisions.csv",
            "pipeline_source.py",
            "manifest.json",
            "completion.json",
        },
        "adjudication lock",
    )
    record, manifest = validate_record_lock(
        root,
        artifact_kind=ADJUDICATION_LOCK_ARTIFACT_KIND,
        record_name=ADJUDICATION_LOCK_RECORD_NAME,
        expected_manifest_identity=expected_manifest_identity,
    )
    by_name = _safe_manifest_map(manifest, "adjudication lock")
    if set(by_name) != {
        ADJUDICATION_LOCK_RECORD_NAME,
        "adjudication_decisions.csv",
        "pipeline_source.py",
    }:
        raise RuntimeError("adjudication-lock member set changed")
    expected_attestation = {
        "visual_endpoint_only": True,
        "raw_majority_clear_bad_only": True,
        "candidate_hypothesis_seen": False,
        "metric_score_threshold_rank_or_alert_seen": False,
        "trajectory_or_intermediate_state_seen": False,
        "feature_product_seen": False,
        "old_label_or_screen_result_seen": False,
        "promotion_allowed": False,
    }
    pack_ref = record.get("adjudication_pack", {})
    if (
        record.get("status") != "LOCKED_CONSERVATIVE_RAW_CLEAR_BAD_ADJUDICATION"
        or record.get("blind_review_source_contract_identity_sha256")
        != source_contract["identity_sha256"]
        or record.get("blind_review_source_manifest_identity_sha256")
        != source_manifest["identity_sha256"]
        or record.get("review_pack_record_identity_sha256")
        != pack_record["identity_sha256"]
        or record.get("review_pack_manifest_identity_sha256")
        != pack_manifest["identity_sha256"]
        or record.get("review_locks") != review_lineage
        or pack_ref.get("record_identity_sha256")
        != adjudication_record["identity_sha256"]
        or pack_ref.get("manifest_identity_sha256")
        != adjudication_manifest["identity_sha256"]
        or record.get("adjudicator_attestation") != expected_attestation
        or record.get("implementation_source_sha256")
        != source_contract["implementation_source_sha256"]
    ):
        raise RuntimeError("adjudication-lock scientific lineage changed")
    decision_path = root / "adjudication_decisions.csv"
    decisions = parse_adjudication_decisions(decision_path, cases)
    action_counts = {
        action: sum(row["action"] == action for row in decisions)
        for action in ADJUDICATION_ACTIONS
    }
    if record.get("decisions") != {
        "member": "adjudication_decisions.csv",
        "columns_exact": list(ADJUDICATION_COLUMNS),
        "row_count": len(decisions),
        "file_sha256": sha256_file(decision_path),
        "action_counts": action_counts,
    }:
        raise RuntimeError("adjudication decision receipt changed")
    return record, manifest, decisions


def build_final_consensus_rows(
    raw: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, str]],
    *,
    pack_manifest_identity: str,
) -> list[dict[str, Any]]:
    by_case = {row["case_id"]: row["action"] for row in decisions}
    raw_bad_count = sum(row["raw_severity"] == "clear_bad" for row in raw)
    if len(by_case) != len(decisions) or len(by_case) != raw_bad_count:
        raise RuntimeError("adjudication decision cardinality differs from raw clear-bad")
    result: list[dict[str, Any]] = []
    used: set[str] = set()
    for row in raw:
        final_severity = str(row["raw_severity"])
        if final_severity == "clear_bad":
            case_id = adjudication_case_id(
                pack_manifest_identity, int(row["sample_index"])
            )
            action = by_case.get(case_id)
            if action == "retain_clear_bad":
                final_severity = "clear_bad"
            elif action == "downgrade_to_mild":
                final_severity = "mild_or_disputed"
            else:
                raise RuntimeError("missing or invalid raw-clear-bad adjudication")
            used.add(case_id)
        result.append(
            {
                "sample_index": int(row["sample_index"]),
                "global_seed": int(row["global_seed"]),
                "class_slot": int(row["class_slot"]),
                "class_id": int(row["class_id"]),
                "final_severity": final_severity,
                "blur_component_consensus": bool(row["blur_component_consensus"]),
                "discrete_structure_component_consensus": bool(
                    row["discrete_structure_component_consensus"]
                ),
            }
        )
    if used != set(by_case) or len(result) != TRAJECTORY_COUNT:
        raise RuntimeError("final adjudication used the wrong case set")
    return result


def consensus_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
            "clean_good": sum(
                row["final_severity"] == "clean_good" for row in subset
            ),
            "clear_bad": len(clear),
            "mild_or_disputed": sum(
                row["final_severity"] == "mild_or_disputed" for row in subset
            ),
            "blur_or_soft_fusion_clear_bad": len(blur),
            "mixed_blur_and_structure_clear_bad": len(mixed),
            "structural_non_blur_clear_bad": len(structural),
            "phenotype_disputed_clear_bad": len(disputed),
        }

    counts = {
        "overall": one(rows),
        "per_class": [
            {
                "class_id": class_id,
                **one([row for row in rows if row["class_id"] == class_id]),
            }
            for class_id in CLASSES
        ],
    }
    return evaluation.validate_aggregate_counts(counts)


def _validate_consensus_exact_tree(root: Path) -> None:
    root = require_directory(root, "final consensus lock")
    expected = {
        "manifest.json",
        "completion.json",
        evaluation.CONSENSUS_AGGREGATE_NAME,
        evaluation.CONSENSUS_ROWS_NAME,
    }
    observed: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"final consensus contains non-regular entry: {relative}")
        observed.add(relative)
    if observed != expected:
        raise RuntimeError("final consensus root must contain exactly the four frozen files")


def publish_consensus(
    *,
    source_lock: Path,
    review_pack: Path,
    review_pack_manifest_identity: str,
    review_locks: Mapping[str, Path],
    review_lock_manifest_identities: Mapping[str, str],
    adjudication_pack: Path,
    adjudication_pack_manifest_identity: str,
    adjudication_lock: Path,
    adjudication_lock_manifest_identity: str,
    output: Path,
) -> Path:
    source_contract, source_manifest = validate_source_lock(source_lock)
    pack_record, pack_manifest, _, _, review_lineage, raw = load_review_chain(
        source_lock=source_lock,
        review_pack=review_pack,
        review_pack_manifest_identity=review_pack_manifest_identity,
        review_locks=review_locks,
        review_lock_manifest_identities=review_lock_manifest_identities,
    )
    adjudication_record, adjudication_manifest, cases = validate_adjudication_pack(
        adjudication_pack,
        expected_manifest_identity=adjudication_pack_manifest_identity,
        source_contract=source_contract,
        source_manifest=source_manifest,
        pack_record=pack_record,
        pack_manifest=pack_manifest,
        review_lineage=review_lineage,
        raw=raw,
    )
    decision_record, decision_manifest, decisions = validate_adjudication_lock(
        adjudication_lock,
        expected_manifest_identity=adjudication_lock_manifest_identity,
        source_contract=source_contract,
        source_manifest=source_manifest,
        pack_record=pack_record,
        pack_manifest=pack_manifest,
        review_lineage=review_lineage,
        adjudication_record=adjudication_record,
        adjudication_manifest=adjudication_manifest,
        cases=cases,
    )
    rows = build_final_consensus_rows(
        raw, decisions, pack_manifest_identity=pack_manifest["identity_sha256"]
    )
    counts = consensus_counts(rows)
    output = require_output_target(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        rows_path = staging / evaluation.CONSENSUS_ROWS_NAME
        write_csv(
            rows_path,
            CONSENSUS_COLUMNS,
            [
                {
                    **{key: row[key] for key in CONSENSUS_COLUMNS[:5]},
                    "blur_component_consensus": str(
                        row["blur_component_consensus"]
                    ).lower(),
                    "discrete_structure_component_consensus": str(
                        row["discrete_structure_component_consensus"]
                    ).lower(),
                }
                for row in rows
            ],
        )
        aggregate: dict[str, Any] = {
            "schema_version": 1,
            "status": "FROZEN_THIRD_POOL_BLIND_CONSENSUS_COUNTS",
            "experiment": evaluation.CONSENSUS_EXPERIMENT,
            "phase1_protocol_identity_sha256": evaluation.EXPECTED_PHASE1_PROTOCOL_IDENTITY,
            "sampling_protocol_identity_sha256": evaluation.EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
            "sampling_pool_identity_sha256": pack_record["sampling_pool"][
                "manifest_identity_sha256"
            ],
            "classes_ordered": list(CLASSES),
            "global_seeds": list(SEEDS),
            "trajectory_count": TRAJECTORY_COUNT,
            "labels_and_phenotypes_immutable": True,
            "three_independent_endpoint_only_reviewers": True,
            "reviewers_score_threshold_alert_trajectory_blind": True,
            "consensus_rows_member": evaluation.CONSENSUS_ROWS_NAME,
            "consensus_rows_file_sha256": sha256_file(rows_path),
            "blind_review_audit_lineage": {
                "source_contract_identity_sha256": source_contract["identity_sha256"],
                "source_manifest_identity_sha256": source_manifest["identity_sha256"],
                "review_pack_record_identity_sha256": pack_record["identity_sha256"],
                "review_pack_manifest_identity_sha256": pack_manifest["identity_sha256"],
                "review_lock_manifest_identities": {
                    reviewer: review_lineage[reviewer]["manifest_identity_sha256"]
                    for reviewer in REVIEWERS
                },
                "adjudication_pack_record_identity_sha256": adjudication_record[
                    "identity_sha256"
                ],
                "adjudication_pack_manifest_identity_sha256": adjudication_manifest[
                    "identity_sha256"
                ],
                "adjudication_record_identity_sha256": decision_record[
                    "identity_sha256"
                ],
                "adjudication_manifest_identity_sha256": decision_manifest[
                    "identity_sha256"
                ],
            },
            "counts": counts,
        }
        aggregate["identity_sha256"] = canonical_sha256(aggregate)
        aggregate_path = staging / evaluation.CONSENSUS_AGGREGATE_NAME
        write_json(aggregate_path, aggregate)
        files = _artifact_records(staging)
        if {row["name"] for row in files} != {
            evaluation.CONSENSUS_AGGREGATE_NAME,
            evaluation.CONSENSUS_ROWS_NAME,
        }:
            raise RuntimeError("consensus staging contains an unexpected member")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "experiment": evaluation.CONSENSUS_EXPERIMENT,
            "aggregate_counts_identity_sha256": aggregate["identity_sha256"],
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        manifest_path = staging / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(manifest_path),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "aggregate_counts_file_sha256": sha256_file(aggregate_path),
                "aggregate_counts_identity_sha256": aggregate["identity_sha256"],
            },
        )
        os.replace(staging, output)
        _validate_consensus_exact_tree(output)
        receipt = evaluation.load_consensus_aggregate_only(
            {
                "path": str(output),
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
            {
                "manifest_identity_sha256": pack_record["sampling_pool"][
                    "manifest_identity_sha256"
                ]
            },
        )
        evaluation.load_full_consensus(
            {
                "path": str(output),
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
            {
                **{
                    key: receipt[key]
                    for key in (
                        "manifest_identity_sha256",
                        "aggregate_identity_sha256",
                        "row_member_declared_sha256",
                    )
                },
                "counts": receipt["counts"],
            },
        )
        return output.resolve()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_synthetic_pool(root: Path) -> str:
    """Create receipt/manifests/endpoints only; forbidden payloads stay absent."""

    root.mkdir()
    execution_plan = {
        "schema_version": 1,
        "status": "FROZEN_FOUR_GPU_CONTIGUOUS_EXECUTION_PLAN",
        "sampling_source_lock": "synthetic-frozen-source-lock",
        "sampling_protocol_identity_sha256": evaluation.EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
        "sampling_manifest_identity_sha256": evaluation.EXPECTED_SAMPLING_MANIFEST_IDENTITY,
        "phase1_protocol_identity_sha256": evaluation.EXPECTED_PHASE1_PROTOCOL_IDENTITY,
        "phase1_threshold_identity_sha256": evaluation.EXPECTED_PHASE1_THRESHOLD_IDENTITY,
        "trace_source": "synthetic-unopened-trace-source.py",
        "trace_source_sha256": hashlib.sha256(b"synthetic-trace-source").hexdigest(),
        "launcher_source_sha256": hashlib.sha256(b"synthetic-launcher").hexdigest(),
        "classes_ordered": list(CLASSES),
        "global_seeds": list(SEEDS),
        "global_seed_count": len(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
        "gpus_ordered": ["0", "1", "2", "3"],
        "assignment": {},
        "assignment_kind": "four ordered contiguous seed blocks",
        "output_root": "synthetic-pool",
        "dit_root": "synthetic-dit",
        "checkpoint": "synthetic-checkpoint",
        "vae_snapshot": "synthetic-vae",
        "required_trace_arrays": [],
        "observation_only": True,
        "labels_reviews_screen_results_or_sample_scores_read": False,
        "score_label_join_performed": False,
    }
    execution_plan_path = root / "execution_plan.json"
    write_json(execution_plan_path, execution_plan)
    seed_receipts: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = root / f"third_pool_v1_seed{seed:03d}"
        images_root = seed_root / "images"
        images_root.mkdir(parents=True)
        output_records: list[dict[str, Any]] = []
        for slot, class_id in enumerate(CLASSES):
            relative = ENDPOINT_RELATIVE[class_id]
            path = seed_root / relative
            image = Image.new(
                "RGB",
                (TILE_SIZE, TILE_SIZE),
                (40 + slot * 70, 80 + slot * 50, 120 + slot * 30),
            )
            seed_offset = seed - SEEDS[0]
            image.putpixel(
                (0, 0),
                (seed_offset & 0xFF, (seed_offset >> 8) & 0xFF, slot),
            )
            image.save(path, format="PNG", pnginfo=PngInfo(), compress_level=9)
            byte_count, file_hash, pixel_hash = _inspect_png(path)
            output_records.append(
                {
                    "relative_path": relative,
                    "bytes": byte_count,
                    "sha256": file_hash,
                    "pixel_sha256": pixel_hash,
                    "mode": "RGB",
                    "size": [TILE_SIZE, TILE_SIZE],
                }
            )
        for relative in sorted(EXPECTED_SEED_OUTPUTS - set(ENDPOINT_RELATIVE.values())):
            output_records.append(
                {
                    "relative_path": relative,
                    "bytes": 0,
                    "sha256": hashlib.sha256(
                        f"synthetic-unopened|{seed}|{relative}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        output_records.sort(key=lambda row: str(row["relative_path"]))
        identity = {
            "runner": "trace_dit_imagenet256_custom_batch",
            "observation_only": True,
            "quality_score": None,
            "selection": None,
            "intervention": None,
            "protocol": {
                "class_ids_ordered": list(CLASSES),
                "global_torch_seed": seed,
                "sampling_steps": 250,
                "image_size": TILE_SIZE,
            },
        }
        identity_hash = canonical_sha256(identity)
        outputs_hash = canonical_sha256(output_records)
        manifest = {
            "schema": 1,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_hash,
            "outputs": output_records,
            "outputs_sha256": outputs_hash,
        }
        manifest_path = seed_root / "manifest.json"
        write_json(manifest_path, manifest)
        completion_path = seed_root / "completion.json"
        write_json(
            completion_path,
            {
                "schema": 1,
                "identity_sha256": identity_hash,
                "manifest_sha256": sha256_file(manifest_path),
                "outputs_sha256": outputs_hash,
                "output_count": len(EXPECTED_SEED_OUTPUTS),
            },
        )
        trace_hash = next(
            row["sha256"]
            for row in output_records
            if row["relative_path"] == "trace.npz"
        )
        seed_receipts.append(
            {
                "seed": seed,
                "relative_output": seed_root.name,
                "identity_sha256": identity_hash,
                "manifest_sha256": sha256_file(manifest_path),
                "completion_sha256": sha256_file(completion_path),
                "outputs_sha256": outputs_hash,
                "output_count": len(EXPECTED_SEED_OUTPUTS),
                "trace_npz_sha256": trace_hash,
            }
        )
    execution_plan_hash = sha256_file(execution_plan_path)
    pool_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "sampling_protocol_identity_sha256": evaluation.EXPECTED_SAMPLING_PROTOCOL_IDENTITY,
        "phase1_protocol_identity_sha256": evaluation.EXPECTED_PHASE1_PROTOCOL_IDENTITY,
        "phase1_threshold_identity_sha256": evaluation.EXPECTED_PHASE1_THRESHOLD_IDENTITY,
        "execution_plan_sha256": execution_plan_hash,
        "seed_count": len(SEEDS),
        "trajectory_count": TRAJECTORY_COUNT,
        "seed_outputs": seed_receipts,
        "runner_logs": [],
        "observation_only": True,
        "labels_reviews_screen_results_or_sample_scores_read": False,
        "score_label_join_performed": False,
    }
    pool_manifest["identity_sha256"] = canonical_sha256(pool_manifest)
    manifest_path = root / "pool_manifest.json"
    write_json(manifest_path, pool_manifest)
    write_json(
        root / "pool_completion.json",
        {
            "complete": True,
            "pool_identity_sha256": pool_manifest["identity_sha256"],
            "pool_manifest_sha256": sha256_file(manifest_path),
            "execution_plan_sha256": execution_plan_hash,
            "seed_count": len(SEEDS),
            "trajectory_count": TRAJECTORY_COUNT,
        },
    )
    return pool_manifest["identity_sha256"]


def _synthetic_completed_reviews(
    root: Path,
    orders: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for reviewer_index, reviewer in enumerate(REVIEWERS):
        rows: list[dict[str, Any]] = []
        for order_row in orders[reviewer]:
            sample_index = int(
                Path(str(order_row["native_image_relative_path"])).stem.split("_")[-1]
            )
            seed = SEEDS[0] + sample_index // len(CLASSES)
            slot = sample_index % len(CLASSES)
            local = seed - SEEDS[0]
            blur_bad = slot == 0 and local < 18
            structure_bad = slot == 1 and local < 15
            disputed_bad = slot == 2 and local < 3
            mild = slot == 2 and 10 <= local < 15
            target_bad = blur_bad or structure_bad or disputed_bad
            severity = 2 if target_bad and reviewer_index < 2 else 0
            if target_bad and reviewer_index == 2:
                severity = 1
            elif mild:
                severity = 1 if reviewer_index < 2 else 0
            flags = {flag: 0 for flag in COMPONENT_FLAGS}
            if blur_bad and reviewer_index == 0:
                flags["global_blur"] = 1
            elif blur_bad and reviewer_index == 1:
                flags["local_blur"] = 1
            if structure_bad and reviewer_index == 0:
                flags["discrete_duplication_or_extra_part"] = 1
            elif structure_bad and reviewer_index == 1:
                flags["limb_or_object_misalignment"] = 1
            rows.append(
                {
                    "review_id": order_row["review_id"],
                    "severity": severity,
                    **flags,
                    NONE_SENTINEL: int(not any(flags.values())),
                }
            )
        path = root / f"completed_{reviewer}.csv"
        write_csv(path, REVIEW_COLUMNS, rows)
        result[reviewer] = path
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="third-pool-blind-review-selftest-") as tmp:
        temporary = Path(tmp)
        source_lock = freeze_source_lock(temporary / "source_lock")
        source_contract, source_manifest = validate_source_lock(source_lock)
        pool = temporary / "synthetic_pool"
        pool_identity = _write_synthetic_pool(pool)
        review_pack = build_review_pack(
            source_lock=source_lock,
            sampling_pool=pool,
            sampling_pool_manifest_identity=pool_identity,
            output=temporary / "review_pack",
        )
        pack_record, pack_manifest, catalog, orders = validate_review_pack(
            review_pack,
            expected_manifest_identity=load_json(
                review_pack / "manifest.json", "synthetic review-pack manifest"
            )["identity_sha256"],
            source_lock=source_lock,
        )
        if (
            len({row["source_png_sha256"] for row in catalog}) != TRAJECTORY_COUNT
            or len({row["source_pixel_sha256"] for row in catalog})
            != TRAJECTORY_COUNT
        ):
            raise AssertionError(
                "synthetic endpoints must have distinct file and pixel hashes"
            )
        completed_reviews = _synthetic_completed_reviews(temporary, orders)
        malformed_review = temporary / "malformed_extra_cell_review.csv"
        malformed_lines = completed_reviews["reviewer_1"].read_text(
            encoding="utf-8"
        ).splitlines()
        malformed_lines[1] += ",unexpected"
        malformed_review.write_text(
            "\n".join(malformed_lines) + "\n", encoding="utf-8"
        )
        try:
            parse_completed_review(
                malformed_review,
                reviewer="reviewer_1",
                expected_order=orders["reviewer_1"],
            )
        except RuntimeError as exc:
            assert "missing or extra cells" in str(exc)
        else:
            raise AssertionError("review CSV extra cell was accepted")
        review_locks: dict[str, Path] = {}
        review_lock_manifest_identities: dict[str, str] = {}
        for reviewer in REVIEWERS:
            review_locks[reviewer] = lock_review(
                source_lock=source_lock,
                review_pack=review_pack,
                review_pack_manifest_identity=pack_manifest["identity_sha256"],
                reviewer=reviewer,
                completed_review_csv=completed_reviews[reviewer],
                attest_blind=True,
                output=temporary / f"{reviewer}_lock",
            )
            review_lock_manifest_identities[reviewer] = load_json(
                review_locks[reviewer] / "manifest.json",
                f"synthetic {reviewer} review-lock manifest",
            )["identity_sha256"]
        adjudication_pack = build_adjudication_pack(
            source_lock=source_lock,
            review_pack=review_pack,
            review_pack_manifest_identity=pack_manifest["identity_sha256"],
            review_locks=review_locks,
            review_lock_manifest_identities=review_lock_manifest_identities,
            output=temporary / "adjudication_pack",
        )
        adjudication_pack_manifest = load_json(
            adjudication_pack / "manifest.json", "synthetic adjudication-pack manifest"
        )
        _, _, _, _, review_lineage, raw = load_review_chain(
            source_lock=source_lock,
            review_pack=review_pack,
            review_pack_manifest_identity=pack_manifest["identity_sha256"],
            review_locks=review_locks,
            review_lock_manifest_identities=review_lock_manifest_identities,
        )
        adjudication_record, _, cases = validate_adjudication_pack(
            adjudication_pack,
            expected_manifest_identity=adjudication_pack_manifest["identity_sha256"],
            source_contract=source_contract,
            source_manifest=source_manifest,
            pack_record=pack_record,
            pack_manifest=pack_manifest,
            review_lineage=review_lineage,
            raw=raw,
        )
        assert len(cases) == adjudication_record["raw_majority"]["raw_clear_bad_count"] == 36
        invalid_path = temporary / "invalid_adjudication.csv"
        write_csv(
            invalid_path,
            ADJUDICATION_COLUMNS,
            [
                {
                    "case_id": row["case_id"],
                    "action": (
                        "promote_to_clear_bad" if index == 0 else "retain_clear_bad"
                    ),
                }
                for index, row in enumerate(cases)
            ],
        )
        try:
            parse_adjudication_decisions(invalid_path, cases)
        except RuntimeError as exc:
            assert "promotion is forbidden" in str(exc)
        else:
            raise AssertionError("forbidden adjudication promotion was accepted")
        completed_adjudication = temporary / "completed_adjudication.csv"
        write_csv(
            completed_adjudication,
            ADJUDICATION_COLUMNS,
            [
                {
                    "case_id": row["case_id"],
                    "action": (
                        "downgrade_to_mild" if index < 2 else "retain_clear_bad"
                    ),
                }
                for index, row in enumerate(cases)
            ],
        )
        adjudication_lock = lock_adjudication(
            source_lock=source_lock,
            review_pack=review_pack,
            review_pack_manifest_identity=pack_manifest["identity_sha256"],
            review_locks=review_locks,
            review_lock_manifest_identities=review_lock_manifest_identities,
            adjudication_pack=adjudication_pack,
            adjudication_pack_manifest_identity=adjudication_pack_manifest[
                "identity_sha256"
            ],
            completed_adjudication_csv=completed_adjudication,
            attest_blind=True,
            output=temporary / "adjudication_lock",
        )
        adjudication_lock_manifest = load_json(
            adjudication_lock / "manifest.json", "synthetic adjudication-lock manifest"
        )
        consensus = publish_consensus(
            source_lock=source_lock,
            review_pack=review_pack,
            review_pack_manifest_identity=pack_manifest["identity_sha256"],
            review_locks=review_locks,
            review_lock_manifest_identities=review_lock_manifest_identities,
            adjudication_pack=adjudication_pack,
            adjudication_pack_manifest_identity=adjudication_pack_manifest[
                "identity_sha256"
            ],
            adjudication_lock=adjudication_lock,
            adjudication_lock_manifest_identity=adjudication_lock_manifest[
                "identity_sha256"
            ],
            output=temporary / "consensus",
        )
        aggregate = load_json(
            consensus / evaluation.CONSENSUS_AGGREGATE_NAME,
            "synthetic consensus aggregate",
        )
        assert aggregate["counts"]["overall"]["clear_bad"] == 34
        assert aggregate["counts"]["overall"]["blur_or_soft_fusion_clear_bad"] == 17
        consensus_manifest = load_json(
            consensus / "manifest.json", "synthetic consensus manifest"
        )
        poison_primary = temporary / "MUST_NOT_EXIST_primary_features"
        poison_visual = temporary / "MUST_NOT_EXIST_visual_features"
        input_lock = evaluation.bind_inputs(
            source_lock=EVALUATOR_SOURCE_LOCK,
            sampling_pool_path=pool,
            sampling_pool_manifest_identity=pool_identity,
            consensus_path=consensus,
            consensus_manifest_identity=consensus_manifest["identity_sha256"],
            primary_path=poison_primary,
            primary_manifest_identity="a" * 64,
            visual_path=poison_visual,
            visual_manifest_identity="b" * 64,
            output=temporary / "evaluator_input_binding",
        )
        stage_a = evaluation.run_stage_a(
            input_lock=input_lock, output=temporary / "evaluator_stage_a"
        )
        stage_a_record = load_json(
            stage_a / "stage_a_gate_receipt.json", "synthetic Stage-A receipt"
        )
        assert stage_a_record["status"] == "EVENT_GATE_PASSED_SCORES_STILL_UNOPENED"
        assert not poison_primary.exists() and not poison_visual.exists()
        assert not any(pool.glob("third_pool_v1_seed*/trace.npz"))
        assert not any(pool.glob("third_pool_v1_seed*/sample.png"))
    print(
        "self-test passed: full 1800-row synthetic pool, terminal-only access, "
        "three distinct review orders/locks, broad-group majority, raw-clear-bad-only "
        "non-promoting adjudication, exact-tree evaluator-v5 consensus, and Stage-A "
        "pass without feature-path access"
    )


def _review_lock_args(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        "reviewer_1": args.reviewer_1_lock,
        "reviewer_2": args.reviewer_2_lock,
        "reviewer_3": args.reviewer_3_lock,
    }
    missing = [reviewer for reviewer, value in values.items() if value is None]
    if missing:
        raise RuntimeError("missing review locks: " + ", ".join(missing))
    return {reviewer: Path(value) for reviewer, value in values.items()}


def _review_lock_identity_args(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "reviewer_1": args.reviewer_1_lock_manifest_identity,
        "reviewer_2": args.reviewer_2_lock_manifest_identity,
        "reviewer_3": args.reviewer_3_lock_manifest_identity,
    }
    missing = [reviewer for reviewer, value in values.items() if value is None]
    if missing:
        raise RuntimeError(
            "missing review-lock manifest identities: " + ", ".join(missing)
        )
    return {
        reviewer: require_hex64(value, f"{reviewer} bound review-lock identity")
        for reviewer, value in values.items()
    }


def _print_record_lock(path: Path, artifact_kind: str, record_name: str) -> None:
    record, manifest = validate_record_lock(
        path, artifact_kind=artifact_kind, record_name=record_name
    )
    print(
        json.dumps(
            {
                "output": str(path),
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
    modes.add_argument("--build-review-pack", action="store_true")
    modes.add_argument("--lock-review", action="store_true")
    modes.add_argument("--build-adjudication-pack", action="store_true")
    modes.add_argument("--lock-adjudication", action="store_true")
    modes.add_argument("--build-consensus", action="store_true")
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--sampling-pool", type=Path)
    parser.add_argument("--sampling-pool-manifest-identity")
    parser.add_argument("--review-pack", type=Path)
    parser.add_argument("--review-pack-manifest-identity")
    parser.add_argument("--reviewer-id", choices=REVIEWERS)
    parser.add_argument("--completed-review-csv", type=Path)
    parser.add_argument("--reviewer-1-lock", type=Path)
    parser.add_argument("--reviewer-2-lock", type=Path)
    parser.add_argument("--reviewer-3-lock", type=Path)
    parser.add_argument("--reviewer-1-lock-manifest-identity")
    parser.add_argument("--reviewer-2-lock-manifest-identity")
    parser.add_argument("--reviewer-3-lock-manifest-identity")
    parser.add_argument("--adjudication-pack", type=Path)
    parser.add_argument("--adjudication-pack-manifest-identity")
    parser.add_argument("--completed-adjudication-csv", type=Path)
    parser.add_argument("--adjudication-lock", type=Path)
    parser.add_argument("--adjudication-lock-manifest-identity")
    parser.add_argument("--attest-blind", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.freeze_source_lock:
        if args.output is None:
            parser.error("--freeze-source-lock requires --output")
        path = freeze_source_lock(args.output)
        _print_record_lock(path, SOURCE_ARTIFACT_KIND, SOURCE_RECORD_NAME)
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
    if args.build_review_pack:
        if (
            args.sampling_pool is None
            or args.sampling_pool_manifest_identity is None
            or args.output is None
        ):
            parser.error(
                "--build-review-pack requires --sampling-pool, "
                "--sampling-pool-manifest-identity, and --output"
            )
        path = build_review_pack(
            source_lock=args.source_lock,
            sampling_pool=args.sampling_pool,
            sampling_pool_manifest_identity=args.sampling_pool_manifest_identity,
            output=args.output,
        )
        _print_record_lock(path, PACK_ARTIFACT_KIND, PACK_RECORD_NAME)
        return 0
    if args.lock_review:
        if (
            args.review_pack is None
            or args.review_pack_manifest_identity is None
            or args.reviewer_id is None
            or args.completed_review_csv is None
            or args.output is None
        ):
            parser.error(
                "--lock-review requires --review-pack, --review-pack-manifest-identity, "
                "--reviewer-id, --completed-review-csv, and --output"
            )
        path = lock_review(
            source_lock=args.source_lock,
            review_pack=args.review_pack,
            review_pack_manifest_identity=args.review_pack_manifest_identity,
            reviewer=args.reviewer_id,
            completed_review_csv=args.completed_review_csv,
            attest_blind=args.attest_blind,
            output=args.output,
        )
        _print_record_lock(path, REVIEW_ARTIFACT_KIND, REVIEW_RECORD_NAME)
        return 0
    if args.build_adjudication_pack:
        if (
            args.review_pack is None
            or args.review_pack_manifest_identity is None
            or args.output is None
        ):
            parser.error(
                "--build-adjudication-pack requires --review-pack, "
                "--review-pack-manifest-identity, three reviewer locks, and --output"
            )
        path = build_adjudication_pack(
            source_lock=args.source_lock,
            review_pack=args.review_pack,
            review_pack_manifest_identity=args.review_pack_manifest_identity,
            review_locks=_review_lock_args(args),
            review_lock_manifest_identities=_review_lock_identity_args(args),
            output=args.output,
        )
        _print_record_lock(
            path, ADJUDICATION_PACK_ARTIFACT_KIND, ADJUDICATION_PACK_RECORD_NAME
        )
        return 0
    if args.lock_adjudication:
        if (
            args.review_pack is None
            or args.review_pack_manifest_identity is None
            or args.adjudication_pack is None
            or args.adjudication_pack_manifest_identity is None
            or args.completed_adjudication_csv is None
            or args.output is None
        ):
            parser.error(
                "--lock-adjudication requires review-pack lineage, three reviewer "
                "locks, adjudication-pack lineage, completed decisions, and --output"
            )
        path = lock_adjudication(
            source_lock=args.source_lock,
            review_pack=args.review_pack,
            review_pack_manifest_identity=args.review_pack_manifest_identity,
            review_locks=_review_lock_args(args),
            review_lock_manifest_identities=_review_lock_identity_args(args),
            adjudication_pack=args.adjudication_pack,
            adjudication_pack_manifest_identity=args.adjudication_pack_manifest_identity,
            completed_adjudication_csv=args.completed_adjudication_csv,
            attest_blind=args.attest_blind,
            output=args.output,
        )
        _print_record_lock(
            path, ADJUDICATION_LOCK_ARTIFACT_KIND, ADJUDICATION_LOCK_RECORD_NAME
        )
        return 0
    if args.build_consensus:
        if (
            args.review_pack is None
            or args.review_pack_manifest_identity is None
            or args.adjudication_pack is None
            or args.adjudication_pack_manifest_identity is None
            or args.adjudication_lock is None
            or args.adjudication_lock_manifest_identity is None
            or args.output is None
        ):
            parser.error(
                "--build-consensus requires review-pack lineage, three reviewer locks, "
                "adjudication-pack/lock lineages, and --output"
            )
        path = publish_consensus(
            source_lock=args.source_lock,
            review_pack=args.review_pack,
            review_pack_manifest_identity=args.review_pack_manifest_identity,
            review_locks=_review_lock_args(args),
            review_lock_manifest_identities=_review_lock_identity_args(args),
            adjudication_pack=args.adjudication_pack,
            adjudication_pack_manifest_identity=args.adjudication_pack_manifest_identity,
            adjudication_lock=args.adjudication_lock,
            adjudication_lock_manifest_identity=args.adjudication_lock_manifest_identity,
            output=args.output,
        )
        manifest = load_json(path / "manifest.json", "final consensus manifest")
        aggregate = load_json(
            path / evaluation.CONSENSUS_AGGREGATE_NAME, "final consensus aggregate"
        )
        print(
            json.dumps(
                {
                    "output": str(path),
                    "manifest_identity_sha256": manifest["identity_sha256"],
                    "aggregate_identity_sha256": aggregate["identity_sha256"],
                    "counts": aggregate["counts"],
                    "status": aggregate["status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable mode")


if __name__ == "__main__":
    raise SystemExit(main())
