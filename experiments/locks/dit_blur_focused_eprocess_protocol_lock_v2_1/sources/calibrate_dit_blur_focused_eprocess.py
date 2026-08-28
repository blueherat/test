#!/usr/bin/env python3
"""Freeze label-free B thresholds for the blur-gated DiT e-process.

The only numerical trajectory feature loaded from the source product is
``decoded_local_blur_severity`` at the nine frozen preterminal checkpoints.
Identifiers and checkpoint axes are loaded solely to validate the 20-path
per-class calibration design.  The CSV feature table, endpoint paths/images,
quality labels, reviews, candidate scores, and all ResNet/other tracks in the
same source archive are never loaded.

This script calibrates an internal state gate; it does not evaluate quality.
Its 17th/20 and 19th/20 order statistics have exchangeability rank meanings,
not clean-image false-positive meanings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np

try:
    from . import observe_dit_blur_focused_eprocess as core
except ImportError:  # pragma: no cover - direct CLI execution
    import observe_dit_blur_focused_eprocess as core


SCHEMA_VERSION = 1
EXPERIMENT = "dit_blur_focused_eprocess_label_free_calibration"
SOURCE_EXPERIMENT = "dit_predxstart_preterminal_visual_tracks_label_free"
SOURCE_ARCHIVE = "time_series.npz"
CALIBRATION_COUNT_PER_CLASS = 20
STATE_GATE_ORDER_INDEX = 16  # zero based: 17th ascending
PURE_B_ORDER_INDEX = 18  # zero based: 19th ascending
CALIBRATION_KEYS = (
    "schema_version",
    "status",
    "calibration_seed_count_per_class",
    "state_gate_order_statistic",
    "pure_B_order_statistic",
    "source_product",
    "ordered_global_seeds",
    "loaded_array_records",
    "calibrator_source_sha256",
    "classes",
    "identity_sha256",
)
CALIBRATION_CLASS_KEYS = (
    "class_id",
    "blur_gate_threshold_by_checkpoint",
    "blur_score_threshold",
)
LOADED_ARRAY_NAMES = (
    "sample_index",
    "global_seed",
    "class_slot",
    "class_id",
    "selected_sampling_step",
    "selected_internal_timestep",
    "decoded_local_blur_severity",
)
FORBIDDEN_SOURCE_NAME_TOKENS = (
    "label",
    "review",
    "annotation",
    "consensus",
    "candidate",
    "endpoint_embedding",
    "inception",
    "dino",
    "clip",
    "fid",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"expected regular JSON file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _self_hash(payload: Mapping[str, Any], key: str) -> str:
    value = dict(payload)
    value.pop(key, None)
    return core._sha256_json(value)


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("source manifest has no file records")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "bytes", "sha256"}:
            raise RuntimeError("source manifest file record is malformed")
        name = row.get("name")
        if not isinstance(name, str) or name in result:
            raise RuntimeError("source manifest file names are invalid or duplicated")
        result[name] = row
    return result


def _validate_bound_file(root: Path, record: Mapping[str, Any]) -> Path:
    path = root / str(record["name"])
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular source product member: {path}")
    if path.stat().st_size != record.get("bytes") or core._sha256_file(path) != record.get(
        "sha256"
    ):
        raise RuntimeError(f"source product member differs from its manifest: {path}")
    return path


def _read_source_product(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = root.expanduser().absolute()
    lowered = root.name.lower()
    if any(token in lowered for token in FORBIDDEN_SOURCE_NAME_TOKENS):
        raise RuntimeError("source product path name looks supervised or externally scored")
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"source product must be one regular directory: {root}")
    root = root.resolve()
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = _load_json(manifest_path)
    completion = _load_json(completion_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment") != SOURCE_EXPERIMENT
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("source is not the completed frozen preterminal visual product")
    identity = dict(manifest)
    recorded_identity = identity.pop("identity_sha256", None)
    if recorded_identity != core._sha256_json(identity):
        raise RuntimeError("source manifest identity is invalid")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != recorded_identity
        or completion.get("manifest_file_sha256") != core._sha256_file(manifest_path)
        or completion.get("payload_sha256") != _self_hash(completion, "payload_sha256")
    ):
        raise RuntimeError("source completion receipt does not bind the manifest")
    records = _manifest_records(manifest)
    required_files = {
        SOURCE_ARCHIVE,
        "source_inventory.json",
        "protocol_snapshot.json",
        "summary.json",
        "provenance.json",
    }
    if not required_files.issubset(records):
        raise RuntimeError("source product lacks a required label-free lineage member")
    paths = {name: _validate_bound_file(root, records[name]) for name in required_files}

    summary = _load_json(paths["summary.json"])
    protocol = _load_json(paths["protocol_snapshot.json"])
    provenance = _load_json(paths["provenance.json"])
    inventory = _load_json(paths["source_inventory.json"])
    supervision = protocol.get("supervision_policy")
    if (
        summary.get("experiment") != SOURCE_EXPERIMENT
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("decoded_images_saved") is not False
        or protocol.get("experiment") != SOURCE_EXPERIMENT
        or protocol.get("status") != "LABEL_FREE_OBSERVATION_ONLY"
        or not isinstance(supervision, dict)
        or any(supervision.get(key) is not False for key in (
            "labels_read_or_emitted",
            "reviews_read",
            "candidate_scores_read",
            "calibration_thresholds_read",
            "alerts_read",
            "auc_or_selection_computed",
        ))
        or provenance.get("decoded_images_saved") is not False
    ):
        raise RuntimeError("source product does not attest the strict label-free boundary")
    records_by_array = inventory.get("time_series_arrays")
    if not isinstance(records_by_array, dict) or not set(LOADED_ARRAY_NAMES).issubset(
        records_by_array
    ):
        raise RuntimeError("source time-series inventory lacks frozen B calibration arrays")

    archive_path = paths[SOURCE_ARCHIVE]
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if not set(LOADED_ARRAY_NAMES).issubset(archive.files):
                raise RuntimeError("source archive lacks frozen B calibration arrays")
            # Deliberately index only this whitelist.  Merely coexisting tracks,
            # including ResNet diagnostics, never enter memory or the method.
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in LOADED_ARRAY_NAMES
            }
            unused_names = sorted(set(archive.files) - set(LOADED_ARRAY_NAMES))
    except (OSError, ValueError) as exc:
        raise RuntimeError("cannot read the label-free B calibration arrays") from exc
    for name, value in arrays.items():
        if core._array_record(value) != records_by_array.get(name):
            raise RuntimeError(f"source B calibration array differs from inventory: {name}")
        if not np.isfinite(value).all():
            raise RuntimeError(f"source B calibration array is non-finite: {name}")

    metadata = {
        "experiment": SOURCE_EXPERIMENT,
        "manifest_identity_sha256": recorded_identity,
        "manifest_file_sha256": core._sha256_file(manifest_path),
        "time_series_file_sha256": core._sha256_file(archive_path),
        "unused_archive_members_not_loaded": unused_names,
    }
    return arrays, metadata


def derive_calibration(
    arrays: Mapping[str, np.ndarray], source_product: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the two fixed order-statistic thresholds without quality data."""

    if set(arrays) != set(LOADED_ARRAY_NAMES):
        raise RuntimeError("calibrator received arrays outside its exact whitelist")
    sample_index = np.asarray(arrays["sample_index"])
    seeds = np.asarray(arrays["global_seed"])
    slots = np.asarray(arrays["class_slot"])
    classes = np.asarray(arrays["class_id"])
    steps = np.asarray(arrays["selected_sampling_step"])
    timesteps = np.asarray(arrays["selected_internal_timestep"])
    blur = np.asarray(arrays["decoded_local_blur_severity"])
    row_count = len(sample_index)
    if (
        sample_index.dtype != np.int32
        or seeds.dtype != np.int64
        or slots.dtype != np.int16
        or classes.dtype != np.int16
        or blur.dtype != np.float64
        or sample_index.shape != (row_count,)
        or seeds.shape != (row_count,)
        or slots.shape != (row_count,)
        or classes.shape != (row_count,)
        or blur.shape != (row_count, len(core.CHECKPOINTS))
    ):
        raise RuntimeError("calibration row arrays have the wrong frozen shape or dtype")
    if not np.array_equal(sample_index, np.arange(row_count, dtype=np.int32)):
        raise RuntimeError("calibration sample_index is not consecutive")
    if steps.dtype != np.int16 or not np.array_equal(
        steps, np.asarray(core.CHECKPOINTS, dtype=np.int16)
    ):
        raise RuntimeError("calibration sampling checkpoints changed")
    if timesteps.dtype != np.int16 or not np.array_equal(
        timesteps, np.asarray(core.INTERNAL_TIMESTEPS, dtype=np.int16)
    ):
        raise RuntimeError("calibration internal timesteps changed")
    if row_count < CALIBRATION_COUNT_PER_CLASS or not np.isfinite(blur).all():
        raise RuntimeError("calibration B rows are empty or non-finite")
    if np.any(classes < 0) or np.any(classes >= 1000) or np.any(slots < 0):
        raise RuntimeError("calibration class identifiers are invalid")

    ordered_classes = sorted(int(value) for value in np.unique(classes))
    if not ordered_classes:
        raise RuntimeError("calibration contains no classes")
    common_seed_tuple: tuple[int, ...] | None = None
    class_rows: list[dict[str, Any]] = []
    for class_id in ordered_classes:
        selected = np.flatnonzero(classes == class_id)
        if len(selected) != CALIBRATION_COUNT_PER_CLASS:
            raise RuntimeError("every class must have exactly 20 calibration paths")
        class_seeds = tuple(sorted(int(value) for value in seeds[selected]))
        if len(set(class_seeds)) != CALIBRATION_COUNT_PER_CLASS:
            raise RuntimeError("a class has duplicate calibration seeds")
        if common_seed_tuple is None:
            common_seed_tuple = class_seeds
        elif class_seeds != common_seed_tuple:
            raise RuntimeError("all selected classes must share the same 20-seed cohort")
        class_slots = np.unique(slots[selected])
        if len(class_slots) != 1:
            raise RuntimeError("a selected class occupies inconsistent class slots")
        values = blur[selected]
        persistence = np.mean(values, axis=1)
        gate = np.sort(values, axis=0, kind="stable")[STATE_GATE_ORDER_INDEX]
        score = float(np.sort(persistence, kind="stable")[PURE_B_ORDER_INDEX])
        class_rows.append(
            {
                "class_id": class_id,
                "blur_gate_threshold_by_checkpoint": [float(value) for value in gate],
                "blur_score_threshold": score,
            }
        )

    if common_seed_tuple is None:
        raise AssertionError("unreachable empty calibration cohort")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "LABEL_FREE_LOCKED",
        "calibration_seed_count_per_class": CALIBRATION_COUNT_PER_CLASS,
        "state_gate_order_statistic": "17th ascending; strict greater",
        "pure_B_order_statistic": "19th ascending; strict greater",
        "source_product": dict(source_product),
        "ordered_global_seeds": list(common_seed_tuple),
        "loaded_array_records": {
            name: core._array_record(np.asarray(arrays[name]))
            for name in LOADED_ARRAY_NAMES
        },
        "calibrator_source_sha256": core._sha256_file(Path(__file__).resolve()),
        "classes": class_rows,
    }
    payload["identity_sha256"] = core._sha256_json(payload)
    validate_calibration(payload)
    return payload


def validate_calibration(payload: Mapping[str, Any]) -> None:
    if set(payload) != set(CALIBRATION_KEYS):
        raise RuntimeError("calibration JSON keys differ from the frozen schema")
    if payload.get("identity_sha256") != _self_hash(payload, "identity_sha256"):
        raise RuntimeError("calibration identity hash is invalid")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "LABEL_FREE_LOCKED"
        or payload.get("calibration_seed_count_per_class")
        != CALIBRATION_COUNT_PER_CLASS
        or payload.get("state_gate_order_statistic")
        != "17th ascending; strict greater"
        or payload.get("pure_B_order_statistic")
        != "19th ascending; strict greater"
    ):
        raise RuntimeError("calibration order-statistic contract changed")
    source = payload.get("source_product")
    if not isinstance(source, dict) or source.get("experiment") != SOURCE_EXPERIMENT:
        raise RuntimeError("calibration source lineage is missing")
    for key in (
        "manifest_identity_sha256",
        "manifest_file_sha256",
        "time_series_file_sha256",
    ):
        value = source.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError("calibration source hash is malformed")
    seeds = payload.get("ordered_global_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != CALIBRATION_COUNT_PER_CLASS
        or len(set(seeds)) != CALIBRATION_COUNT_PER_CLASS
        or not all(type(value) is int for value in seeds)
    ):
        raise RuntimeError("calibration seed cohort is malformed")
    records = payload.get("loaded_array_records")
    if not isinstance(records, dict) or set(records) != set(LOADED_ARRAY_NAMES):
        raise RuntimeError("calibration loaded-array audit is malformed")
    source_hash = payload.get("calibrator_source_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise RuntimeError("calibrator source hash is malformed")
    rows = payload.get("classes")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("calibration has no class thresholds")
    observed: set[int] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(CALIBRATION_CLASS_KEYS):
            raise RuntimeError("calibration class row is malformed")
        class_id = row.get("class_id")
        gate = row.get("blur_gate_threshold_by_checkpoint")
        score = row.get("blur_score_threshold")
        if type(class_id) is not int or class_id in observed or not 0 <= class_id < 1000:
            raise RuntimeError("calibration class id is invalid or duplicated")
        if (
            not isinstance(gate, list)
            or len(gate) != len(core.CHECKPOINTS)
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in gate)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise RuntimeError("calibration thresholds are malformed")
        observed.add(class_id)
    if [row["class_id"] for row in rows] != sorted(observed):
        raise RuntimeError("calibration class rows must be sorted")


def publish(source_root: Path, output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to overwrite calibration output: {output}")
    arrays, source = _read_source_product(source_root)
    payload = derive_calibration(arrays, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    core._atomic_json_dump(payload, output)
    observed = _load_json(output)
    validate_calibration(observed)
    if observed != payload:
        raise RuntimeError("calibration changed during atomic publication")
    return output


def _synthetic_arrays() -> dict[str, np.ndarray]:
    class_ids = (7, 11, 29)
    seeds = np.arange(1000, 1000 + CALIBRATION_COUNT_PER_CLASS, dtype=np.int64)
    rows = len(class_ids) * len(seeds)
    classes = np.tile(np.asarray(class_ids, dtype=np.int16), len(seeds))
    repeated_seeds = np.repeat(seeds, len(class_ids))
    slots = np.tile(np.arange(len(class_ids), dtype=np.int16), len(seeds))
    blur = np.empty((rows, len(core.CHECKPOINTS)), dtype=np.float64)
    for row in range(rows):
        blur[row] = 0.01 * repeated_seeds[row] + classes[row] + np.arange(
            len(core.CHECKPOINTS), dtype=np.float64
        )
    return {
        "sample_index": np.arange(rows, dtype=np.int32),
        "global_seed": repeated_seeds,
        "class_slot": slots,
        "class_id": classes,
        "selected_sampling_step": np.asarray(core.CHECKPOINTS, dtype=np.int16),
        "selected_internal_timestep": np.asarray(core.INTERNAL_TIMESTEPS, dtype=np.int16),
        "decoded_local_blur_severity": blur,
    }


def self_test() -> None:
    source = {
        "experiment": SOURCE_EXPERIMENT,
        "manifest_identity_sha256": "1" * 64,
        "manifest_file_sha256": "2" * 64,
        "time_series_file_sha256": "3" * 64,
        "unused_archive_members_not_loaded": ["resnet18_target_log_odds"],
    }
    arrays = _synthetic_arrays()
    payload = derive_calibration(arrays, source)
    validate_calibration(payload)
    first = payload["classes"][0]
    selected = arrays["class_id"] == first["class_id"]
    values = arrays["decoded_local_blur_severity"][selected]
    expected_gate = np.sort(values, axis=0)[STATE_GATE_ORDER_INDEX]
    expected_score = np.sort(np.mean(values, axis=1))[PURE_B_ORDER_INDEX]
    if not np.array_equal(first["blur_gate_threshold_by_checkpoint"], expected_gate):
        raise AssertionError("17th/20 B gate order statistic changed")
    if first["blur_score_threshold"] != expected_score:
        raise AssertionError("19th/20 pure-B order statistic changed")
    poisoned = dict(arrays)
    poisoned["human_label"] = np.zeros(len(values), dtype=np.int8)
    try:
        derive_calibration(poisoned, source)
    except RuntimeError:
        pass
    else:
        raise AssertionError("calibrator accepted a non-whitelisted array")
    print("self-test passed: label-free 17th/20 and 19th/20 B calibration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-product", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.self_test:
        self_test()
        return 0
    if args.source_product is None or args.output is None:
        raise RuntimeError("--source-product and --output are required outside --self-test")
    output = publish(
        args.source_product.expanduser().absolute(), args.output.expanduser().absolute()
    )
    print(f"frozen label-free B thresholds at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
