#!/usr/bin/env python3
"""Freeze label-free class thresholds for the two DiT third-pool candidates.

Only the previously reserved calibration seeds 30..49 are used.  Candidate B
uses a strict upper-tail rule and candidate C uses a strict lower-tail rule.
The script validates the complete input products and their source lineage, but
it never opens a visual label, review, image, screening result, or trajectory.
It emits thresholds and cryptographic receipts only--never individual scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .freeze_dit_bad_good_third_pool_protocol import (
        CALIBRATION_SEEDS,
        CLASSES,
        PRIMARY_FEATURE,
        VISUAL_FEATURE,
        canonical_sha256,
        load_json,
        require_real_directory,
        require_regular,
        sha256_file,
        validate_label_free_product,
        validate_protocol_lock,
        without_identity,
        write_json,
    )
except ImportError:  # pragma: no cover - direct CLI execution.
    from freeze_dit_bad_good_third_pool_protocol import (
        CALIBRATION_SEEDS,
        CLASSES,
        PRIMARY_FEATURE,
        VISUAL_FEATURE,
        canonical_sha256,
        load_json,
        require_real_directory,
        require_regular,
        sha256_file,
        validate_label_free_product,
        validate_protocol_lock,
        without_identity,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_third_pool_protocol_lock_v1"
)
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_bad_good_third_pool_threshold_lock_v1"
REFERENCE_PRODUCT_SEEDS = tuple(range(30, 130))
ALPHAS = (0.10, 0.05)
IDENTIFIER_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
)


def product_from_protocol(
    protocol: Mapping[str, Any], key: str, override: Path | None
) -> Path:
    lineage = protocol.get("input_lineage", {}).get(key)
    if not isinstance(lineage, dict) or not isinstance(lineage.get("path"), str):
        raise RuntimeError(f"protocol lacks input lineage: {key}")
    frozen = Path(lineage["path"]).expanduser().resolve()
    selected = frozen if override is None else override.expanduser().resolve()
    if selected != frozen:
        raise RuntimeError(
            f"input override differs from frozen protocol for {key}: {selected} != {frozen}"
        )
    return selected


def revalidate_product(
    root: Path,
    frozen_lineage: Mapping[str, Any],
    *,
    feature: str,
    expected_experiment: str,
    expected_summary_status: str,
    supervision_field: str,
) -> dict[str, Any]:
    observed = validate_label_free_product(
        root,
        feature=feature,
        expected_experiment=expected_experiment,
        expected_summary_status=expected_summary_status,
        supervision_field=supervision_field,
    )
    if observed != dict(frozen_lineage):
        raise RuntimeError(f"label-free product differs from frozen protocol: {root}")
    return observed


def _parse_int(row: Mapping[str, str], field: str, path: Path) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid integer {field} in {path}") from exc


def read_calibration_column(
    root: Path, feature: str
) -> tuple[dict[tuple[int, int, int], float], dict[tuple[int, int, int], int]]:
    """Use scores only for seeds 30..49 while auditing all row identifiers."""

    root = require_real_directory(root, "label-free product")
    path = require_regular(root / "sample_features.csv", "sample feature table")
    expected = {
        (seed, slot, class_id)
        for seed in REFERENCE_PRODUCT_SEEDS
        for slot, class_id in enumerate(CLASSES)
    }
    observed: set[tuple[int, int, int]] = set()
    sample_indices: dict[tuple[int, int, int], int] = {}
    calibration: dict[tuple[int, int, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {*IDENTIFIER_COLUMNS, feature}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"sample feature table lacks required columns: {path}")
        for row in reader:
            seed = _parse_int(row, "global_seed", path)
            slot = _parse_int(row, "class_slot", path)
            class_id = _parse_int(row, "class_id", path)
            key = (seed, slot, class_id)
            if key in observed:
                raise RuntimeError(f"duplicate sample key in {path}: {key}")
            observed.add(key)
            sample_indices[key] = _parse_int(row, "sample_index", path)
            _parse_int(row, "run_index", path)
            if seed in CALIBRATION_SEEDS:
                try:
                    value = float(row[feature])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"invalid calibration value for {feature}: {key}") from exc
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite calibration value for {feature}: {key}")
                calibration[key] = value
    if observed != expected:
        missing = len(expected - observed)
        extra = len(observed - expected)
        raise RuntimeError(
            f"sample cohort differs from seeds30..129 x three classes: missing={missing}, extra={extra}"
        )
    expected_calibration = {
        (seed, slot, class_id)
        for seed in CALIBRATION_SEEDS
        for slot, class_id in enumerate(CLASSES)
    }
    if set(calibration) != expected_calibration or len(calibration) != 60:
        raise RuntimeError("calibration subset is not exact seeds30..49 x three classes")
    return calibration, sample_indices


def upper_tail_threshold(values: np.ndarray, alpha: float) -> dict[str, Any]:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or len(values) != 20 or not np.isfinite(values).all():
        raise ValueError("upper-tail calibration requires exactly 20 finite values")
    order = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
    order = min(max(order, 1), len(values))
    numerator = len(values) + 1 - order
    return {
        "threshold": float(values[order - 1]),
        "calibration_count": int(len(values)),
        "calibration_order_statistic_1_based": order,
        "tail": "upper",
        "strict_comparison": "third_pool_raw_score > threshold",
        "finite_sample_marginal_trigger_probability_upper_bound": float(
            numerator / (len(values) + 1)
        ),
        "finite_sample_bound_fraction": f"{numerator}/{len(values) + 1}",
    }


def lower_tail_threshold(values: np.ndarray, alpha: float) -> dict[str, Any]:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or len(values) != 20 or not np.isfinite(values).all():
        raise ValueError("lower-tail calibration requires exactly 20 finite values")
    order = int(np.floor((len(values) + 1) * alpha))
    order = min(max(order, 1), len(values))
    numerator = order
    return {
        "threshold": float(values[order - 1]),
        "calibration_count": int(len(values)),
        "calibration_order_statistic_1_based": order,
        "tail": "lower",
        "strict_comparison": "third_pool_raw_score < threshold",
        "finite_sample_marginal_trigger_probability_upper_bound": float(
            numerator / (len(values) + 1)
        ),
        "finite_sample_bound_fraction": f"{numerator}/{len(values) + 1}",
    }


def alpha_key(alpha: float) -> str:
    return f"alpha_{alpha:.2f}".replace(".", "p")


def _calibration_hash(
    values: Mapping[tuple[int, int, int], float], class_id: int
) -> str:
    rows = [
        {
            "global_seed": seed,
            "class_slot": slot,
            "class_id": observed_class,
            "raw_score": float(values[(seed, slot, observed_class)]),
        }
        for seed in CALIBRATION_SEEDS
        for slot, observed_class in enumerate(CLASSES)
        if observed_class == class_id
    ]
    return canonical_sha256(rows)


def build_threshold_record(
    protocol: Mapping[str, Any],
    protocol_lock: Path,
    primary_lineage: Mapping[str, Any],
    visual_lineage: Mapping[str, Any],
    primary_values: Mapping[tuple[int, int, int], float],
    visual_values: Mapping[tuple[int, int, int], float],
) -> dict[str, Any]:
    thresholds: dict[str, Any] = {
        "B_blur_mean": {
            "feature": VISUAL_FEATURE,
            "raw_orientation": "bad_high",
            "classes": {},
        },
        "C_c3_low_jump": {
            "feature": PRIMARY_FEATURE,
            "raw_orientation": "bad_low",
            "classes": {},
        },
    }
    for class_id in CLASSES:
        slot = CLASSES.index(class_id)
        blur = np.asarray(
            [visual_values[(seed, slot, class_id)] for seed in CALIBRATION_SEEDS],
            dtype=np.float64,
        )
        c3 = np.asarray(
            [primary_values[(seed, slot, class_id)] for seed in CALIBRATION_SEEDS],
            dtype=np.float64,
        )
        thresholds["B_blur_mean"]["classes"][str(class_id)] = {
            "calibration_values_ordered_by_seed_sha256": _calibration_hash(
                visual_values, class_id
            ),
            **{
                alpha_key(alpha): upper_tail_threshold(blur, alpha)
                for alpha in ALPHAS
            },
        }
        thresholds["C_c3_low_jump"]["classes"][str(class_id)] = {
            "calibration_values_ordered_by_seed_sha256": _calibration_hash(
                primary_values, class_id
            ),
            **{
                alpha_key(alpha): lower_tail_threshold(c3, alpha)
                for alpha in ALPHAS
            },
        }

    protocol_manifest = protocol_lock / "manifest.json"
    protocol_completion = protocol_lock / "completion.json"
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_LABEL_FREE_BEFORE_THIRD_POOL_SAMPLING_OR_SCORE_LABEL_JOIN",
        "third_pool_protocol_identity_sha256": protocol["identity_sha256"],
        "third_pool_protocol_file_sha256": sha256_file(
            protocol_lock / "third_pool_protocol.json"
        ),
        "third_pool_protocol_manifest_file_sha256": sha256_file(protocol_manifest),
        "third_pool_protocol_manifest_identity_sha256": load_json(protocol_manifest)[
            "identity_sha256"
        ],
        "third_pool_protocol_completion_file_sha256": sha256_file(protocol_completion),
        "calibration": {
            "classes": list(CLASSES),
            "seeds": list(CALIBRATION_SEEDS),
            "count_per_candidate_class": len(CALIBRATION_SEEDS),
            "total_distinct_trajectory_rows": len(CLASSES) * len(CALIBRATION_SEEDS),
            "alphas": list(ALPHAS),
            "visual_labels_used": False,
            "review_or_endpoint_image_used": False,
            "third_pool_values_used": False,
            "screening_result_used": False,
            "candidate_selection_used_calibration_values": False,
        },
        "thresholds": thresholds,
        "input_lineage": {
            "primary_label_free_product": dict(primary_lineage),
            "visual_label_free_product": dict(visual_lineage),
        },
        "implementation_source_sha256": sha256_file(Path(__file__).resolve()),
        "guarantee_scope": (
            "Under within-class exchangeability of the 20 reserved calibration scores "
            "and one future score, strict alpha0.10 and alpha0.05 alerts have marginal "
            "trigger bounds 2/21 and 1/21. These are overall intervention budgets, not "
            "clean-good conditional false-positive rates."
        ),
        "access_audit": {
            "only_candidate_columns_used": [VISUAL_FEATURE, PRIMARY_FEATURE],
            "only_calibration_seed_scores_used": list(CALIBRATION_SEEDS),
            "non_calibration_score_values_used": False,
            "visual_label_columns_read": False,
            "visual_label_or_review_files_opened": False,
            "endpoint_images_opened": False,
            "trajectory_archives_opened": False,
            "screening_results_opened": False,
            "individual_scores_emitted": False,
        },
    }
    record["identity_sha256"] = canonical_sha256(record)
    return record


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name in {"manifest.json", "completion.json"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"threshold artifact must not be a symlink: {path}")
        records.append(
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def validate_threshold_lock(root: Path) -> dict[str, Any]:
    root = require_real_directory(root, "third-pool threshold lock")
    threshold_path = require_regular(root / "thresholds_locked.json", "threshold record")
    manifest_path = require_regular(root / "manifest.json", "threshold manifest")
    completion_path = require_regular(root / "completion.json", "threshold completion")
    record = load_json(threshold_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    record_identity = record.get("identity_sha256")
    manifest_identity = manifest.get("identity_sha256")
    if (
        not isinstance(record_identity, str)
        or canonical_sha256(without_identity(record)) != record_identity
        or not isinstance(manifest_identity, str)
        or canonical_sha256(without_identity(manifest)) != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("threshold_identity_sha256") != record_identity
        or manifest.get("third_pool_protocol_identity_sha256")
        != record.get("third_pool_protocol_identity_sha256")
        or manifest.get("files") != artifact_records(root)
        or completion.get("complete") is not True
        or completion.get("threshold_file_sha256") != sha256_file(threshold_path)
        or completion.get("threshold_identity_sha256") != record_identity
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
    ):
        raise RuntimeError(f"third-pool threshold lock validation failed: {root}")
    thresholds = record.get("thresholds", {})
    if set(thresholds) != {"B_blur_mean", "C_c3_low_jump"}:
        raise RuntimeError("threshold candidate family changed")
    for candidate, expected_tail, expected_orders in (
        ("B_blur_mean", "upper", {"alpha_0p10": 19, "alpha_0p05": 20}),
        ("C_c3_low_jump", "lower", {"alpha_0p10": 2, "alpha_0p05": 1}),
    ):
        classes = thresholds[candidate].get("classes", {})
        if set(classes) != {str(value) for value in CLASSES}:
            raise RuntimeError(f"threshold class set changed: {candidate}")
        for values in classes.values():
            for alpha, expected_order in expected_orders.items():
                row = values.get(alpha, {})
                if (
                    row.get("tail") != expected_tail
                    or row.get("calibration_count") != 20
                    or row.get("calibration_order_statistic_1_based") != expected_order
                    or not math.isfinite(row.get("threshold", math.nan))
                ):
                    raise RuntimeError(f"threshold contract changed: {candidate}/{alpha}")
    return record


def publish(
    protocol_lock: Path,
    primary_override: Path | None,
    visual_override: Path | None,
    output: Path,
) -> Path:
    protocol_lock = require_real_directory(protocol_lock, "third-pool protocol lock")
    protocol = validate_protocol_lock(protocol_lock)
    primary_root = product_from_protocol(
        protocol, "primary_label_free_product", primary_override
    )
    visual_root = product_from_protocol(protocol, "visual_label_free_product", visual_override)
    frozen_primary = protocol["input_lineage"]["primary_label_free_product"]
    frozen_visual = protocol["input_lineage"]["visual_label_free_product"]
    primary_lineage = revalidate_product(
        primary_root,
        frozen_primary,
        feature=PRIMARY_FEATURE,
        expected_experiment="dit_bad_good_custom_trace_metric_discovery",
        expected_summary_status="DISCOVERY_ONLY_NOT_AN_INTERVENTION_TRIGGER",
        supervision_field="labels_joined",
    )
    visual_lineage = revalidate_product(
        visual_root,
        frozen_visual,
        feature=VISUAL_FEATURE,
        expected_experiment="dit_predxstart_preterminal_visual_tracks_label_free",
        expected_summary_status="COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION",
        supervision_field="labels_read_or_emitted",
    )
    primary_values, primary_indices = read_calibration_column(primary_root, PRIMARY_FEATURE)
    visual_values, visual_indices = read_calibration_column(visual_root, VISUAL_FEATURE)
    if primary_indices != visual_indices:
        raise RuntimeError("primary and visual label-free products have different sample order")
    record = build_threshold_record(
        protocol,
        protocol_lock,
        primary_lineage,
        visual_lineage,
        primary_values,
        visual_values,
    )

    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite third-pool threshold lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "thresholds_locked.json", record)
        shutil.copy2(Path(__file__).resolve(), staging / "calibrator_source.py")
        members = artifact_records(staging)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "third_pool_protocol_identity_sha256": protocol["identity_sha256"],
            "threshold_identity_sha256": record["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "threshold_file_sha256": sha256_file(staging / "thresholds_locked.json"),
                "threshold_identity_sha256": record["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
            },
        )
        validate_threshold_lock(staging)
        os.replace(staging, output)
        validate_threshold_lock(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    values = np.arange(1.0, 21.0)
    high_010 = upper_tail_threshold(values, 0.10)
    high_005 = upper_tail_threshold(values, 0.05)
    low_010 = lower_tail_threshold(values, 0.10)
    low_005 = lower_tail_threshold(values, 0.05)
    assert high_010["calibration_order_statistic_1_based"] == 19
    assert high_010["threshold"] == 19.0
    assert high_010["finite_sample_bound_fraction"] == "2/21"
    assert high_005["calibration_order_statistic_1_based"] == 20
    assert high_005["threshold"] == 20.0
    assert high_005["finite_sample_bound_fraction"] == "1/21"
    assert low_010["calibration_order_statistic_1_based"] == 2
    assert low_010["threshold"] == 2.0
    assert low_010["finite_sample_bound_fraction"] == "2/21"
    assert low_005["calibration_order_statistic_1_based"] == 1
    assert low_005["threshold"] == 1.0
    assert low_005["finite_sample_bound_fraction"] == "1/21"
    assert alpha_key(0.10) == "alpha_0p10" and alpha_key(0.05) == "alpha_0p05"
    print(
        "self-test passed: exact n=20 upper/lower order statistics, strict tails, "
        "and 2/21 plus 1/21 marginal bounds"
    )


def compact_thresholds(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        candidate: {
            class_id: {
                alpha: values[alpha]["threshold"]
                for alpha in ("alpha_0p10", "alpha_0p05")
            }
            for class_id, values in candidate_record["classes"].items()
        }
        for candidate, candidate_record in record["thresholds"].items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, default=DEFAULT_PROTOCOL_LOCK)
    parser.add_argument("--primary-root", type=Path)
    parser.add_argument("--visual-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.validate is not None:
        record = validate_threshold_lock(args.validate)
        print(
            json.dumps(
                {
                    "output": str(args.validate.expanduser().absolute()),
                    "threshold_identity_sha256": record["identity_sha256"],
                    "thresholds": compact_thresholds(record),
                    "status": "valid",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    output = publish(
        args.protocol_lock, args.primary_root, args.visual_root, args.output
    )
    record = validate_threshold_lock(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "threshold_identity_sha256": record["identity_sha256"],
                "thresholds": compact_thresholds(record),
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
