#!/usr/bin/env python3
"""Apply the immutable seeds30..49 conformal thresholds to expansion scores.

Thresholds are replayed from the locked calibration rows and are never fit or
updated with expansion values.  This program is label-free and accepts only
the exact 360-row seeds130..249 expansion score product.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .dit_bad_good_expansion_contract import (
        CALIBRATION_SEEDS,
        CANDIDATE_LOCK,
        CLASSES,
        EXPANSION_SEEDS,
        PRIMARY_ALERT_005,
        PRIMARY_ALERT_010,
        canonical_sha256,
        load_json,
        require_canonical_identity,
        require_planned_path,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )
except ImportError:  # pragma: no cover
    from dit_bad_good_expansion_contract import (
        CALIBRATION_SEEDS,
        CANDIDATE_LOCK,
        CLASSES,
        EXPANSION_SEEDS,
        PRIMARY_ALERT_005,
        PRIMARY_ALERT_010,
        canonical_sha256,
        load_json,
        require_canonical_identity,
        require_planned_path,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"
)


def _validate_members(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    members = {str(item.get("name")): item for item in manifest.get("files", [])}
    if not members:
        raise RuntimeError(f"manifest has no members: {root}")
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"manifest-bound member changed: {path}")
    return members


def validate_calibration_lock(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"calibration lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    record_path = root / "calibration_locked.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    record = load_json(record_path)
    manifest_identity = require_canonical_identity(manifest, "calibration manifest")
    record_identity = require_canonical_identity(record, "calibration record")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("calibration_file_sha256") != sha256_file(record_path)
        or completion.get("calibration_identity_sha256") != record_identity
        or manifest.get("calibration_identity_sha256") != record_identity
        or record.get("status")
        != "FROZEN_LABEL_FREE_BEFORE_INFERENTIAL_VISUAL_LABEL_JOIN"
        or record.get("visual_labels_read_or_emitted") is not False
        or record.get("score_function_was_selected_on_calibration_data") is not False
        or record.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or record.get("calibrated_score") != "S_UNION"
        or tuple(record.get("calibration_seeds", ())) != CALIBRATION_SEEDS
    ):
        raise RuntimeError("calibration lock is invalid or changed")
    members = _validate_members(root, manifest)
    expected_source_hashes = {
        "calibrator": members.get("calibrator_source.py", {}).get("sha256"),
        "imported_scorer_helper": members.get("scorer_helper_source.py", {}).get(
            "sha256"
        ),
    }
    if record.get("implementation_source_sha256") != expected_source_hashes:
        raise RuntimeError("calibration implementation source hashes are not self-bound")

    calibration_path = root / "calibration_scores_locked.csv"
    rows = pd.read_csv(calibration_path)
    expected = {
        (seed, slot, class_id)
        for seed in CALIBRATION_SEEDS
        for slot, class_id in enumerate(CLASSES)
    }
    observed = {
        (int(row.global_seed), int(row.class_slot), int(row.class_id))
        for row in rows[["global_seed", "class_slot", "class_id"]].itertuples(index=False)
    }
    if (
        len(rows) != 60
        or observed != expected
        or rows[["global_seed", "class_id"]].duplicated().any()
        or not rows["cohort_role"].eq("label_free_calibration").all()
        or not np.isfinite(rows["S_UNION"].to_numpy(float)).all()
    ):
        raise RuntimeError("calibration rows do not replay the exact 3x20 cohort")
    replayed: dict[str, Any] = {}
    for class_id in CLASSES:
        scores = np.sort(
            rows.loc[rows["class_id"].eq(class_id), "S_UNION"].to_numpy(float)
        )
        class_thresholds: dict[str, Any] = {}
        for alpha in (0.10, 0.05):
            order = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
            order = min(max(order, 1), len(scores))
            key = f"alpha_{alpha:.2f}".replace(".", "p")
            class_thresholds[key] = {
                "threshold": float(scores[order - 1]),
                "calibration_count": int(len(scores)),
                "calibration_order_statistic_1_based": order,
                "strict_comparison": "evaluation_S_UNION > threshold",
                "finite_sample_marginal_trigger_probability_upper_bound": float(
                    (len(scores) + 1 - order) / (len(scores) + 1)
                ),
            }
        replayed[str(class_id)] = class_thresholds
    if replayed != record.get("thresholds"):
        raise RuntimeError("locked calibration thresholds do not replay")
    return record, manifest_identity


def validate_expansion_score_product(
    root: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"score product must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    summary = load_json(root / "summary.json")
    require_canonical_identity(manifest, "expansion score manifest")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or manifest.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or summary.get("status")
        != "COMPLETE_LABEL_FREE_FROZEN_CANDIDATE_EXPANSION_SCORES"
        or summary.get("candidate_protocol_identity_sha256")
        != protocol["identity_sha256"]
        or summary.get("sample_count") != 360
        or tuple(summary.get("seeds", ())) != EXPANSION_SEEDS
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("thresholds_read_or_reestimated") is not False
        or summary.get("alerts_read_or_emitted") is not False
        or summary.get("formula_changed") is not False
        or summary.get("normalizers_reestimated") is not False
        or summary.get("expansion_final_label_lock", {}).get(
            "row_level_labels_decoded"
        )
        is not False
    ):
        raise RuntimeError("expansion score product is invalid or supervised")
    members = _validate_members(root, manifest)
    score_path = root / "frozen_candidate_scores_label_free.csv"
    if completion.get("scores_file_sha256") != members.get(score_path.name, {}).get(
        "sha256"
    ):
        raise RuntimeError("expansion score CSV completion binding failed")
    frame = pd.read_csv(score_path)
    forbidden_tokens = ("label", "review", "consensus", "severity", "adjudic")
    if any(any(token in name.lower() for token in forbidden_tokens) for name in frame):
        raise RuntimeError("expansion score CSV contains visual-supervision columns")
    required = {
        "sample_index",
        "run_index",
        "global_seed",
        "class_slot",
        "class_id",
        "trace_dir",
        "endpoint_png_path",
        "cohort_role",
        "A_posterior_logstd_concentration_jump",
        "B_withheld_channel_predx0_cusum",
        "z_A_low_is_bad",
        "z_B_high_is_bad",
        "S_INTERSECTION",
        "S_UNION",
    }
    if not required.issubset(frame.columns):
        raise RuntimeError("expansion score table lacks frozen columns")
    expected = {(seed, class_id) for seed in EXPANSION_SEEDS for class_id in CLASSES}
    observed = {
        (int(row.global_seed), int(row.class_id))
        for row in frame[["global_seed", "class_id"]].itertuples(index=False)
    }
    if (
        len(frame) != 360
        or observed != expected
        or frame[["global_seed", "class_id"]].duplicated().any()
        or not frame["cohort_role"].eq("inferential_expansion").all()
        or not np.isfinite(
            frame[
                [
                    "A_posterior_logstd_concentration_jump",
                    "B_withheld_channel_predx0_cusum",
                    "z_A_low_is_bad",
                    "z_B_high_is_bad",
                    "S_INTERSECTION",
                    "S_UNION",
                ]
            ].to_numpy(float)
        ).all()
    ):
        raise RuntimeError("expansion score rows are incomplete or invalid")
    return manifest, frame


def publish(
    candidate_lock: Path,
    calibration_lock: Path,
    scores_root: Path,
    output: Path,
) -> Path:
    protocol = validate_candidate_lock(candidate_lock)
    expansion_protocol = validate_expansion_lock()
    pipeline = validate_pipeline_source_lock(Path(__file__).name)
    require_planned_path(pipeline, "candidate_v5_lock", candidate_lock)
    require_planned_path(pipeline, "calibration_lock", calibration_lock)
    require_planned_path(pipeline, "label_free_candidate_scores", scores_root)
    require_planned_path(pipeline, "label_free_calibrated_alerts", output)
    score_manifest, scores = validate_expansion_score_product(scores_root, protocol)
    calibration, calibration_manifest_identity = validate_calibration_lock(
        calibration_lock, protocol
    )
    evaluation = scores.copy()
    evaluation[PRIMARY_ALERT_010] = False
    evaluation[PRIMARY_ALERT_005] = False
    for class_id in CLASSES:
        mask = evaluation["class_id"].eq(class_id)
        thresholds = calibration["thresholds"][str(class_id)]
        evaluation.loc[mask, PRIMARY_ALERT_010] = (
            evaluation.loc[mask, "S_UNION"]
            > float(thresholds["alpha_0p10"]["threshold"])
        )
        evaluation.loc[mask, PRIMARY_ALERT_005] = (
            evaluation.loc[mask, "S_UNION"]
            > float(thresholds["alpha_0p05"]["threshold"])
        )

    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite expansion alert product: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        evaluation.to_csv(
            staging / "evaluation_scores_and_alerts_label_free.csv", index=False
        )
        summary = {
            "schema_version": 1,
            "status": "COMPLETE_LABEL_FREE_CALIBRATED_EXPANSION_ALERTS",
            "candidate_protocol_identity_sha256": protocol["identity_sha256"],
            "expansion_protocol_identity_sha256": expansion_protocol[
                "identity_sha256"
            ],
            "candidate_score_manifest_identity_sha256": score_manifest[
                "identity_sha256"
            ],
            "calibration_identity_sha256": calibration["identity_sha256"],
            "calibration_manifest_identity_sha256": calibration_manifest_identity,
            "sample_count": 360,
            "classes": list(CLASSES),
            "seeds": list(EXPANSION_SEEDS),
            "labels_read_or_emitted": False,
            "thresholds_reestimated": False,
            "calibration_rows_reused_as_threshold_source_only": True,
            "formula_or_normalizer_changed": False,
            "alert_counts": {
                "alpha0p10": int(evaluation[PRIMARY_ALERT_010].sum()),
                "alpha0p05": int(evaluation[PRIMARY_ALERT_005].sum()),
            },
        }
        write_json(staging / "summary.json", summary)
        shutil.copy2(Path(__file__).resolve(), staging / "applicator_source.py")
        helper = Path(__file__).resolve().with_name("dit_bad_good_expansion_contract.py")
        shutil.copy2(helper, staging / "expansion_contract_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": protocol["identity_sha256"],
            "calibration_identity_sha256": calibration["identity_sha256"],
            "files": members,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
                "manifest_identity_sha256": manifest["identity_sha256"],
                "alerts_file_sha256": sha256_file(
                    staging / "evaluation_scores_and_alerts_label_free.csv"
                ),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    protocol = validate_candidate_lock(CANDIDATE_LOCK)
    validate_expansion_lock()
    assert protocol["normalization"]["reference_seeds"] == list(range(10, 30))
    values = np.asarray([0.0, 1.0, 2.0])
    assert np.array_equal(values > 1.0, np.asarray([False, False, True]))
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, default=CANDIDATE_LOCK)
    parser.add_argument("--calibration-lock", type=Path, default=DEFAULT_CALIBRATION_LOCK)
    parser.add_argument("--scores-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.scores_root is None or args.output is None:
        parser.error("--scores-root and --output are required")
    output = publish(
        args.candidate_lock, args.calibration_lock, args.scores_root, args.output
    )
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
