#!/usr/bin/env python3
"""Apply the frozen label-free calibration lock to evaluation seeds 50..129."""

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
    from .score_dit_bad_good_frozen_candidates import (
        canonical_sha256,
        load_json,
        sha256_file,
        validate_lock,
    )
    from .calibrate_dit_bad_good_conformal_thresholds import validate_score_product
except ImportError:  # pragma: no cover - direct CLI execution.
    from score_dit_bad_good_frozen_candidates import (
        canonical_sha256,
        load_json,
        sha256_file,
        validate_lock,
    )
    from calibrate_dit_bad_good_conformal_thresholds import validate_score_product


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v4"
)
DEFAULT_CALIBRATION_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"
)
SCORER_HELPER_SOURCE = ROOT / "experiments/score_dit_bad_good_frozen_candidates.py"
CALIBRATOR_HELPER_SOURCE = ROOT / "experiments/calibrate_dit_bad_good_conformal_thresholds.py"


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_calibration_lock(
    root: Path, candidate_protocol: dict[str, Any], score_manifest: dict[str, Any]
) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"calibration lock must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    record_path = root / "calibration_locked.json"
    record = load_json(record_path)
    manifest_without_identity = dict(manifest)
    manifest_identity = manifest_without_identity.pop("identity_sha256", None)
    record_without_identity = dict(record)
    record_identity = record_without_identity.pop("identity_sha256", None)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or manifest_identity != canonical_sha256(manifest_without_identity)
        or completion.get("calibration_file_sha256") != sha256_file(record_path)
        or completion.get("calibration_identity_sha256") != record_identity
        or manifest.get("calibration_identity_sha256") != record_identity
        or record_identity != canonical_sha256(record_without_identity)
        or record.get("status") != "FROZEN_LABEL_FREE_BEFORE_INFERENTIAL_VISUAL_LABEL_JOIN"
        or record.get("visual_labels_read_or_emitted") is not False
        or record.get("candidate_protocol_identity_sha256")
        != candidate_protocol["identity_sha256"]
        or record.get("candidate_score_manifest_identity_sha256")
        != score_manifest["identity_sha256"]
    ):
        raise RuntimeError("calibration lock is incomplete, supervised, or bound to other scores")
    members = {item["name"]: item for item in manifest.get("files", [])}
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"calibration lock member changed: {path}")
    expected_source_hashes = {
        "calibrator": members.get("calibrator_source.py", {}).get("sha256"),
        "imported_scorer_helper": members.get("scorer_helper_source.py", {}).get("sha256"),
    }
    if record.get("implementation_source_sha256") != expected_source_hashes:
        raise RuntimeError("calibration implementation source hashes are not self-bound")

    calibration_path = root / "calibration_scores_locked.csv"
    calibration_rows = pd.read_csv(calibration_path)
    classes = tuple(candidate_protocol["fresh_confirmation"]["classes"])
    calibration_seeds = tuple(record.get("calibration_seeds", []))
    class_slots = {class_id: index for index, class_id in enumerate(classes)}
    expected_rows = {
        (seed, class_slots[class_id], class_id)
        for seed in calibration_seeds
        for class_id in classes
    }
    observed_rows = {
        (int(row.global_seed), int(row.class_slot), int(row.class_id))
        for row in calibration_rows[["global_seed", "class_slot", "class_id"]].itertuples(
            index=False
        )
    }
    if (
        len(calibration_rows) != record.get("calibration_sample_count")
        or observed_rows != expected_rows
        or calibration_rows[["global_seed", "class_id"]].duplicated().any()
        or not calibration_rows["cohort_role"].eq("label_free_calibration").all()
        or not np.isfinite(calibration_rows["S_UNION"]).all()
    ):
        raise RuntimeError("calibration rows do not reproduce the frozen calibration cohort")
    replayed: dict[str, Any] = {}
    for class_id in classes:
        scores = np.sort(
            calibration_rows.loc[
                calibration_rows["class_id"] == class_id, "S_UNION"
            ].to_numpy(
                dtype=np.float64
            )
        )
        class_record: dict[str, Any] = {}
        for alpha in (0.10, 0.05):
            order = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
            order = min(max(order, 1), len(scores))
            key = f"alpha_{alpha:.2f}".replace(".", "p")
            class_record[key] = {
                "threshold": float(scores[order - 1]),
                "calibration_count": int(len(scores)),
                "calibration_order_statistic_1_based": order,
                "strict_comparison": "evaluation_S_UNION > threshold",
                "finite_sample_marginal_trigger_probability_upper_bound": float(
                    (len(scores) + 1 - order) / (len(scores) + 1)
                ),
            }
        replayed[str(class_id)] = class_record
    if replayed != record.get("thresholds"):
        raise RuntimeError("calibration thresholds do not replay from locked calibration scores")
    return record


def publish(
    candidate_lock: Path,
    calibration_lock: Path,
    scores_root: Path,
    output: Path,
) -> Path:
    protocol = validate_lock(candidate_lock.resolve())
    score_manifest, scores = validate_score_product(scores_root.resolve(), protocol)
    calibration = validate_calibration_lock(
        calibration_lock.resolve(), protocol, score_manifest
    )
    evaluation = scores.loc[scores["cohort_role"] == "inferential_evaluation"].copy()
    expected = protocol["fresh_confirmation"]["inferential_evaluation"]
    expected_seeds = tuple(
        range(expected["seeds"]["start_inclusive"], expected["seeds"]["stop_inclusive"] + 1)
    )
    if (
        len(evaluation) != expected["trajectory_count"]
        or tuple(sorted(int(value) for value in evaluation["global_seed"].unique()))
        != expected_seeds
    ):
        raise RuntimeError("evaluation score rows differ from the frozen split")
    classes = tuple(protocol["fresh_confirmation"]["classes"])
    class_slots = {class_id: index for index, class_id in enumerate(classes)}
    expected_rows = {
        (seed, class_slots[class_id], class_id)
        for seed in expected_seeds
        for class_id in classes
    }
    observed_rows = {
        (int(row.global_seed), int(row.class_slot), int(row.class_id))
        for row in evaluation[["global_seed", "class_slot", "class_id"]].itertuples(index=False)
    }
    if (
        observed_rows != expected_rows
        or evaluation[["global_seed", "class_id"]].duplicated().any()
    ):
        raise RuntimeError("evaluation rows are not the exact frozen seed x class Cartesian product")
    evaluation["alert_alpha0p10_conformal"] = False
    evaluation["alert_alpha0p05_conformal"] = False
    covered = pd.Series(False, index=evaluation.index)
    for class_id in classes:
        mask = evaluation["class_id"] == class_id
        covered.loc[mask] = True
        thresholds = calibration["thresholds"][str(class_id)]
        evaluation.loc[mask, "alert_alpha0p10_conformal"] = (
            evaluation.loc[mask, "S_UNION"] > thresholds["alpha_0p10"]["threshold"]
        )
        evaluation.loc[mask, "alert_alpha0p05_conformal"] = (
            evaluation.loc[mask, "S_UNION"] > thresholds["alpha_0p05"]["threshold"]
        )
    if not covered.all():
        raise RuntimeError("some evaluation rows lack calibrated alert decisions")
    forbidden = {"label", "raw_consensus_label", "primary_label", "severity_score"}
    if forbidden & set(evaluation.columns):
        raise RuntimeError("evaluation alert table contains visual labels")

    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite calibrated alert output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        evaluation.to_csv(staging / "evaluation_scores_and_alerts_label_free.csv", index=False)
        summary = {
            "schema_version": 1,
            "status": "COMPLETE_LABEL_FREE_CALIBRATED_EVALUATION_ALERTS",
            "candidate_protocol_identity_sha256": protocol["identity_sha256"],
            "candidate_score_manifest_identity_sha256": score_manifest["identity_sha256"],
            "calibration_identity_sha256": calibration["identity_sha256"],
            "sample_count": int(len(evaluation)),
            "seeds": list(expected_seeds),
            "labels_read_or_emitted": False,
            "thresholds_reestimated": False,
            "alert_counts": {
                "alpha0p10": int(evaluation["alert_alpha0p10_conformal"].sum()),
                "alpha0p05": int(evaluation["alert_alpha0p05_conformal"].sum()),
            },
            "implementation_source_sha256": {
                "applicator": sha256_file(Path(__file__).resolve()),
                "imported_calibrator_helper": sha256_file(CALIBRATOR_HELPER_SOURCE),
                "imported_scorer_helper": sha256_file(SCORER_HELPER_SOURCE),
            },
        }
        write_json(staging / "summary.json", summary)
        shutil.copy2(Path(__file__).resolve(), staging / "applicator_source.py")
        shutil.copy2(CALIBRATOR_HELPER_SOURCE, staging / "calibrator_helper_source.py")
        shutil.copy2(SCORER_HELPER_SOURCE, staging / "scorer_helper_source.py")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-lock", type=Path, default=DEFAULT_CANDIDATE_LOCK)
    parser.add_argument("--calibration-lock", type=Path, default=DEFAULT_CALIBRATION_LOCK)
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = publish(
        args.candidate_lock, args.calibration_lock, args.scores_root, args.output
    )
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
