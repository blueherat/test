#!/usr/bin/env python3
"""Freeze split-conformal alert thresholds from fresh label-free seeds 30..49."""

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
except ImportError:  # pragma: no cover - direct CLI execution.
    from score_dit_bad_good_frozen_candidates import (
        canonical_sha256,
        load_json,
        sha256_file,
        validate_lock,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_LOCK = (
    ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
)
DEFAULT_OUTPUT = ROOT / "experiments/locks/dit_bad_good_conformal_calibration_lock_v1"
SCORER_HELPER_SOURCE = ROOT / "experiments/score_dit_bad_good_frozen_candidates.py"
ALPHAS = (0.10, 0.05)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_score_product(
    root: Path, candidate_protocol: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"score product must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    summary = load_json(root / "summary.json")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest.get("identity_sha256")
        or manifest.get("candidate_protocol_identity_sha256")
        != candidate_protocol["identity_sha256"]
        or summary.get("candidate_protocol_identity_sha256")
        != candidate_protocol["identity_sha256"]
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("alerts_emitted") is not False
    ):
        raise RuntimeError("frozen candidate score product is invalid or supervised")
    files = {item["name"]: item for item in manifest.get("files", [])}
    score_path = root / "frozen_candidate_scores_label_free.csv"
    item = files.get(score_path.name)
    if (
        item is None
        or not score_path.is_file()
        or score_path.is_symlink()
        or score_path.stat().st_size != item.get("bytes")
        or sha256_file(score_path) != item.get("sha256")
        or completion.get("scores_file_sha256") != item.get("sha256")
    ):
        raise RuntimeError("candidate score CSV is not hash-bound")
    frame = pd.read_csv(score_path)
    forbidden = {"label", "raw_consensus_label", "primary_label", "severity_score"}
    if forbidden & set(frame.columns):
        raise RuntimeError("calibration score product contains visual labels")
    required = {"sample_index", "global_seed", "class_id", "cohort_role", "S_UNION"}
    if not required.issubset(frame.columns) or not np.isfinite(frame["S_UNION"]).all():
        raise RuntimeError("calibration score product lacks finite frozen scores")
    return manifest, frame


def build_calibration(
    candidate_protocol: dict[str, Any], scores_manifest: dict[str, Any], frame: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    fresh = candidate_protocol["fresh_confirmation"]
    classes = tuple(fresh["classes"])
    calibration_spec = fresh["label_free_conformal_calibration"]["seeds"]
    evaluation_spec = fresh["inferential_evaluation"]["seeds"]
    calibration_seeds = tuple(
        range(calibration_spec["start_inclusive"], calibration_spec["stop_inclusive"] + 1)
    )
    evaluation_seeds = tuple(
        range(evaluation_spec["start_inclusive"], evaluation_spec["stop_inclusive"] + 1)
    )
    calibration = frame.loc[frame["global_seed"].isin(calibration_seeds)].copy()
    evaluation = frame.loc[frame["global_seed"].isin(evaluation_seeds)]
    class_slots = {class_id: index for index, class_id in enumerate(classes)}
    expected_rows = {
        (seed, class_slots[class_id], class_id)
        for seed in (*calibration_seeds, *evaluation_seeds)
        for class_id in classes
    }
    observed_rows = {
        (int(row.global_seed), int(row.class_slot), int(row.class_id))
        for row in frame[["global_seed", "class_slot", "class_id"]].itertuples(index=False)
    }
    if (
        len(calibration) != len(classes) * len(calibration_seeds)
        or len(evaluation) != len(classes) * len(evaluation_seeds)
        or not calibration["cohort_role"].eq("label_free_calibration").all()
        or not evaluation["cohort_role"].eq("inferential_evaluation").all()
        or set(calibration["global_seed"]) & set(evaluation["global_seed"])
        or observed_rows != expected_rows
        or frame[["global_seed", "class_id"]].duplicated().any()
    ):
        raise RuntimeError("calibration/evaluation split differs from the frozen candidate protocol")
    thresholds: dict[str, Any] = {}
    for class_id in classes:
        part = calibration.loc[calibration["class_id"] == class_id].sort_values(
            ["global_seed", "sample_index"]
        )
        if tuple(int(value) for value in part["global_seed"]) != calibration_seeds:
            raise RuntimeError(f"class {class_id} calibration seeds are incomplete")
        sorted_scores = np.sort(part["S_UNION"].to_numpy(dtype=np.float64))
        class_thresholds: dict[str, Any] = {}
        for alpha in ALPHAS:
            order = int(np.ceil((len(sorted_scores) + 1) * (1.0 - alpha)))
            order = min(max(order, 1), len(sorted_scores))
            key = f"alpha_{alpha:.2f}".replace(".", "p")
            class_thresholds[key] = {
                "threshold": float(sorted_scores[order - 1]),
                "calibration_count": int(len(sorted_scores)),
                "calibration_order_statistic_1_based": order,
                "strict_comparison": "evaluation_S_UNION > threshold",
                "finite_sample_marginal_trigger_probability_upper_bound": float(
                    (len(sorted_scores) + 1 - order) / (len(sorted_scores) + 1)
                ),
            }
        thresholds[str(class_id)] = class_thresholds
    keep = [
        "sample_index",
        "run_index",
        "global_seed",
        "class_slot",
        "class_id",
        "cohort_role",
        "A_posterior_logstd_concentration_jump",
        "B_withheld_channel_predx0_cusum",
        "z_A_low_is_bad",
        "z_B_high_is_bad",
        "S_INTERSECTION",
        "S_UNION",
    ]
    calibration = calibration[keep].sort_values(["global_seed", "class_slot"])
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_LABEL_FREE_BEFORE_INFERENTIAL_VISUAL_LABEL_JOIN",
        "candidate_protocol_identity_sha256": candidate_protocol["identity_sha256"],
        "candidate_score_manifest_identity_sha256": scores_manifest["identity_sha256"],
        "score_function_was_selected_on_calibration_data": False,
        "visual_labels_read_or_emitted": False,
        "calibration_classes": list(classes),
        "calibration_seeds": list(calibration_seeds),
        "evaluation_seeds_excluded": list(evaluation_seeds),
        "calibration_sample_count": int(len(calibration)),
        "thresholds": thresholds,
        "calibrated_score": "S_UNION",
        "implementation_source_sha256": {
            "calibrator": sha256_file(Path(__file__).resolve()),
            "imported_scorer_helper": sha256_file(SCORER_HELPER_SOURCE),
        },
        "guarantee_scope": (
            "Within each class, if the 20 calibration scores and a future evaluation score "
            "are exchangeable, strict exceedance is bounded by the recorded marginal rate. "
            "This is an overall trigger bound, not good-image conditional FPR."
        ),
    }
    record["identity_sha256"] = canonical_sha256(record)
    return record, calibration


def publish(candidate_lock: Path, scores_root: Path, output: Path) -> Path:
    candidate_protocol = validate_lock(candidate_lock.resolve())
    scores_manifest, frame = validate_score_product(scores_root.resolve(), candidate_protocol)
    record, calibration = build_calibration(candidate_protocol, scores_manifest, frame)
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite calibration lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "calibration_locked.json", record)
        calibration.to_csv(staging / "calibration_scores_locked.csv", index=False)
        shutil.copy2(Path(__file__).resolve(), staging / "calibrator_source.py")
        shutil.copy2(SCORER_HELPER_SOURCE, staging / "scorer_helper_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "candidate_protocol_identity_sha256": candidate_protocol["identity_sha256"],
            "calibration_identity_sha256": record["identity_sha256"],
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
                "calibration_file_sha256": sha256_file(staging / "calibration_locked.json"),
                "calibration_identity_sha256": record["identity_sha256"],
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
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = publish(args.candidate_lock, args.scores_root, args.output)
    print(json.dumps({"output": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
