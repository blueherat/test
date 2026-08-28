#!/usr/bin/env python3
"""Lock three expansion reviews and conservative visual-only adjudication.

The ``reviews`` action creates the raw 2-of-3 consensus.  The ``adjudicate``
action permits only retaining or downgrading raw majority clear-bad examples;
it cannot promote any sample.  Neither action accepts or opens trajectory
metrics, candidate scores, calibration thresholds, alerts, or other cohorts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from .build_dit_bad_good_expansion_blind_review_pack import validate_completed
    from .dit_bad_good_expansion_contract import (
        ALLOWED_FLAGS,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        EXPANSION_SEEDS,
        LABEL_BAD,
        LABEL_EXCLUDED,
        LABEL_GOOD,
        REVIEWERS,
        canonical_sha256,
        expected_keys,
        load_json,
        require_canonical_identity,
        require_planned_path,
        sample_key,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )
except ImportError:  # pragma: no cover
    from build_dit_bad_good_expansion_blind_review_pack import validate_completed
    from dit_bad_good_expansion_contract import (
        ALLOWED_FLAGS,
        CANDIDATE_PROTOCOL_IDENTITY,
        CLASSES,
        EXPANSION_SEEDS,
        LABEL_BAD,
        LABEL_EXCLUDED,
        LABEL_GOOD,
        REVIEWERS,
        canonical_sha256,
        expected_keys,
        load_json,
        require_canonical_identity,
        require_planned_path,
        sample_key,
        sha256_file,
        validate_candidate_lock,
        validate_expansion_lock,
        validate_pipeline_source_lock,
        write_json,
    )


RAW_STATUS = "EXPANSION_LABELS_LOCKED_WITHOUT_SCORE_OR_ALERT_ACCESS"
FINAL_STATUS = "FINAL_EXPANSION_VISUAL_LABELS_LOCKED_BEFORE_ANY_SCORE_JOIN"
ALLOWED_DECISIONS = {"retain_clear_bad", "downgrade_to_mild"}


def label_from_scores(scores: list[int]) -> str:
    if sum(score >= 2 for score in scores) >= 2:
        return LABEL_BAD
    if sum(score == 0 for score in scores) >= 2:
        return LABEL_GOOD
    return LABEL_EXCLUDED


def _pack_endpoints(root: Path) -> tuple[str, dict[str, dict[str, Any]], str]:
    identity = validate_completed(root)
    blind_identity = str(identity["identity_sha256"])
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    rows = manifest.get("source_endpoints")
    if not isinstance(rows, list) or len(rows) != 360:
        raise RuntimeError("expansion blind pack lacks exact endpoint rows")
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        class_id = row.get("class_id")
        seed = row.get("seed")
        if type(class_id) is not int or type(seed) is not int:
            raise RuntimeError("blind-pack endpoint key is malformed")
        key = sample_key(class_id, seed)
        copy = root / str(row.get("copied_endpoint_relative_path"))
        if (
            key in by_key
            or class_id not in CLASSES
            or seed not in EXPANSION_SEEDS
            or not copy.is_file()
            or copy.is_symlink()
            or copy.stat().st_size != row.get("source_endpoint_bytes")
            or sha256_file(copy) != row.get("source_endpoint_sha256")
            or row.get("source_endpoint_size") != [256, 256]
            or row.get("source_endpoint_mode") != "RGB"
        ):
            raise RuntimeError(f"blind-pack endpoint binding failed: {key}")
        by_key[key] = row
    if set(by_key) != expected_keys():
        raise RuntimeError("blind pack is not exact expansion Cartesian product")
    return blind_identity, by_key, sha256_file(manifest_path)


def _parse_reviews(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--review must be REVIEWER=PATH")
        reviewer, raw_path = value.split("=", 1)
        if reviewer in parsed:
            raise argparse.ArgumentTypeError(f"duplicate reviewer: {reviewer}")
        parsed[reviewer] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(REVIEWERS):
        raise argparse.ArgumentTypeError(
            f"reviews must be exactly {','.join(REVIEWERS)}"
        )
    return parsed


def _validate_reviews(
    paths: dict[str, Path], blind_identity: str
) -> dict[str, dict[str, dict[str, Any]]]:
    expected = expected_keys()
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer in REVIEWERS:
        path = paths[reviewer]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"review is missing or indirect: {path}")
        document = load_json(path)
        if (
            document.get("reviewer") != reviewer
            or document.get("independent_review") is not True
            or document.get("metrics_seen") is not False
            or document.get("candidate_scores_seen", False) is not False
            or document.get("calibration_thresholds_seen", False) is not False
            or document.get("alerts_seen", False) is not False
            or document.get("trajectories_seen") is not False
            or document.get("signals_summaries_or_research_hypotheses_seen") is not False
            or document.get("other_reviews_seen") is not False
            or document.get("single_reviewer_draft") is not True
            or document.get("blind_pack_identity_sha256") != blind_identity
        ):
            raise RuntimeError(f"review {reviewer} has invalid blinding declaration")
        annotations = document.get("annotations")
        if not isinstance(annotations, dict) or set(annotations) != expected:
            raise RuntimeError(f"review {reviewer} does not cover exact 360 samples")
        for key, row in annotations.items():
            class_id = int(key[5:9])
            seed = int(key[-3:])
            flags = row.get("flags")
            if (
                row.get("class_id") != class_id
                or row.get("seed") != seed
                or type(row.get("score")) is not int
                or row["score"] not in range(4)
                or not isinstance(flags, list)
                or not flags
                or len(flags) != len(set(flags))
                or not set(flags).issubset(ALLOWED_FLAGS)
                or ("none" in flags and flags != ["none"])
                or not isinstance(row.get("reason"), str)
                or not row["reason"].strip()
            ):
                raise RuntimeError(f"invalid review annotation: {reviewer}/{key}")
        result[reviewer] = annotations
    return result


def _raw_consensus(
    reviews: dict[str, dict[str, dict[str, Any]]],
    endpoints: dict[str, dict[str, Any]],
    blind_identity: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for class_id in CLASSES:
        for seed in EXPANSION_SEEDS:
            key = sample_key(class_id, seed)
            scores = {reviewer: reviews[reviewer][key]["score"] for reviewer in REVIEWERS}
            flags = {reviewer: reviews[reviewer][key]["flags"] for reviewer in REVIEWERS}
            reasons = {reviewer: reviews[reviewer][key]["reason"] for reviewer in REVIEWERS}
            vote_counts = {
                flag: sum(flag in flags[reviewer] for reviewer in REVIEWERS)
                for flag in sorted(ALLOWED_FLAGS - {"none"})
            }
            label = label_from_scores(list(scores.values()))
            endpoint = endpoints[key]
            rows.append(
                {
                    "sample_key": key,
                    "class_id": class_id,
                    "seed": seed,
                    "global_seed": seed,
                    "review_scores": scores,
                    "review_flags": flags,
                    "review_reasons": reasons,
                    "flag_vote_counts": vote_counts,
                    "majority_flags": [
                        flag for flag, count in vote_counts.items() if count >= 2
                    ],
                    "clear_bad_vote_count": sum(value >= 2 for value in scores.values()),
                    "clean_good_vote_count": sum(value == 0 for value in scores.values()),
                    "primary_label": label,
                    "binary_primary_included": label in {LABEL_BAD, LABEL_GOOD},
                    "native_image": {
                        "path": endpoint["source_endpoint_path"],
                        "file_sha256": endpoint["source_endpoint_sha256"],
                        "pixel_sha256": endpoint["source_endpoint_pixel_sha256"],
                        "mode": endpoint["source_endpoint_mode"],
                        "size": endpoint["source_endpoint_size"],
                        "blind_copy_relative_path": endpoint[
                            "copied_endpoint_relative_path"
                        ],
                    },
                }
            )
    counts = {
        label: sum(row["primary_label"] == label for row in rows)
        for label in (LABEL_BAD, LABEL_GOOD, LABEL_EXCLUDED)
    }
    consensus: dict[str, Any] = {
        "schema_version": 1,
        "status": RAW_STATUS,
        "cohort": "expansion_seed130_249",
        "blind_pack_identity_sha256": blind_identity,
        "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
        "blinding_audit": {
            "reviewer_count": 3,
            "reviewers": list(REVIEWERS),
            "endpoint_only_review": True,
            "metric_values_visible_to_reviewers": False,
            "candidate_scores_visible_to_reviewers": False,
            "calibration_thresholds_visible_to_reviewers": False,
            "alert_decisions_visible_to_reviewers": False,
            "trajectories_visible_to_reviewers": False,
            "other_reviews_visible_to_each_reviewer": False,
            "labels_locked_before_score_join": True,
        },
        "rule": {
            "clear_bad": "at least two of three independent scores are 2 or 3",
            "clean_good": "at least two of three independent scores are 0",
            "mild_or_disputed": "neither majority",
            "metric_or_signal_used": False,
        },
        "counts": counts,
        "rows": rows,
    }
    consensus["identity_sha256"] = canonical_sha256(consensus)
    return consensus


def _publish_lock(
    output: Path,
    consensus: dict[str, Any],
    extra_sources: dict[str, Path],
    manifest_extra: dict[str, Any],
) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite label lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "consensus_locked.json", consensus)
        for name, path in extra_sources.items():
            shutil.copy2(path, staging / name)
        shutil.copy2(Path(__file__).resolve(), staging / "locker_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "consensus_identity_sha256": consensus["identity_sha256"],
            "blind_pack_identity_sha256": consensus["blind_pack_identity_sha256"],
            "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
            "counts": consensus["counts"],
            **manifest_extra,
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
                "consensus_file_sha256": sha256_file(
                    staging / "consensus_locked.json"
                ),
                "consensus_identity_sha256": consensus["identity_sha256"],
                "locked_row_count": 360,
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def publish_reviews(
    blind_pack: Path, review_paths: dict[str, Path], output: Path
) -> Path:
    validate_candidate_lock()
    validate_expansion_lock()
    pipeline = validate_pipeline_source_lock(Path(__file__).name)
    require_planned_path(pipeline, "endpoint_only_blind_pack", blind_pack)
    require_planned_path(pipeline, "raw_consensus_lock", output)
    for reviewer, path in review_paths.items():
        require_planned_path(pipeline, "review_drafts", path, reviewer=reviewer)
    blind_identity, endpoints, blind_manifest_hash = _pack_endpoints(
        blind_pack.expanduser().resolve()
    )
    reviews = _validate_reviews(review_paths, blind_identity)
    consensus = _raw_consensus(reviews, endpoints, blind_identity)
    return _publish_lock(
        output,
        consensus,
        {
            f"review_{reviewer}_locked.json": review_paths[reviewer]
            for reviewer in REVIEWERS
        },
        {"blind_pack_manifest_sha256": blind_manifest_hash},
    )


def _validate_raw_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"raw label lock is missing or indirect: {root}")
    manifest_path = root / "manifest.json"
    consensus_path = root / "consensus_locked.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    consensus = load_json(consensus_path)
    manifest_identity = require_canonical_identity(manifest, "raw label manifest")
    consensus_identity = require_canonical_identity(consensus, "raw consensus")
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or completion.get("consensus_file_sha256") != sha256_file(consensus_path)
        or completion.get("consensus_identity_sha256") != consensus_identity
        or completion.get("locked_row_count") != 360
        or manifest.get("consensus_identity_sha256") != consensus_identity
        or consensus.get("status") != RAW_STATUS
        or consensus.get("candidate_protocol_identity_sha256")
        != CANDIDATE_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("raw expansion consensus lock is invalid")
    members = {str(item.get("name")): item for item in manifest.get("files", [])}
    for name, item in members.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
        ):
            raise RuntimeError(f"raw consensus member changed: {path}")
    rows = consensus.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        raise RuntimeError("raw expansion consensus lacks 360 rows")
    keys = [row.get("sample_key") for row in rows]
    if set(keys) != expected_keys() or len(keys) != len(set(keys)):
        raise RuntimeError("raw expansion consensus rows are not exact")
    return manifest, consensus


def _validate_adjudication(
    path: Path, raw_bad_keys: set[str], blind_identity: str
) -> dict[str, dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"adjudication draft is missing or indirect: {path}")
    document = load_json(path)
    if (
        document.get("visual_only_adjudication") is not True
        or document.get("metrics_seen") is not False
        or document.get("candidate_scores_seen") is not False
        or document.get("calibration_thresholds_seen") is not False
        or document.get("alert_decisions_seen") is not False
        or document.get("trajectories_seen") is not False
        or document.get("other_samples_promoted") is not False
        or document.get("blind_pack_identity_sha256") != blind_identity
        or document.get("adjudication_scope") != "raw_majority_clear_bad_only"
    ):
        raise RuntimeError("adjudication blinding declaration is invalid")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict) or set(decisions) != raw_bad_keys:
        raise RuntimeError("adjudication must cover exactly raw clear-bad keys")
    for key, row in decisions.items():
        if (
            not isinstance(row, dict)
            or row.get("decision") not in ALLOWED_DECISIONS
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
        ):
            raise RuntimeError(f"invalid adjudication decision: {key}")
    return decisions


def _final_consensus(
    raw: dict[str, Any], decisions: dict[str, dict[str, str]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in raw["rows"]:
        row = dict(source)
        raw_label = str(row["primary_label"])
        row["raw_primary_label"] = raw_label
        if raw_label == LABEL_BAD:
            decision = decisions[row["sample_key"]]
            row["adjudication"] = decision
            row["primary_label"] = (
                LABEL_BAD
                if decision["decision"] == "retain_clear_bad"
                else LABEL_EXCLUDED
            )
        else:
            row["adjudication"] = None
        row["binary_primary_included"] = row["primary_label"] in {
            LABEL_BAD,
            LABEL_GOOD,
        }
        rows.append(row)
    counts = {
        label: sum(row["primary_label"] == label for row in rows)
        for label in (LABEL_BAD, LABEL_GOOD, LABEL_EXCLUDED)
    }
    final: dict[str, Any] = {
        "schema_version": 1,
        "status": FINAL_STATUS,
        "cohort": "expansion_seed130_249",
        "blind_pack_identity_sha256": raw["blind_pack_identity_sha256"],
        "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
        "raw_consensus_identity_sha256": raw["identity_sha256"],
        "blinding_audit": {
            **raw["blinding_audit"],
            "adjudicator_saw_metric_values": False,
            "adjudicator_saw_candidate_scores": False,
            "adjudicator_saw_calibration_thresholds": False,
            "adjudicator_saw_alert_decisions": False,
            "adjudicator_saw_trajectories": False,
            "adjudication_could_only_retain_or_downgrade_raw_clear_bad": True,
            "labels_locked_before_score_join": True,
        },
        "adjudication_rule": {
            "scope": "raw majority clear-bad rows only",
            "promotion_allowed": False,
        },
        "raw_clear_bad_count": sum(
            row["raw_primary_label"] == LABEL_BAD for row in rows
        ),
        "retained_clear_bad_count": counts[LABEL_BAD],
        "counts": counts,
        "rows": rows,
    }
    final["identity_sha256"] = canonical_sha256(final)
    return final


def publish_adjudication(raw_lock: Path, adjudication: Path, output: Path) -> Path:
    validate_candidate_lock()
    validate_expansion_lock()
    pipeline = validate_pipeline_source_lock(Path(__file__).name)
    require_planned_path(pipeline, "raw_consensus_lock", raw_lock)
    require_planned_path(pipeline, "adjudication_draft", adjudication)
    require_planned_path(pipeline, "final_consensus_lock", output)
    raw_manifest, raw = _validate_raw_lock(raw_lock)
    raw_bad = {
        row["sample_key"] for row in raw["rows"] if row["primary_label"] == LABEL_BAD
    }
    decisions = _validate_adjudication(
        adjudication.expanduser().resolve(),
        raw_bad,
        raw["blind_pack_identity_sha256"],
    )
    final = _final_consensus(raw, decisions)
    return _publish_lock(
        output,
        final,
        {"adjudication_locked.json": adjudication.expanduser().resolve()},
        {
            "raw_consensus_identity_sha256": raw["identity_sha256"],
            "raw_consensus_manifest_identity_sha256": raw_manifest["identity_sha256"],
            "raw_consensus_manifest_file_sha256": sha256_file(
                raw_lock.expanduser().resolve() / "manifest.json"
            ),
        },
    )


def self_test() -> None:
    assert label_from_scores([2, 2, 0]) == LABEL_BAD
    assert label_from_scores([0, 0, 3]) == LABEL_GOOD
    assert label_from_scores([0, 1, 2]) == LABEL_EXCLUDED
    assert len(expected_keys()) == 360
    raw = {
        "identity_sha256": "raw",
        "blind_pack_identity_sha256": "pack",
        "blinding_audit": {},
        "rows": [
            {"sample_key": "a", "primary_label": LABEL_BAD},
            {"sample_key": "b", "primary_label": LABEL_GOOD},
        ],
    }
    final = _final_consensus(
        raw, {"a": {"decision": "downgrade_to_mild", "reason": "ambiguous"}}
    )
    assert final["counts"] == {
        LABEL_BAD: 0,
        LABEL_GOOD: 1,
        LABEL_EXCLUDED: 1,
    }
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="action")
    reviews = sub.add_parser("reviews")
    reviews.add_argument("--blind-pack", type=Path, required=True)
    reviews.add_argument("--review", action="append", default=[])
    reviews.add_argument("--output", type=Path, required=True)
    adjudicate = sub.add_parser("adjudicate")
    adjudicate.add_argument("--raw-lock", type=Path, required=True)
    adjudicate.add_argument("--adjudication", type=Path, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.action == "reviews":
        output = publish_reviews(
            args.blind_pack, _parse_reviews(args.review), args.output
        )
    elif args.action == "adjudicate":
        output = publish_adjudication(
            args.raw_lock, args.adjudication, args.output
        )
    else:
        parser.error("select reviews or adjudicate")
    consensus = load_json(output / "consensus_locked.json")
    print(json.dumps({"output": str(output), "counts": consensus["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
