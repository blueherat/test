#!/usr/bin/env python3
"""Unseal and analyze the paired score-blind v2.2 alarm audit.

This analysis is valid only after both complete reviewer CSVs exist.  It keeps
the two reviewer readouts separate, reports a strict 2/2 clear-bad consensus,
and uses an exact one-sided paired sign/McNemar calculation.  The reviewers in
this exploratory audit are independent blind model-agent runs, not independent
human experts; no result here can authorize rollback.
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
from typing import Any, Iterable, Mapping

try:
    from .explore_dit_v22_third_pool_retrospective import (
        canonical_sha256,
        load_json,
        sha256_file,
        write_json,
    )
    from .prepare_dit_v22_alarm_blind_audit import RESPONSE_COLUMNS
except ImportError:  # pragma: no cover
    from explore_dit_v22_third_pool_retrospective import (
        canonical_sha256,
        load_json,
        sha256_file,
        write_json,
    )
    from prepare_dit_v22_alarm_blind_audit import RESPONSE_COLUMNS


QUALITY_CODE = {"clean_good": 0, "mild_or_uncertain": 1, "clear_bad": 2}
BOOLEAN = {"true": True, "false": False}


def validate_delivery(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = load_json(root / "manifest.json")
    identity = manifest.get("identity_sha256")
    payload = dict(manifest)
    payload.pop("identity_sha256", None)
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_kind") != "DIT_V22_ALARM_SCORE_BLIND_DELIVERY"
        or manifest.get("anonymous_count") != 52
        or manifest.get("score_arm_mapping_present") is not False
        or manifest.get("old_labels_or_external_representations_present") is not False
        or canonical_sha256(payload) != identity
    ):
        raise RuntimeError("blind delivery identity changed")
    records = {row["name"]: row for row in manifest.get("files", [])}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != observed:
        raise RuntimeError("blind delivery exact tree changed")
    for name, record in records.items():
        path = root / name
        if path.is_symlink() or record.get("bytes") != path.stat().st_size or record.get(
            "sha256"
        ) != sha256_file(path):
            raise RuntimeError(f"blind delivery member changed: {name}")
    return manifest


def validate_mapping(path: Path, delivery_identity: str) -> dict[str, Any]:
    payload = load_json(path.expanduser().resolve())
    identity = payload.get("identity_sha256")
    without = dict(payload)
    without.pop("identity_sha256", None)
    rows = payload.get("mapping")
    if (
        payload.get("status") != "SEALED_UNTIL_BOTH_REVIEWS_COMPLETE"
        or payload.get("artifact_kind") != "DIT_V22_ALARM_PRIVATE_MAPPING"
        or payload.get("delivery_identity_sha256") != delivery_identity
        or payload.get("alarm_count") != 26
        or payload.get("control_count") != 26
        or payload.get("exact_schedule_matched_pair_count") != 26
        or not isinstance(rows, list)
        or len(rows) != 52
        or canonical_sha256(without) != identity
    ):
        raise RuntimeError("private mapping identity changed")
    ids = [row.get("anonymous_id") for row in rows if isinstance(row, dict)]
    if len(ids) != 52 or len(set(ids)) != 52:
        raise RuntimeError("private mapping anonymous axis changed")
    return payload


def read_response(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular reviewer response: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESPONSE_COLUMNS:
            raise RuntimeError(f"review response columns changed: {path}")
        raw = list(reader)
    if len(raw) != len(expected_ids):
        raise RuntimeError(f"review response row count changed: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in raw:
        anonymous_id = row["anonymous_id"]
        if anonymous_id in result or anonymous_id not in expected_ids:
            raise RuntimeError(f"review response ID invalid: {anonymous_id}")
        quality = row["relative_quality"]
        if quality not in QUALITY_CODE:
            raise RuntimeError(f"invalid relative quality: {quality}")
        try:
            blur = BOOLEAN[row["blur_or_soft_fusion"].lower()]
            topology = BOOLEAN[row["topology_or_attachment_error"].lower()]
        except KeyError as exc:
            raise RuntimeError("review boolean must be true/false") from exc
        localized = row["localized_problem"].strip()
        reason = row["short_reason"].strip()
        if not reason:
            raise RuntimeError(f"review reason is empty: {anonymous_id}")
        if quality == "clean_good" and (blur or topology):
            raise RuntimeError(f"clean_good row carries a hard defect flag: {anonymous_id}")
        if quality != "clean_good" and not localized:
            raise RuntimeError(f"non-clean row lacks localized problem: {anonymous_id}")
        result[anonymous_id] = {
            "relative_quality": quality,
            "quality_code": QUALITY_CODE[quality],
            "clear_bad": quality == "clear_bad",
            "blur_or_soft_fusion": blur,
            "topology_or_attachment_error": topology,
            "localized_problem": localized,
            "short_reason": reason,
        }
    if set(result) != expected_ids:
        raise RuntimeError("review response anonymous axis is incomplete")
    return result


def one_sided_binomial_tail(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def paired_binary(
    joined: list[dict[str, Any]], field: str, reviewer: str
) -> dict[str, Any]:
    pairs: dict[int, dict[str, bool]] = {}
    for row in joined:
        pairs.setdefault(row["pair_index"], {})[row["arm"]] = bool(
            row[reviewer][field]
        )
    alarm_only = 0
    control_only = 0
    both = 0
    neither = 0
    for pair in pairs.values():
        if set(pair) != {"alarm", "control"}:
            raise RuntimeError("paired audit arm is incomplete")
        state = (pair["alarm"], pair["control"])
        alarm_only += state == (True, False)
        control_only += state == (False, True)
        both += state == (True, True)
        neither += state == (False, False)
    discordant = alarm_only + control_only
    return {
        "alarm_only": alarm_only,
        "control_only": control_only,
        "both": both,
        "neither": neither,
        "paired_rate_difference": (alarm_only - control_only) / len(pairs),
        "one_sided_exact_p_alarm_greater": one_sided_binomial_tail(
            alarm_only, discordant
        ),
        "discordant_pairs": discordant,
    }


def arm_summary(
    joined: list[dict[str, Any]], reviewer: str, arm: str
) -> dict[str, Any]:
    rows = [row[reviewer] for row in joined if row["arm"] == arm]
    return {
        "count": len(rows),
        "clean_good": sum(row["relative_quality"] == "clean_good" for row in rows),
        "mild_or_uncertain": sum(
            row["relative_quality"] == "mild_or_uncertain" for row in rows
        ),
        "clear_bad": sum(row["clear_bad"] for row in rows),
        "clear_bad_rate": sum(row["clear_bad"] for row in rows) / len(rows),
        "blur_or_soft_fusion_rate": sum(row["blur_or_soft_fusion"] for row in rows)
        / len(rows),
        "topology_or_attachment_error_rate": sum(
            row["topology_or_attachment_error"] for row in rows
        )
        / len(rows),
        "mean_ordinal_severity": sum(row["quality_code"] for row in rows) / len(rows),
    }


def binary_kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("kappa inputs are malformed")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    p_left = sum(left) / len(left)
    p_right = sum(right) / len(right)
    expected = p_left * p_right + (1.0 - p_left) * (1.0 - p_right)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def publish(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite audit analysis: {output}")
    delivery = validate_delivery(args.delivery)
    mapping = validate_mapping(args.mapping, str(delivery["identity_sha256"]))
    mapping_rows = mapping["mapping"]
    expected_ids = {row["anonymous_id"] for row in mapping_rows}
    review_paths = [args.reviewer_1.expanduser().resolve(), args.reviewer_2.expanduser().resolve()]
    reviews = [read_response(path, expected_ids) for path in review_paths]
    joined: list[dict[str, Any]] = []
    for mapping_row in mapping_rows:
        anonymous_id = mapping_row["anonymous_id"]
        joined.append(
            {
                **mapping_row,
                "reviewer_1": reviews[0][anonymous_id],
                "reviewer_2": reviews[1][anonymous_id],
            }
        )
    result: dict[str, Any] = {
        "status": "RETROSPECTIVE_BLIND_MODEL_REVIEW_EXPLORATORY_ONLY",
        "delivery_identity_sha256": delivery["identity_sha256"],
        "mapping_identity_sha256": mapping["identity_sha256"],
        "score_shard_manifest_ids": mapping["score_shard_manifest_ids"],
        "reviewer_response_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in review_paths
        ],
        "reviewer_independence_limit": (
            "two separately forked score-blind model-agent reviews; not independent human experts"
        ),
        "pair_design": {
            "pairs": 26,
            "same_class_pairs": 26,
            "exact_start_schedule_pairs": 26,
            "selection_used_old_labels": False,
        },
        "reviewers": {},
    }
    for reviewer in ("reviewer_1", "reviewer_2"):
        result["reviewers"][reviewer] = {
            "alarm": arm_summary(joined, reviewer, "alarm"),
            "control": arm_summary(joined, reviewer, "control"),
            "paired_clear_bad": paired_binary(joined, "clear_bad", reviewer),
            "paired_blur_or_soft_fusion": paired_binary(
                joined, "blur_or_soft_fusion", reviewer
            ),
            "paired_topology_or_attachment_error": paired_binary(
                joined, "topology_or_attachment_error", reviewer
            ),
        }
    left = [row["reviewer_1"]["clear_bad"] for row in joined]
    right = [row["reviewer_2"]["clear_bad"] for row in joined]
    result["agreement"] = {
        "binary_clear_bad_exact_agreement": sum(
            a == b for a, b in zip(left, right, strict=True)
        )
        / len(left),
        "binary_clear_bad_cohen_kappa": binary_kappa(left, right),
        "three_level_exact_agreement": sum(
            row["reviewer_1"]["relative_quality"]
            == row["reviewer_2"]["relative_quality"]
            for row in joined
        )
        / len(joined),
    }
    # Strict 2/2 consensus is deliberately conservative.  It is analyzed as
    # another paired endpoint, not used to change E or its threshold.
    for row in joined:
        row["strict_consensus"] = {
            "clear_bad": row["reviewer_1"]["clear_bad"]
            and row["reviewer_2"]["clear_bad"],
            "blur_or_soft_fusion": row["reviewer_1"]["blur_or_soft_fusion"]
            and row["reviewer_2"]["blur_or_soft_fusion"],
            "topology_or_attachment_error": row["reviewer_1"][
                "topology_or_attachment_error"
            ]
            and row["reviewer_2"]["topology_or_attachment_error"],
        }
    result["strict_2of2_consensus"] = {
        "alarm_clear_bad": sum(
            row["strict_consensus"]["clear_bad"]
            for row in joined
            if row["arm"] == "alarm"
        ),
        "control_clear_bad": sum(
            row["strict_consensus"]["clear_bad"]
            for row in joined
            if row["arm"] == "control"
        ),
        "paired_clear_bad": paired_binary(joined, "clear_bad", "strict_consensus"),
        "paired_blur_or_soft_fusion": paired_binary(
            joined, "blur_or_soft_fusion", "strict_consensus"
        ),
        "paired_topology_or_attachment_error": paired_binary(
            joined, "topology_or_attachment_error", "strict_consensus"
        ),
    }
    result["claim_limit"] = (
        "This tests whether the frozen E>=10 event enriches visible defects relative to "
        "same-class exact-start-schedule controls in an already viewed old pool. It is not "
        "prospective confirmation, does not estimate deployment TPR/FPR, and cannot authorize rollback."
    )
    result["identity_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "results.json", result)
        with (staging / "joined_private_rows.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            columns = (
                "anonymous_id",
                "pair_index",
                "arm",
                "global_seed",
                "class_id",
                "reviewer_1_quality",
                "reviewer_1_blur",
                "reviewer_1_topology",
                "reviewer_2_quality",
                "reviewer_2_blur",
                "reviewer_2_topology",
            )
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in joined:
                writer.writerow(
                    {
                        "anonymous_id": row["anonymous_id"],
                        "pair_index": row["pair_index"],
                        "arm": row["arm"],
                        "global_seed": row["global_seed"],
                        "class_id": row["class_id"],
                        "reviewer_1_quality": row["reviewer_1"]["relative_quality"],
                        "reviewer_1_blur": str(
                            row["reviewer_1"]["blur_or_soft_fusion"]
                        ).lower(),
                        "reviewer_1_topology": str(
                            row["reviewer_1"]["topology_or_attachment_error"]
                        ).lower(),
                        "reviewer_2_quality": row["reviewer_2"]["relative_quality"],
                        "reviewer_2_blur": str(
                            row["reviewer_2"]["blur_or_soft_fusion"]
                        ).lower(),
                        "reviewer_2_topology": str(
                            row["reviewer_2"]["topology_or_attachment_error"]
                        ).lower(),
                    }
                )
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_ALARM_BLIND_AUDIT_ANALYSIS",
            "result_identity_sha256": result["identity_sha256"],
            "files": [
                {
                    "name": name,
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("results.json", "joined_private_rows.csv")
            ],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        write_json(
            staging / "completion.json",
            {
                "complete": True,
                "manifest_identity_sha256": manifest["identity_sha256"],
                "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def self_test() -> None:
    if one_sided_binomial_tail(3, 3) != 0.125:
        raise AssertionError("exact binomial tail changed")
    if one_sided_binomial_tail(0, 0) is not None:
        raise AssertionError("zero-discordance tail must be undefined")
    if binary_kappa([True, False], [True, False]) != 1.0:
        raise AssertionError("binary kappa perfect agreement changed")
    print("v2.2 alarm blind-audit analysis self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--reviewer-1", type=Path, required=True)
    parser.add_argument("--reviewer-2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
    else:
        publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
