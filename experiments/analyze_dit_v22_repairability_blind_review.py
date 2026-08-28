#!/usr/bin/env python3
"""Analyze three complete score/role-blind repairability reviews.

Reviewer files are completely validated against the anonymous delivery axis
before the private mapping is opened.  The analysis uses fixed 2-of-3 rules
and a fixed-seed path-cluster bootstrap.  It is an exploratory observational
effect-modification readout, never a deployment or causal authorization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

try:
    from .prepare_dit_v22_repairability_blind_review import (
        RESPONSE_COLUMNS,
        canonical_sha256,
        load_json,
        self_hashed,
        sha256_file,
        write_json,
    )
except ImportError:  # pragma: no cover
    from prepare_dit_v22_repairability_blind_review import (
        RESPONSE_COLUMNS,
        canonical_sha256,
        load_json,
        self_hashed,
        sha256_file,
        write_json,
    )


QUALITY = ("clean_good", "mild_or_uncertain", "clear_bad")
BOOLEAN = {"true": True, "false": False}
PRESERVATION = ("yes", "no", "uncertain")
PREFERENCE = ("left", "right", "tie")
BOOTSTRAP_SEED = 2026082812
BOOTSTRAP_REPLICATES = 50_000
JOINT_ROLE = "joint_E_and_B"
B_ONLY_ROLE = "B_only_exact_schedule_B_matched_control"
CONSENSUS_DEFINITIONS = {
    "baseline_repair_opportunity": "2/3 reviewers each judge the baseline both non-clean AND blurred",
    "semantic_preservation": "2/3 identity_composition_preserved=yes",
    "improvement": "2/3 preferred_side=fresh",
    "fresh_failure": "2/3 fresh quality=clear_bad",
    "success": "opportunity AND preservation AND improvement AND NOT fresh_failure",
    "path_has_baseline_opportunity": "at least 4 of that path's 8 anonymous comparisons meet pair-level opportunity",
}


def validate_delivery(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = self_hashed(root / "manifest.json", "identity_sha256")
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_kind")
        != "DIT_V22_REPAIRABILITY_SCORE_ROLE_BLIND_DELIVERY_V1"
        or manifest.get("anonymous_pair_count") != 128
        or manifest.get("native_image_count") != 256
        or manifest.get("pair_image_count") != 128
        or manifest.get("reviewer_response_template_count") != 3
        or manifest.get("private_mapping_present") is not False
        or manifest.get("nonvisual_measurements_present") is not False
    ):
        raise RuntimeError("blind delivery contract changed")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if set(records) != observed:
        raise RuntimeError("blind delivery exact tree changed")
    for name, record in records.items():
        path = root / name
        if (
            path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"blind delivery member changed: {name}")
    return manifest


def template_ids(delivery_root: Path) -> list[str]:
    axes: list[list[str]] = []
    for reviewer in range(1, 4):
        path = delivery_root / f"reviewer_{reviewer}_response.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RESPONSE_COLUMNS:
                raise RuntimeError("delivery response-template header changed")
            rows = list(reader)
        if len(rows) != 128:
            raise RuntimeError("delivery response-template row count changed")
        ids = []
        for row in rows:
            identifier = row["anonymous_pair_id"]
            if any(row[name].strip() for name in RESPONSE_COLUMNS[1:]):
                raise RuntimeError("delivery response template is not empty")
            ids.append(identifier)
        if len(set(ids)) != 128:
            raise RuntimeError("delivery response-template IDs are not unique")
        axes.append(ids)
    if not all(axis == axes[0] for axis in axes[1:]):
        raise RuntimeError("delivery response-template axes differ")
    return axes[0]


def read_response(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular complete response: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESPONSE_COLUMNS:
            raise RuntimeError(f"review response header changed: {path}")
        raw = list(reader)
    if len(raw) != 128:
        raise RuntimeError(f"review response must contain 128 rows: {path}")
    result: dict[str, dict[str, Any]] = {}
    for row in raw:
        identifier = row["anonymous_pair_id"]
        if identifier in result or identifier not in expected_ids:
            raise RuntimeError(f"invalid anonymous pair ID: {identifier}")
        left_quality = row["left_quality"].strip()
        right_quality = row["right_quality"].strip()
        if left_quality not in QUALITY or right_quality not in QUALITY:
            raise RuntimeError(f"invalid quality value: {identifier}")
        try:
            left_blur = BOOLEAN[row["left_blur"].strip().lower()]
            right_blur = BOOLEAN[row["right_blur"].strip().lower()]
            left_topology = BOOLEAN[row["left_topology"].strip().lower()]
            right_topology = BOOLEAN[row["right_topology"].strip().lower()]
        except KeyError as exc:
            raise RuntimeError(f"review booleans must be true/false: {identifier}") from exc
        preservation = row["identity_composition_preserved"].strip().lower()
        preference = row["preferred_side"].strip().lower()
        reason = row["localized_reason"].strip()
        if preservation not in PRESERVATION:
            raise RuntimeError(f"invalid preservation value: {identifier}")
        if preference not in PREFERENCE:
            raise RuntimeError(f"invalid preferred side: {identifier}")
        if not reason:
            raise RuntimeError(f"localized_reason is required: {identifier}")
        result[identifier] = {
            "left_quality": left_quality,
            "right_quality": right_quality,
            "left_blur": left_blur,
            "right_blur": right_blur,
            "left_topology": left_topology,
            "right_topology": right_topology,
            "identity_composition_preserved": preservation,
            "preferred_side": preference,
            "localized_reason": reason,
        }
    if set(result) != expected_ids:
        raise RuntimeError(f"review response anonymous axis incomplete: {path}")
    return result


def validate_mapping(path: Path, delivery_identity: str, expected_ids: set[str]) -> dict[str, Any]:
    mapping = self_hashed(path.expanduser().resolve(), "identity_sha256")
    rows = mapping.get("mapping")
    if (
        mapping.get("status") != "SEALED_UNTIL_THREE_COMPLETE_REVIEWS"
        or mapping.get("artifact_kind") != "DIT_V22_REPAIRABILITY_PRIVATE_MAPPING_V1"
        or mapping.get("delivery_identity_sha256") != delivery_identity
        or mapping.get("comparison_count") != 128
        or mapping.get("job_count") != 32
        or mapping.get("fresh_attempts_per_job") != 4
        or mapping.get("scores_labels_or_external_features_used_for_pack_order_or_sides") is not False
        or not isinstance(rows, list)
        or len(rows) != 128
    ):
        raise RuntimeError("private mapping contract changed")
    ids = {row.get("anonymous_pair_id") for row in rows if isinstance(row, dict)}
    if ids != expected_ids:
        raise RuntimeError("private mapping anonymous axis changed")
    job_attempts: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        if row.get("role") not in {JOINT_ROLE, B_ONLY_ROLE}:
            raise RuntimeError("private mapping role changed")
        if row.get("rollback_sampling_step") not in {109, 149}:
            raise RuntimeError("private mapping rollback step changed")
        if row.get("baseline_side") not in {"left", "right"}:
            raise RuntimeError("private mapping baseline side changed")
        if row.get("fresh_side") == row.get("baseline_side") or row.get("fresh_side") not in {"left", "right"}:
            raise RuntimeError("private mapping side assignment changed")
        attempt = int(row.get("fresh_attempt"))
        if attempt not in {1, 2, 3, 4}:
            raise RuntimeError("private mapping fresh attempt changed")
        job_attempts[int(row["job_index"])].add(attempt)
    if len(job_attempts) != 32 or any(value != {1, 2, 3, 4} for value in job_attempts.values()):
        raise RuntimeError("private mapping job/attempt matrix changed")
    return mapping


def at_least_two(votes: Iterable[bool]) -> bool:
    return sum(bool(value) for value in votes) >= 2


def role_view(review: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    baseline = str(mapping["baseline_side"])
    fresh = str(mapping["fresh_side"])
    return {
        "baseline_quality": review[f"{baseline}_quality"],
        "fresh_quality": review[f"{fresh}_quality"],
        "baseline_blur": review[f"{baseline}_blur"],
        "fresh_blur": review[f"{fresh}_blur"],
        "baseline_topology": review[f"{baseline}_topology"],
        "fresh_topology": review[f"{fresh}_topology"],
        "preservation": review["identity_composition_preserved"],
        "preferred_fresh": review["preferred_side"] == fresh,
        "preferred_baseline": review["preferred_side"] == baseline,
        "tie": review["preferred_side"] == "tie",
        "localized_reason": review["localized_reason"],
    }


def consensus_row(mapping: dict[str, Any], role_reviews: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_nonclean = at_least_two(row["baseline_quality"] != "clean_good" for row in role_reviews)
    baseline_blur = at_least_two(row["baseline_blur"] for row in role_reviews)
    # Opportunity is first defined within a reviewer and then voted.  This
    # prevents two different reviewer minorities from manufacturing a joint
    # non-clean-and-blur phenotype.
    opportunity = at_least_two(
        row["baseline_quality"] != "clean_good" and row["baseline_blur"]
        for row in role_reviews
    )
    preservation = at_least_two(row["preservation"] == "yes" for row in role_reviews)
    improvement = at_least_two(row["preferred_fresh"] for row in role_reviews)
    fresh_clear_bad = at_least_two(row["fresh_quality"] == "clear_bad" for row in role_reviews)
    success = opportunity and preservation and improvement and not fresh_clear_bad
    return {
        **mapping,
        "reviews": role_reviews,
        "baseline_nonclean_2of3": baseline_nonclean,
        "baseline_blur_2of3": baseline_blur,
        "baseline_repair_opportunity_2of3": opportunity,
        "semantic_preservation_2of3_yes": preservation,
        "improvement_2of3_prefer_fresh": improvement,
        "fresh_failure_2of3_clear_bad": fresh_clear_bad,
        "success": success,
        "worsening_2of3_prefer_baseline": at_least_two(row["preferred_baseline"] for row in role_reviews),
        "fresh_blur_2of3": at_least_two(row["fresh_blur"] for row in role_reviews),
        "fresh_topology_2of3": at_least_two(row["fresh_topology"] for row in role_reviews),
    }


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    opportunities = sum(row["baseline_repair_opportunity_2of3"] for row in rows)
    successes = sum(row["success"] for row in rows)
    return {
        "comparison_count": count,
        "unique_path_count": len({(row["role"], row["pair_index"]) for row in rows}),
        "baseline_nonclean_rate": mean([float(row["baseline_nonclean_2of3"]) for row in rows]),
        "baseline_blur_rate": mean([float(row["baseline_blur_2of3"]) for row in rows]),
        "baseline_repair_opportunity_rate": opportunities / count if count else None,
        "semantic_preservation_rate": mean([float(row["semantic_preservation_2of3_yes"]) for row in rows]),
        "improvement_rate": mean([float(row["improvement_2of3_prefer_fresh"]) for row in rows]),
        "fresh_clear_bad_failure_rate": mean([float(row["fresh_failure_2of3_clear_bad"]) for row in rows]),
        "worsening_rate": mean([float(row["worsening_2of3_prefer_baseline"]) for row in rows]),
        "success_count": successes,
        "success_rate_all_comparisons": successes / count if count else None,
        "success_rate_among_opportunities": successes / opportunities if opportunities else None,
    }


def path_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["role"]), int(row["pair_index"]))].append(row)
    output = []
    for (role, pair_index), block in sorted(groups.items()):
        if len(block) != 8 or {row["rollback_sampling_step"] for row in block} != {109, 149}:
            raise RuntimeError("each original path must contribute exactly 8 comparisons at two steps")
        if any(sum(row["rollback_sampling_step"] == step for row in block) != 4 for step in (109, 149)):
            raise RuntimeError("each path/step must contribute four fresh attempts")
        output.append(
            {
                "role": role,
                "pair_index": pair_index,
                "global_seed": int(block[0]["global_seed"]),
                "class_id": int(block[0]["class_id"]),
                "comparison_count": 8,
                "baseline_opportunity_fraction": mean([float(row["baseline_repair_opportunity_2of3"]) for row in block]),
                "path_has_baseline_opportunity": sum(row["baseline_repair_opportunity_2of3"] for row in block) >= 4,
                "preservation_rate": mean([float(row["semantic_preservation_2of3_yes"]) for row in block]),
                "improvement_rate": mean([float(row["improvement_2of3_prefer_fresh"]) for row in block]),
                "fresh_clear_bad_failure_rate": mean([float(row["fresh_failure_2of3_clear_bad"]) for row in block]),
                "success_rate": mean([float(row["success"]) for row in block]),
                "step109_success_rate": mean([float(row["success"]) for row in block if row["rollback_sampling_step"] == 109]),
                "step149_success_rate": mean([float(row["success"]) for row in block if row["rollback_sampling_step"] == 149]),
            }
        )
    if len(output) != 16:
        raise RuntimeError("expected exactly 16 original path clusters")
    return output


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_difference(
    paths: list[dict[str, Any]], field: str, *, seed_offset: int = 0
) -> dict[str, Any]:
    joint = [float(row[field]) for row in paths if row["role"] == JOINT_ROLE]
    bonly = [float(row[field]) for row in paths if row["role"] == B_ONLY_ROLE]
    if len(joint) != 8 or len(bonly) != 8:
        raise RuntimeError("joint/B-only path cluster counts changed")
    observed = float(mean(joint)) - float(mean(bonly))
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        joint_draw = sum(joint[rng.randrange(len(joint))] for _ in joint) / len(joint)
        bonly_draw = sum(bonly[rng.randrange(len(bonly))] for _ in bonly) / len(bonly)
        draws.append(joint_draw - bonly_draw)
    return {
        "estimand": f"mean path-level {field}: joint_E_and_B minus B_only_control",
        "joint_mean": mean(joint),
        "B_only_mean": mean(bonly),
        "descriptive_difference": observed,
        "fixed_seed_path_cluster_bootstrap": {
            "seed": BOOTSTRAP_SEED + seed_offset,
            "replicates": BOOTSTRAP_REPLICATES,
            "percentile_95_CI": [quantile(draws, 0.025), quantile(draws, 0.975)],
        },
    }


def fleiss_kappa(items: list[list[str]]) -> float | None:
    if not items or any(len(row) != 3 for row in items):
        raise ValueError("Fleiss-kappa items require exactly three ratings")
    categories = sorted({value for row in items for value in row})
    n = 3
    p_bar = sum(
        sum(count * count for count in Counter(row).values()) - n for row in items
    ) / (len(items) * n * (n - 1))
    category_counts = Counter(value for row in items for value in row)
    p_expected = sum((category_counts[value] / (len(items) * n)) ** 2 for value in categories)
    return None if p_expected == 1.0 else (p_bar - p_expected) / (1.0 - p_expected)


def agreement(reviews: list[dict[str, dict[str, Any]]], ids: list[str]) -> dict[str, Any]:
    item_sets: dict[str, list[list[str]]] = {
        "side_quality": [],
        "side_blur": [],
        "side_topology": [],
        "identity_composition_preserved": [],
        "preferred_side": [],
    }
    for identifier in ids:
        for side in ("left", "right"):
            item_sets["side_quality"].append([review[identifier][f"{side}_quality"] for review in reviews])
            item_sets["side_blur"].append([str(review[identifier][f"{side}_blur"]) for review in reviews])
            item_sets["side_topology"].append([str(review[identifier][f"{side}_topology"]) for review in reviews])
        item_sets["identity_composition_preserved"].append(
            [review[identifier]["identity_composition_preserved"] for review in reviews]
        )
        item_sets["preferred_side"].append([review[identifier]["preferred_side"] for review in reviews])
    return {
        name: {
            "item_count": len(items),
            "unanimous_exact_agreement": sum(len(set(row)) == 1 for row in items) / len(items),
            "fleiss_kappa": fleiss_kappa(items),
        }
        for name, items in item_sets.items()
    }


def publish(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite analysis: {output}")
    delivery_root = args.delivery.expanduser().resolve()
    mapping_path = args.mapping.expanduser().resolve()
    if delivery_root == mapping_path or delivery_root in mapping_path.parents:
        raise RuntimeError("private mapping must be physically separate from delivery")
    delivery = validate_delivery(delivery_root)
    ordered_ids = template_ids(delivery_root)
    expected_ids = set(ordered_ids)
    # This ordering is deliberate: the private mapping is not read at all until
    # every independently supplied response passes the complete schema/axis test.
    review_paths = [path.expanduser().resolve() for path in args.reviewers]
    if len(review_paths) != 3 or len(set(review_paths)) != 3:
        raise RuntimeError("exactly three distinct reviewer response paths are required")
    reviews = [read_response(path, expected_ids) for path in review_paths]
    mapping = validate_mapping(mapping_path, str(delivery["identity_sha256"]), expected_ids)
    mapping_by_id = {row["anonymous_pair_id"]: row for row in mapping["mapping"]}
    joined: list[dict[str, Any]] = []
    for identifier in ordered_ids:
        map_row = mapping_by_id[identifier]
        role_reviews = [role_view(review[identifier], map_row) for review in reviews]
        joined.append(consensus_row(map_row, role_reviews))
    paths = path_summaries(joined)
    by_role = {
        role: summarize([row for row in joined if row["role"] == role])
        for role in (JOINT_ROLE, B_ONLY_ROLE)
    }
    by_role_step = {
        role: {
            str(step): summarize(
                [
                    row
                    for row in joined
                    if row["role"] == role and row["rollback_sampling_step"] == step
                ]
            )
            for step in (109, 149)
        }
        for role in (JOINT_ROLE, B_ONLY_ROLE)
    }
    opportunity_paths = {
        role: sum(row["path_has_baseline_opportunity"] for row in paths if row["role"] == role)
        for role in (JOINT_ROLE, B_ONLY_ROLE)
    }
    result: dict[str, Any] = {
        "status": "RETROSPECTIVE_REPAIRABILITY_BLIND_REVIEW_EXPLORATORY_ONLY",
        "delivery_identity_sha256": delivery["identity_sha256"],
        "mapping_identity_sha256": mapping["identity_sha256"],
        "reviewer_response_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in review_paths
        ],
        "design": {
            "original_paths": 16,
            "jobs_path_by_step": 32,
            "fresh_attempts_per_job": 4,
            "baseline_fresh_pairs": 128,
            "reviews_per_pair": 3,
            "no_FID_features_or_endpoint_scores": True,
        },
        "fixed_consensus_definitions": CONSENSUS_DEFINITIONS,
        "overall": summarize(joined),
        "by_role": by_role,
        "by_role_and_rollback_step": by_role_step,
        "per_path_eight_attempt_rates": paths,
        "baseline_opportunity_path_counts": opportunity_paths,
        "frozen_guard": {
            "minimum_opportunity_paths_per_role": 4,
            "satisfied": all(value >= 4 for value in opportunity_paths.values()),
            "interpretation": (
                "between-role comparison is inconclusive when either role has fewer than four paths with a repairable baseline defect"
            ),
        },
        "joint_vs_B_only_descriptive": {
            "overall_success_rate": cluster_difference(paths, "success_rate", seed_offset=0),
            "step109_success_rate": cluster_difference(paths, "step109_success_rate", seed_offset=109),
            "step149_success_rate": cluster_difference(paths, "step149_success_rate", seed_offset=149),
        },
        "reviewer_agreement": agreement(reviews, ordered_ids),
        "limits": [
            "Exploratory observational effect-modification analysis: E-group membership is observational even though suffix noise is randomized within path.",
            "The 128 comparisons are repeated attempts nested in only 16 original paths; attempt-level rows are not independent paths.",
            "Bootstrap intervals are descriptive fixed-seed path-cluster intervals, not preregistered confirmatory inference.",
            "No causal claim, deployment claim, rollback authorization, Ville/TV guarantee, or best-of-N claim follows from this analysis.",
            "No FID, Inception, DINO, CLIP, endpoint representation, or internal score is used as the visual outcome.",
        ],
    }
    result["identity_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "results.json", result)
        with (staging / "joined_private_comparisons.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            columns = (
                "anonymous_pair_id",
                "role",
                "pair_index",
                "global_seed",
                "class_id",
                "rollback_sampling_step",
                "fresh_attempt",
                "baseline_repair_opportunity_2of3",
                "semantic_preservation_2of3_yes",
                "improvement_2of3_prefer_fresh",
                "fresh_failure_2of3_clear_bad",
                "success",
            )
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in joined:
                writer.writerow({name: row[name] for name in columns})
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_REPAIRABILITY_BLIND_REVIEW_ANALYSIS_V1",
            "result_identity_sha256": result["identity_sha256"],
            "files": [
                {
                    "name": name,
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("results.json", "joined_private_comparisons.csv")
            ],
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
        }
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def self_test() -> None:
    if not at_least_two([True, True, False]) or at_least_two([True, False, False]):
        raise AssertionError("2/3 consensus changed")
    if fleiss_kappa([["a", "a", "a"], ["b", "b", "b"]]) != 1.0:
        raise AssertionError("Fleiss kappa perfect agreement changed")
    if quantile([0.0, 1.0], 0.5) != 0.5:
        raise AssertionError("bootstrap quantile interpolation changed")
    mapping = {"baseline_side": "right", "fresh_side": "left"}
    review = {
        "left_quality": "clean_good",
        "right_quality": "mild_or_uncertain",
        "left_blur": False,
        "right_blur": True,
        "left_topology": False,
        "right_topology": False,
        "identity_composition_preserved": "yes",
        "preferred_side": "left",
        "localized_reason": "edge",
    }
    view = role_view(review, mapping)
    if view["baseline_quality"] != "mild_or_uncertain" or not view["preferred_fresh"]:
        raise AssertionError("left/right role unblinding changed")
    row = consensus_row(
        {"baseline_side": "right", "fresh_side": "left"}, [view, view, view]
    )
    if not row["success"]:
        raise AssertionError("fixed successful-repair definition changed")
    split_votes = [
        dict(view, baseline_quality="mild_or_uncertain", baseline_blur=False),
        dict(view, baseline_quality="clean_good", baseline_blur=True),
        dict(view, baseline_quality="mild_or_uncertain", baseline_blur=True),
    ]
    if consensus_row(mapping, split_votes)["baseline_repair_opportunity_2of3"]:
        raise AssertionError("opportunity must be a 2/3 vote on the within-reviewer conjunction")
    with tempfile.TemporaryDirectory(prefix="repair-review-selftest-") as temporary:
        response = Path(temporary) / "response.csv"
        with response.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
            writer.writeheader()
            for index in range(127):
                writer.writerow(
                    {
                        "anonymous_pair_id": f"R{index:04d}",
                        "left_quality": "clean_good",
                        "right_quality": "clean_good",
                        "left_blur": "false",
                        "right_blur": "false",
                        "left_topology": "false",
                        "right_topology": "false",
                        "identity_composition_preserved": "yes",
                        "preferred_side": "tie",
                        "localized_reason": "no visible difference",
                    }
                )
        try:
            read_response(response, {f"R{index:04d}" for index in range(128)})
        except RuntimeError:
            pass
        else:
            raise AssertionError("an incomplete 127-row response did not fail closed")
    print("repairability blind-review analysis self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, required=False)
    parser.add_argument("--mapping", type=Path, required=False)
    parser.add_argument("--reviewers", nargs=3, type=Path, required=False)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.delivery is None or args.mapping is None or args.reviewers is None or args.output is None:
        raise SystemExit("--delivery, --mapping, three --reviewers paths, and --output are required")
    publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
