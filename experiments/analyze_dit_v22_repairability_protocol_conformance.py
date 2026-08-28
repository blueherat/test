#!/usr/bin/env python3
"""Protocol-conformance corrigendum for the DiT v2.2 repairability pilot.

The frozen v1 analysis accidentally operationalized "successful repair" with
two-of-three overall fresh-side preference.  The upstream selection protocol
instead requires visible blur/fusion to be clearly reduced.  This script keeps
the completed blind responses immutable and implements that literal criterion
as a two-of-three within-reviewer transition from baseline_blur=true to
fresh_blur=false.  It is a transparent post-unblinding corrigendum, not a new
confirmatory experiment and not an intervention authorization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
SELECTION_LOCK = ROOT / "experiments/locks/dit_v22_repairability_pilot_lock_v1_2"
REVIEW_LOCK = ROOT / "experiments/locks/dit_v22_repairability_review_source_lock_v1"
LOCKED_REVIEW_SOURCES = REVIEW_LOCK / "sources"
sys.path.insert(0, str(LOCKED_REVIEW_SOURCES))

import analyze_dit_v22_repairability_blind_review as frozen  # noqa: E402


SELECTION_LOCK_ID = "16acd0bffda207ed73ef78a62909e53997bef68baae66cdffedede1bb207fbd0"
SELECTION_PROTOCOL_ID = "f39c5a8bfbbc129d6e80ca5e38a07dfd886c6c41faff15337042127e78b3ae77"
REVIEW_LOCK_ID = "ca06061606ad46719e059298819fc0c66d79a9bcdd0be740635966274ed1bcab"
REVIEW_CONTRACT_ID = "3de445dde23e433131e5b82b47d15504a7b308103426548b442def5706f5a6aa"
ORIGINAL_ANALYSIS_ID = "055768a7e7b519c79e037b8649744a1cc9e055a033d9c73c1be6596da3da3005"
DEFAULT_DELIVERY = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_blind_review_v1_delivery"
)
DEFAULT_MAPPING = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_blind_review_v1_private/sealed_mapping.json"
)
DEFAULT_REVIEWS = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_blind_review_v1_reviews"
)
DEFAULT_ORIGINAL_ANALYSIS = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_blind_review_v1_analysis"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_protocol_conformance_v2"
)
BOOTSTRAP_SEED = 2026084000
BOOTSTRAP_REPLICATES = 100_000
JOINT = frozen.JOINT_ROLE
B_ONLY = frozen.B_ONLY_ROLE


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing regular JSON: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def self_hashed(path: Path, key: str = "identity_sha256") -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"invalid self hash: {path}")
    return value


def validate_upstream_protocol() -> dict[str, Any]:
    selection_manifest = self_hashed(SELECTION_LOCK / "manifest.json")
    selection_protocol = self_hashed(SELECTION_LOCK / "protocol.json")
    review_manifest = self_hashed(REVIEW_LOCK / "manifest.json")
    review_contract = self_hashed(REVIEW_LOCK / "review_contract.json")
    if (
        selection_manifest.get("identity_sha256") != SELECTION_LOCK_ID
        or selection_protocol.get("identity_sha256") != SELECTION_PROTOCOL_ID
        or selection_manifest.get("protocol_identity_sha256") != SELECTION_PROTOCOL_ID
        or review_manifest.get("identity_sha256") != REVIEW_LOCK_ID
        or review_contract.get("identity_sha256") != REVIEW_CONTRACT_ID
        or review_manifest.get("review_contract_identity_sha256") != REVIEW_CONTRACT_ID
    ):
        raise RuntimeError("upstream lock identity changed")
    literal = selection_protocol.get("evaluation_frozen_before_outputs", {}).get(
        "successful_repair"
    )
    expected = (
        "visible blur/fusion is clearly reduced AND class/identity/object count/main "
        "pose/composition are preserved AND no new equally severe defect appears"
    )
    if literal != expected:
        raise RuntimeError("upstream successful-repair literal changed")
    return {
        "selection_lock_identity_sha256": SELECTION_LOCK_ID,
        "selection_protocol_identity_sha256": SELECTION_PROTOCOL_ID,
        "review_lock_identity_sha256": REVIEW_LOCK_ID,
        "review_contract_identity_sha256": REVIEW_CONTRACT_ID,
        "upstream_successful_repair_literal": literal,
    }


def validate_original_analysis(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    manifest = self_hashed(root / "manifest.json")
    result = self_hashed(root / "results.json")
    if (
        manifest.get("artifact_kind")
        != "DIT_V22_REPAIRABILITY_BLIND_REVIEW_ANALYSIS_V1"
        or manifest.get("status") != "complete"
        or manifest.get("result_identity_sha256") != ORIGINAL_ANALYSIS_ID
        or result.get("identity_sha256") != ORIGINAL_ANALYSIS_ID
    ):
        raise RuntimeError("original frozen analysis changed")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    for name in ("results.json", "joined_private_comparisons.csv"):
        path = root / name
        record = records.get(name)
        if (
            not isinstance(record, dict)
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"original analysis member changed: {name}")
    completion = load_json(root / "completion.json")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
    ):
        raise RuntimeError("original analysis completion changed")
    return result


def at_least_two(values: Iterable[bool]) -> bool:
    return sum(bool(value) for value in values) >= 2


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def binomial_upper_tail(successes: int, trials: int) -> float | None:
    if trials == 0:
        return None
    return sum(math.comb(trials, k) for k in range(successes, trials + 1)) / (2**trials)


def consensus_row(mapping: dict[str, Any], reviews: list[dict[str, Any]]) -> dict[str, Any]:
    opportunity = at_least_two(
        row["baseline_quality"] != "clean_good" and row["baseline_blur"]
        for row in reviews
    )
    # This is the most literal deterministic resolution available from the
    # already-frozen binary blur fields: the same reviewer must see blur on the
    # baseline and no blur on the fresh side before casting a reduction vote.
    blur_reduced = at_least_two(
        row["baseline_blur"] and not row["fresh_blur"] for row in reviews
    )
    preservation = at_least_two(row["preservation"] == "yes" for row in reviews)
    fresh_clear_bad = at_least_two(row["fresh_quality"] == "clear_bad" for row in reviews)
    fresh_topology = at_least_two(row["fresh_topology"] for row in reviews)
    new_topology = at_least_two(
        (not row["baseline_topology"]) and row["fresh_topology"] for row in reviews
    )
    preferred_fresh = at_least_two(row["preferred_fresh"] for row in reviews)
    strict_success = opportunity and blur_reduced and preservation and not fresh_clear_bad
    return {
        **mapping,
        "baseline_repair_opportunity_2of3": opportunity,
        "blur_reduced_within_reviewer_2of3": blur_reduced,
        "semantic_preservation_2of3_yes": preservation,
        "fresh_failure_2of3_clear_bad": fresh_clear_bad,
        "fresh_topology_2of3": fresh_topology,
        "new_topology_transition_2of3": new_topology,
        "preferred_fresh_2of3": preferred_fresh,
        "protocol_literal_success": strict_success,
        "protocol_literal_success_plus_no_fresh_topology_sensitivity": (
            strict_success and not fresh_topology
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    opportunities = sum(row["baseline_repair_opportunity_2of3"] for row in rows)
    successes = sum(row["protocol_literal_success"] for row in rows)
    return {
        "comparison_count": count,
        "unique_path_count": len({(row["role"], row["pair_index"]) for row in rows}),
        "baseline_opportunity_count": opportunities,
        "baseline_opportunity_rate": opportunities / count if count else None,
        "blur_reduced_count": sum(row["blur_reduced_within_reviewer_2of3"] for row in rows),
        "blur_reduced_rate": mean(
            [float(row["blur_reduced_within_reviewer_2of3"]) for row in rows]
        ),
        "semantic_preservation_rate": mean(
            [float(row["semantic_preservation_2of3_yes"]) for row in rows]
        ),
        "fresh_clear_bad_failure_rate": mean(
            [float(row["fresh_failure_2of3_clear_bad"]) for row in rows]
        ),
        "protocol_literal_success_count": successes,
        "protocol_literal_success_rate_all_comparisons": successes / count if count else None,
        "protocol_literal_success_rate_among_opportunities": (
            successes / opportunities if opportunities else None
        ),
        "preferred_fresh_rate": mean([float(row["preferred_fresh_2of3"]) for row in rows]),
        "new_topology_transition_rate": mean(
            [float(row["new_topology_transition_2of3"]) for row in rows]
        ),
        "strict_success_plus_no_fresh_topology_count": sum(
            row["protocol_literal_success_plus_no_fresh_topology_sensitivity"] for row in rows
        ),
    }


def path_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["role"]), int(row["pair_index"]))].append(row)
    output = []
    for (role, pair_index), block in sorted(groups.items()):
        if len(block) != 8:
            raise RuntimeError("each path must contain eight comparisons")
        if any(sum(row["rollback_sampling_step"] == step for row in block) != 4 for step in (109, 149)):
            raise RuntimeError("each path/step must contain four comparisons")
        output.append(
            {
                "role": role,
                "pair_index": pair_index,
                "global_seed": int(block[0]["global_seed"]),
                "class_id": int(block[0]["class_id"]),
                "baseline_opportunity_fraction": mean(
                    [float(row["baseline_repair_opportunity_2of3"]) for row in block]
                ),
                "path_has_baseline_opportunity": sum(
                    row["baseline_repair_opportunity_2of3"] for row in block
                )
                >= 4,
                "protocol_literal_success_count": sum(
                    row["protocol_literal_success"] for row in block
                ),
                "protocol_literal_success_rate": mean(
                    [float(row["protocol_literal_success"]) for row in block]
                ),
                "step109_protocol_literal_success_rate": mean(
                    [
                        float(row["protocol_literal_success"])
                        for row in block
                        if row["rollback_sampling_step"] == 109
                    ]
                ),
                "step149_protocol_literal_success_rate": mean(
                    [
                        float(row["protocol_literal_success"])
                        for row in block
                        if row["rollback_sampling_step"] == 149
                    ]
                ),
            }
        )
    if len(output) != 16:
        raise RuntimeError("expected sixteen path clusters")
    return output


def paired_diagnostic(paths: list[dict[str, Any]], field: str, seed_offset: int) -> dict[str, Any]:
    by_key = {(row["role"], row["pair_index"]): row for row in paths}
    differences = [
        float(by_key[(JOINT, pair)][field]) - float(by_key[(B_ONLY, pair)][field])
        for pair in range(8)
    ]
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = [
        sum(differences[rng.randrange(8)] for _ in range(8)) / 8
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    leave_one_out = [
        sum(value for index, value in enumerate(differences) if index != omitted) / 7
        for omitted in range(8)
    ]
    positives = sum(value > 0 for value in differences)
    negatives = sum(value < 0 for value in differences)
    non_ties = positives + negatives
    return {
        "estimand": f"matched path mean {field}: joint_E_and_B minus B-only",
        "matched_pair_differences": differences,
        "mean_difference": sum(differences) / 8,
        "positive_negative_tie_counts": {
            "joint_better": positives,
            "joint_worse": negatives,
            "tie": 8 - non_ties,
        },
        "one_sided_fair_sign_tail_joint_better": binomial_upper_tail(positives, non_ties),
        "one_sided_fair_sign_tail_joint_worse": binomial_upper_tail(negatives, non_ties),
        "paired_path_bootstrap_posthoc": {
            "seed": BOOTSTRAP_SEED + seed_offset,
            "replicates": BOOTSTRAP_REPLICATES,
            "percentile_95_CI": [quantile(draws, 0.025), quantile(draws, 0.975)],
        },
        "leave_one_pair_out_mean_range": [min(leave_one_out), max(leave_one_out)],
    }


def build_result(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lineage = validate_upstream_protocol()
    original = validate_original_analysis(args.original_analysis)
    delivery = frozen.validate_delivery(args.delivery)
    ordered_ids = frozen.template_ids(args.delivery)
    expected_ids = set(ordered_ids)
    review_paths = [args.reviews / f"reviewer_{index}.csv" for index in (1, 2, 3)]
    reviews = [frozen.read_response(path, expected_ids) for path in review_paths]
    mapping = frozen.validate_mapping(args.mapping, delivery["identity_sha256"], expected_ids)
    mapping_by_id = {row["anonymous_pair_id"]: row for row in mapping["mapping"]}
    rows = []
    for identifier in ordered_ids:
        map_row = mapping_by_id[identifier]
        role_reviews = [frozen.role_view(review[identifier], map_row) for review in reviews]
        rows.append(consensus_row(map_row, role_reviews))
    paths = path_summaries(rows)
    by_role = {
        role: summarize([row for row in rows if row["role"] == role])
        for role in (JOINT, B_ONLY)
    }
    by_role_step = {
        role: {
            str(step): summarize(
                [
                    row
                    for row in rows
                    if row["role"] == role and row["rollback_sampling_step"] == step
                ]
            )
            for step in (109, 149)
        }
        for role in (JOINT, B_ONLY)
    }
    opportunity_paths = {
        role: sum(
            row["path_has_baseline_opportunity"] for row in paths if row["role"] == role
        )
        for role in (JOINT, B_ONLY)
    }
    common_pairs = [
        pair
        for pair in range(8)
        if all(
            next(
                row["path_has_baseline_opportunity"]
                for row in paths
                if row["role"] == role and row["pair_index"] == pair
            )
            for role in (JOINT, B_ONLY)
        )
    ]
    path_by_key = {(row["role"], row["pair_index"]): row for row in paths}
    common_differences = [
        path_by_key[(JOINT, pair)]["protocol_literal_success_rate"]
        - path_by_key[(B_ONLY, pair)]["protocol_literal_success_rate"]
        for pair in common_pairs
    ]
    first_success = {
        role: original["by_role"][role]["success_count"] for role in (JOINT, B_ONLY)
    }
    literal_success = {
        role: by_role[role]["protocol_literal_success_count"] for role in (JOINT, B_ONLY)
    }
    result: dict[str, Any] = {
        "status": "PROTOCOL_CONFORMANCE_CORRIGENDUM_EXPLORATORY_GUARD_FAILED",
        "artifact_kind": "DIT_V22_REPAIRABILITY_PROTOCOL_CONFORMANCE_ANALYSIS_V2",
        "lineage": {
            **lineage,
            "delivery_identity_sha256": delivery["identity_sha256"],
            "mapping_identity_sha256": mapping["identity_sha256"],
            "original_analysis_identity_sha256": original["identity_sha256"],
            "reviewer_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in review_paths
            ],
        },
        "correction": {
            "problem": (
                "The frozen v1 analyzer substituted 2/3 overall preference for the upstream "
                "requirement that visible blur/fusion be clearly reduced."
            ),
            "protocol_conformance_rule": (
                "opportunity AND 2/3 within-reviewer baseline_blur=true -> fresh_blur=false "
                "AND 2/3 preservation=yes AND NOT 2/3 fresh quality=clear_bad"
            ),
            "resolution_timing_limit": (
                "The source protocol predates outputs, but this exact vote-level resolution "
                "was implemented after unblinding; treat it as a corrigendum, not a fresh "
                "confirmatory test."
            ),
            "v1_success_counts": first_success,
            "protocol_literal_success_counts": literal_success,
            "v1_only_false_success_counts": {
                role: first_success[role] - literal_success[role] for role in (JOINT, B_ONLY)
            },
            "literal_only_success_count": 0,
        },
        "design": {
            "paths_per_role": 8,
            "rollback_steps": [109, 149],
            "fresh_attempts_per_path_and_step": 4,
            "comparison_count": 128,
            "reviewers_per_comparison": 3,
            "all_attempts_retained": True,
            "FID_Inception_DINO_CLIP_or_endpoint_features_used": False,
        },
        "by_role": by_role,
        "by_role_and_rollback_step": by_role_step,
        "per_path": paths,
        "frozen_opportunity_guard": {
            "minimum_opportunity_paths_per_role": 4,
            "counts": opportunity_paths,
            "satisfied": all(value >= 4 for value in opportunity_paths.values()),
            "consequence": "between-role repairability comparison is inconclusive",
        },
        "matched_path_posthoc_diagnostics": {
            "overall": paired_diagnostic(paths, "protocol_literal_success_rate", 0),
            "step109": paired_diagnostic(
                paths, "step109_protocol_literal_success_rate", 109
            ),
            "step149": paired_diagnostic(
                paths, "step149_protocol_literal_success_rate", 149
            ),
        },
        "common_opportunity_pairs_descriptive": {
            "pair_indices": common_pairs,
            "pair_count": len(common_pairs),
            "joint_minus_B_only_path_rate_differences": common_differences,
            "mean_difference": mean(common_differences),
            "claim_limit": "only paths for which both observational roles had a visible opportunity; n is tiny",
        },
        "topology_sensitivity": {
            "literal_success_count": sum(row["protocol_literal_success"] for row in rows),
            "literal_success_plus_no_fresh_topology_count": sum(
                row["protocol_literal_success_plus_no_fresh_topology_sensitivity"]
                for row in rows
            ),
            "numerically_changes_result": any(
                row["protocol_literal_success"]
                != row["protocol_literal_success_plus_no_fresh_topology_sensitivity"]
                for row in rows
            ),
            "interpretation": (
                "Fresh clear_bad is the primary severe-new-defect gate. An extra any-topology "
                "gate is stricter than the source wording; here it changes zero successes."
            ),
        },
        "decision": {
            "code": "NO_INCREMENTAL_E_REPAIRABILITY_SUPPORT_GUARD_FAILED",
            "interpretation": (
                "The frozen design declares the between-role comparison inconclusive. The "
                "pooled opportunity-conditioned rate is not matched and cannot rescue E; "
                "all four non-tied overall matched-path differences favor B-only, and the "
                "three pairs with opportunity in both roles are negative, tied, negative."
            ),
            "action": (
                "Do not use the current E process as a bad-image or rollback trigger. Retain "
                "its martingale identity as bookkeeping only; any future quality claim needs "
                "a genuinely new internal alternative and disjoint validation."
            ),
        },
        "claim_limits": [
            "The opportunity guard failed: 3/8 joint paths and 6/8 B-only paths.",
            "E group membership is observational; suffix noise alone is randomized.",
            "Paired bootstrap intervals and fair-sign tails are post-hoc stability descriptions, not confirmatory inference.",
            "Unconditional worsening or success differences cannot be interpreted as E causing harm or benefit.",
            "The exact vote-level blur-reduction resolution was implemented after unblinding to conform to the older source literal.",
            "Reviewer instances were score/role blind model agents, not independent human experts.",
            "No causal, deployment, automatic rollback, Ville/TV, FID, or general-quality claim follows.",
        ],
    }
    result["identity_sha256"] = canonical_sha256(result)
    return result, rows


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def publish(args: argparse.Namespace) -> None:
    output = args.output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite corrigendum: {output}")
    result, rows = build_result(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "results.json", result)
        columns = (
            "anonymous_pair_id",
            "role",
            "pair_index",
            "global_seed",
            "rollback_sampling_step",
            "fresh_attempt",
            "baseline_repair_opportunity_2of3",
            "blur_reduced_within_reviewer_2of3",
            "semantic_preservation_2of3_yes",
            "fresh_failure_2of3_clear_bad",
            "fresh_topology_2of3",
            "protocol_literal_success",
        )
        with (staging / "protocol_conformance_comparisons.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row[name] for name in columns})
        shutil.copyfile(Path(__file__).resolve(), staging / "analyzer_source.py")
        files = [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(staging.iterdir())
            if path.is_file()
        ]
        manifest: dict[str, Any] = {
            "status": "complete",
            "artifact_kind": "DIT_V22_REPAIRABILITY_PROTOCOL_CONFORMANCE_ANALYSIS_V2",
            "result_identity_sha256": result["identity_sha256"],
            "files": files,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "result_identity_sha256": result["identity_sha256"],
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        write_json(staging / "completion.json", completion)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def self_test() -> None:
    reviews = [
        {
            "baseline_quality": "mild_or_uncertain",
            "fresh_quality": "clean_good",
            "baseline_blur": True,
            "fresh_blur": False,
            "baseline_topology": False,
            "fresh_topology": False,
            "preservation": "yes",
            "preferred_fresh": True,
        }
        for _ in range(3)
    ]
    row = consensus_row({"role": JOINT, "pair_index": 0}, reviews)
    if not row["protocol_literal_success"]:
        raise AssertionError("literal success construction changed")
    reviews[1]["fresh_blur"] = True
    reviews[2]["fresh_blur"] = True
    row = consensus_row({"role": JOINT, "pair_index": 0}, reviews)
    if row["protocol_literal_success"]:
        raise AssertionError("blur reduction vote is no longer required")
    print("repairability protocol-conformance analyzer self-test passed")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery", type=Path, default=DEFAULT_DELIVERY)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--original-analysis", type=Path, default=DEFAULT_ORIGINAL_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
