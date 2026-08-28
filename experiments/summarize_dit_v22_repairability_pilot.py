#!/usr/bin/env python3
"""Post-hoc matched-pair summary of the frozen DiT v2.2 repairability pilot.

This script does not read images, reviewer-side labels, the sealed mapping, or
any FID/embedding feature.  It authenticates the completed frozen analysis and
computes transparent path-level paired diagnostics that were not part of the
frozen primary analyzer.  The output is descriptive and cannot repair a failed
frozen opportunity guard or create intervention authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ANALYSIS = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_blind_review_v1_analysis"
)
DEFAULT_OUTPUT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_repairability_pilot_v1_2_matched_pair_summary_v1.json"
)
EXPECTED_RESULT_ID = "055768a7e7b519c79e037b8649744a1cc9e055a033d9c73c1be6596da3da3005"
JOINT = "joint_E_and_B"
B_ONLY = "B_only_exact_schedule_B_matched_control"
BOOTSTRAP_SEED = 2026082813
BOOTSTRAP_REPLICATES = 100_000


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


def validate_self_hash(value: dict[str, Any], key: str, context: str) -> None:
    observed = value.get(key)
    payload = dict(value)
    payload.pop(key, None)
    if not isinstance(observed, str) or canonical_sha256(payload) != observed:
        raise RuntimeError(f"invalid {key}: {context}")


def authenticate(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid analysis root: {root}")
    manifest = load_json(root / "manifest.json")
    validate_self_hash(manifest, "identity_sha256", "manifest")
    if (
        manifest.get("status") != "complete"
        or manifest.get("artifact_kind")
        != "DIT_V22_REPAIRABILITY_BLIND_REVIEW_ANALYSIS_V1"
    ):
        raise RuntimeError("wrong or incomplete analysis artifact")
    records = {row.get("name"): row for row in manifest.get("files", [])}
    if set(records) != {"results.json", "joined_private_comparisons.csv"}:
        raise RuntimeError("analysis member axis changed")
    for name, record in records.items():
        path = root / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise RuntimeError(f"analysis member changed: {name}")
    result = load_json(root / "results.json")
    validate_self_hash(result, "identity_sha256", "results")
    if (
        result.get("identity_sha256") != EXPECTED_RESULT_ID
        or result.get("identity_sha256") != manifest.get("result_identity_sha256")
        or result.get("status")
        != "RETROSPECTIVE_REPAIRABILITY_BLIND_REVIEW_EXPLORATORY_ONLY"
        or result.get("design", {}).get("no_FID_features_or_endpoint_scores") is not True
    ):
        raise RuntimeError("frozen result identity or scientific contract changed")
    completion = load_json(root / "completion.json")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256")
        != sha256_file(root / "manifest.json")
    ):
        raise RuntimeError("analysis completion binding changed")
    return result


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


def paired_summary(differences: list[float], label: str, seed_offset: int) -> dict[str, Any]:
    if len(differences) != 8:
        raise RuntimeError(f"expected eight matched differences for {label}")
    observed = sum(differences) / len(differences)
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    leave_one_out = [
        sum(value for index, value in enumerate(differences) if index != omitted)
        / (len(differences) - 1)
        for omitted in range(len(differences))
    ]
    positives = sum(value > 0 for value in differences)
    negatives = sum(value < 0 for value in differences)
    non_ties = positives + negatives
    return {
        "estimand": label,
        "matched_pair_differences": differences,
        "mean_difference": observed,
        "positive_negative_tie_counts": {
            "joint_better": positives,
            "joint_worse": negatives,
            "tie": len(differences) - non_ties,
        },
        "one_sided_exact_sign_p_joint_better": binomial_upper_tail(positives, non_ties),
        "one_sided_exact_sign_p_joint_worse": binomial_upper_tail(negatives, non_ties),
        "paired_path_bootstrap": {
            "seed": BOOTSTRAP_SEED + seed_offset,
            "replicates": BOOTSTRAP_REPLICATES,
            "percentile_95_CI": [quantile(draws, 0.025), quantile(draws, 0.975)],
        },
        "leave_one_pair_out_mean_range": [min(leave_one_out), max(leave_one_out)],
    }


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    paths = result.get("per_path_eight_attempt_rates")
    if not isinstance(paths, list) or len(paths) != 16:
        raise RuntimeError("path axis changed")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in paths:
        key = (str(row["role"]), int(row["pair_index"]))
        if key in indexed:
            raise RuntimeError("duplicate role/pair path")
        indexed[key] = row
    expected = {(role, pair) for role in (JOINT, B_ONLY) for pair in range(8)}
    if set(indexed) != expected:
        raise RuntimeError("matched role/pair matrix changed")

    def differences(field: str) -> list[float]:
        return [
            float(indexed[(JOINT, pair)][field])
            - float(indexed[(B_ONLY, pair)][field])
            for pair in range(8)
        ]

    by_role = result["by_role"]
    joint = by_role[JOINT]
    bonly = by_role[B_ONLY]
    joint_opportunities = int(round(joint["baseline_repair_opportunity_rate"] * 64))
    bonly_opportunities = int(round(bonly["baseline_repair_opportunity_rate"] * 64))
    joint_successes = int(joint["success_count"])
    bonly_successes = int(bonly["success_count"])
    if (joint_opportunities, bonly_opportunities, joint_successes, bonly_successes) != (
        23,
        48,
        14,
        30,
    ):
        raise RuntimeError("role opportunity/success totals changed")

    summary: dict[str, Any] = {
        "status": "POSTHOC_MATCHED_PATH_DIAGNOSTIC_NO_INTERVENTION_AUTHORITY",
        "source_result_identity_sha256": result["identity_sha256"],
        "frozen_primary_guard": result["frozen_guard"],
        "matched_path_count_per_role": 8,
        "all_attempts_retained": True,
        "FID_Inception_DINO_CLIP_or_endpoint_features_used": False,
        "paired_path_diagnostics": {
            "overall_success": paired_summary(
                differences("success_rate"),
                "mean path success rate: joint_E_and_B minus matched B-only",
                0,
            ),
            "step109_success": paired_summary(
                differences("step109_success_rate"),
                "mean path step-109 success rate: joint_E_and_B minus matched B-only",
                109,
            ),
            "step149_success": paired_summary(
                differences("step149_success_rate"),
                "mean path step-149 success rate: joint_E_and_B minus matched B-only",
                149,
            ),
            "baseline_opportunity_fraction": paired_summary(
                differences("baseline_opportunity_fraction"),
                "mean path baseline-opportunity fraction: joint_E_and_B minus matched B-only",
                1000,
            ),
        },
        "opportunity_conditioned_descriptive": {
            "joint": {
                "successes": joint_successes,
                "opportunities": joint_opportunities,
                "rate": joint_successes / joint_opportunities,
            },
            "B_only": {
                "successes": bonly_successes,
                "opportunities": bonly_opportunities,
                "rate": bonly_successes / bonly_opportunities,
            },
            "joint_minus_B_only_rate": joint_successes / joint_opportunities
            - bonly_successes / bonly_opportunities,
            "claim_limit": "attempt-level descriptive ratio with repeated comparisons nested in paths",
        },
        "safety_descriptive": {
            "joint_worsening_rate": joint["worsening_rate"],
            "B_only_worsening_rate": bonly["worsening_rate"],
            "joint_minus_B_only_worsening": joint["worsening_rate"]
            - bonly["worsening_rate"],
            "joint_preservation_rate": joint["semantic_preservation_rate"],
            "B_only_preservation_rate": bonly["semantic_preservation_rate"],
        },
        "decision": {
            "code": "NO_INCREMENTAL_E_REPAIRABILITY_SUPPORT_GUARD_FAILED",
            "reason": (
                "The frozen opportunity guard failed (3/8 joint paths versus 6/8 B-only); "
                "among visible opportunities the rates were nearly equal, every non-tied "
                "matched path difference favored B-only, and joint worsening was higher."
            ),
            "scientific_action": (
                "Do not use E as a rollback trigger. Retire its current repairability "
                "interpretation unless a genuinely new, independently frozen mechanism is proposed."
            ),
        },
        "claim_limits": [
            "This summary was specified after the frozen primary analysis completed and is post-hoc descriptive.",
            "The frozen opportunity guard failed, so the pilot cannot estimate a between-role E repairability effect.",
            "Only 8 matched paths per role and two ImageNet classes were studied.",
            "Reviewer instances were score/role blind but were model agents, not independent human experts.",
            "No causal, deployment, Ville/TV, automatic rollback, or general quality claim follows.",
        ],
    }
    summary["identity_sha256"] = canonical_sha256(summary)
    return summary


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().absolute()
    if os.path.lexists(path):
        raise RuntimeError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stdout-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(authenticate(args.analysis))
    if not args.stdout_only:
        write_json_atomic(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
