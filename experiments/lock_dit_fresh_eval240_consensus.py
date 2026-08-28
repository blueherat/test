#!/usr/bin/env python3
"""Lock three endpoint-only reviews before any fresh label/score join."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
BLIND_PACK = (
    DATA_ROOT
    / "cross_scale_evidence/bad_good_metric_confirmation_v1/"
    "blind_review_evaluation_seed050_129_v1"
)
BLIND_PACK_IDENTITY = "59791e2fe6b319bb312060991efed01e6b1e9d5ad608e8a5b38e77c6f4a241ff"
CANDIDATE_PROTOCOL_IDENTITY = (
    "198a82a7c8a0ab79d901c76a5c810f4a40889604a66f18e995d0699f73c12bce"
)
CANDIDATE_LOCK = ROOT / "experiments/locks/dit_bad_good_candidate_confirmation_lock_v5"
REVIEW_PATHS = {
    reviewer: ROOT
    / f"experiments/annotations/dit_fresh_eval240_review_{reviewer}_v1_draft.json"
    for reviewer in "GHI"
}
DEFAULT_OUTPUT = ROOT / "experiments/annotations/dit_fresh_eval240_consensus_lock_v1"
CLASSES = (207, 602, 795)
SEEDS = tuple(range(50, 130))
ALLOWED_FLAGS = {
    "none",
    "global_blur",
    "local_blur",
    "fusion_duplication",
    "topology_attachment",
    "limb_object_misalignment",
    "texture_break",
    "other",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def sample_key(class_id: int, seed: int) -> str:
    return f"class{class_id:04d}_seed{seed:03d}"


def expected_keys() -> set[str]:
    return {sample_key(class_id, seed) for class_id in CLASSES for seed in SEEDS}


def validate_blind_pack(root: Path) -> dict[str, dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"blind pack is missing or indirect: {root}")
    contract_path = root / "review_contract.json"
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        contract.get("identity_sha256") != BLIND_PACK_IDENTITY
        or completion.get("identity_sha256") != BLIND_PACK_IDENTITY
        or completion.get("manifest_sha256") != sha256_file(manifest_path)
        or completion.get("endpoint_count") != 240
        or completion.get("grid_cell_count") != 240
        or completion.get("calibration_seed_count") != 0
        or manifest.get("status") != "complete"
        or manifest.get("identity", {}).get("cartesian_product")
        != {"class_count": 3, "endpoint_count": 240, "exact": True, "seed_count": 80}
        or contract.get("automatic_quality_scoring") is not False
        or contract.get("automatic_ranking_or_selection") is not False
    ):
        raise RuntimeError("blind review pack identity or evidence contract changed")
    rows = manifest.get("source_endpoints")
    if not isinstance(rows, list) or len(rows) != 240:
        raise RuntimeError("blind pack lacks exact endpoint records")
    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        class_id = row.get("class_id")
        seed = row.get("seed")
        if type(class_id) is not int or type(seed) is not int:
            raise RuntimeError("blind pack endpoint key is malformed")
        key = sample_key(class_id, seed)
        copied = root / str(row.get("copied_endpoint_relative_path"))
        if (
            key in by_key
            or class_id not in CLASSES
            or seed not in SEEDS
            or not copied.is_file()
            or copied.is_symlink()
            or copied.stat().st_size != row.get("source_endpoint_bytes")
            or sha256_file(copied) != row.get("source_endpoint_sha256")
            or row.get("source_endpoint_size") != [256, 256]
            or row.get("source_endpoint_mode") != "RGB"
        ):
            raise RuntimeError(f"blind pack endpoint binding failed: {key}")
        by_key[key] = row
    if set(by_key) != expected_keys():
        raise RuntimeError("blind pack is not the exact evaluation Cartesian product")
    return by_key


def validate_candidate_lock(root: Path) -> None:
    manifest_path = root / "manifest.json"
    protocol_path = root / "candidate_protocol.json"
    manifest = load_json(manifest_path)
    completion = load_json(root / "completion.json")
    protocol = load_json(protocol_path)
    if (
        manifest.get("status") != "complete"
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("protocol_file_sha256") != sha256_file(protocol_path)
        or completion.get("protocol_identity_sha256") != CANDIDATE_PROTOCOL_IDENTITY
        or protocol.get("identity_sha256") != CANDIDATE_PROTOCOL_IDENTITY
    ):
        raise RuntimeError("candidate v5 lock changed")


def validate_reviews(paths: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    keys = expected_keys()
    reviews: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"review is missing or indirect: {path}")
        document = load_json(path)
        if (
            document.get("reviewer") != reviewer
            or document.get("independent_review") is not True
            or document.get("metrics_seen") is not False
            or document.get("trajectories_seen") is not False
            or document.get("signals_summaries_or_research_hypotheses_seen") is not False
            or document.get("other_reviews_seen") is not False
            or document.get("single_reviewer_draft") is not True
            or document.get("blind_pack_identity_sha256") != BLIND_PACK_IDENTITY
        ):
            raise RuntimeError(f"review {reviewer} has an invalid blinding declaration")
        annotations = document.get("annotations")
        if not isinstance(annotations, dict) or set(annotations) != keys:
            raise RuntimeError(f"review {reviewer} does not cover the exact 240 samples")
        for key, row in annotations.items():
            expected_class = int(key[5:9])
            expected_seed = int(key[-3:])
            flags = row.get("flags")
            if (
                row.get("class_id") != expected_class
                or row.get("seed") != expected_seed
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
                raise RuntimeError(f"invalid review {reviewer} annotation: {key}")
        reviews[reviewer] = annotations
    return reviews


def label_from_scores(scores: list[int]) -> str:
    if sum(score >= 2 for score in scores) >= 2:
        return "clear_bad"
    if sum(score == 0 for score in scores) >= 2:
        return "clean_good"
    return "mild_or_disputed"


def build_consensus(
    reviews: dict[str, dict[str, dict[str, Any]]],
    endpoints: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for class_id in CLASSES:
        for seed in SEEDS:
            key = sample_key(class_id, seed)
            review_scores = {reviewer: reviews[reviewer][key]["score"] for reviewer in "GHI"}
            review_flags = {reviewer: reviews[reviewer][key]["flags"] for reviewer in "GHI"}
            review_reasons = {reviewer: reviews[reviewer][key]["reason"] for reviewer in "GHI"}
            flag_vote_counts = {
                flag: sum(flag in review_flags[reviewer] for reviewer in "GHI")
                for flag in sorted(ALLOWED_FLAGS - {"none"})
            }
            label = label_from_scores(list(review_scores.values()))
            endpoint = endpoints[key]
            rows.append(
                {
                    "sample_key": key,
                    "class_id": class_id,
                    "seed": seed,
                    "global_seed": seed,
                    "review_scores": review_scores,
                    "review_flags": review_flags,
                    "review_reasons": review_reasons,
                    "flag_vote_counts": flag_vote_counts,
                    "majority_flags": [
                        flag for flag, votes in flag_vote_counts.items() if votes >= 2
                    ],
                    "clear_bad_vote_count": sum(score >= 2 for score in review_scores.values()),
                    "clean_good_vote_count": sum(score == 0 for score in review_scores.values()),
                    "primary_label": label,
                    "binary_primary_included": label in {"clear_bad", "clean_good"},
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
        for label in ("clear_bad", "clean_good", "mild_or_disputed")
    }
    consensus: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_WITHOUT_SCORE_OR_ALERT_ACCESS_BEFORE_ANY_LABEL_SCORE_JOIN",
        "blind_pack_identity_sha256": BLIND_PACK_IDENTITY,
        "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
        "blinding_audit": {
            "reviewer_count": 3,
            "endpoint_only_review": True,
            "metric_values_visible_to_reviewers": False,
            "alert_decisions_visible_to_reviewers": False,
            "trajectories_visible_to_reviewers": False,
            "other_reviews_visible_to_each_reviewer": False,
            "labels_locked_before_score_join": True,
        },
        "rule": {
            "clear_bad": "at least two of three independent scores are 2 or 3",
            "clean_good": "at least two of three independent scores are 0",
            "mild_or_disputed": "neither clear-bad nor clean-good majority",
            "metric_or_signal_used": False,
        },
        "counts": counts,
        "rows": rows,
    }
    consensus["identity_sha256"] = canonical_sha256(consensus)
    return consensus


def publish(blind_pack: Path, review_paths: dict[str, Path], output: Path) -> Path:
    validate_candidate_lock(CANDIDATE_LOCK)
    endpoints = validate_blind_pack(blind_pack)
    reviews = validate_reviews(review_paths)
    consensus = build_consensus(reviews, endpoints)
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite consensus lock: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "consensus_locked.json", consensus)
        for reviewer, source in review_paths.items():
            shutil.copy2(source, staging / f"review_{reviewer}_locked.json")
        shutil.copy2(Path(__file__).resolve(), staging / "locker_source.py")
        members = [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(staging.iterdir())
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "consensus_identity_sha256": consensus["identity_sha256"],
            "blind_pack_identity_sha256": BLIND_PACK_IDENTITY,
            "candidate_protocol_identity_sha256": CANDIDATE_PROTOCOL_IDENTITY,
            "blind_pack_manifest_sha256": sha256_file(blind_pack / "manifest.json"),
            "counts": consensus["counts"],
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
                "consensus_file_sha256": sha256_file(staging / "consensus_locked.json"),
                "consensus_identity_sha256": consensus["identity_sha256"],
                "locked_row_count": 240,
            },
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def self_test() -> None:
    assert len(expected_keys()) == 240
    assert label_from_scores([2, 2, 0]) == "clear_bad"
    assert label_from_scores([0, 0, 3]) == "clean_good"
    assert label_from_scores([0, 1, 2]) == "mild_or_disputed"
    print("self-test passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-pack", type=Path, default=BLIND_PACK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    output = publish(
        args.blind_pack.expanduser().resolve(),
        {reviewer: path.resolve() for reviewer, path in REVIEW_PATHS.items()},
        args.output.expanduser().absolute(),
    )
    consensus = load_json(output / "consensus_locked.json")
    print(json.dumps({"output": str(output), "counts": consensus["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
