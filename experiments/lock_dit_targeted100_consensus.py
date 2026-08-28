#!/usr/bin/env python3
"""Lock three evidence-blind reviews for the targeted DiT 100-image pilot.

This script reads only the three review drafts and the reviewed baseline image
artifacts.  It never reads a trajectory, feature, score, or research summary.
The immutable output binds every consensus row to one native PNG, all reviewed
native/nearest/smooth grids, the three review files, and this source snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("EQVAE_DATA_ROOT", "/data/users/zhoushunyu/eqvae"))
REVIEW_PATHS = {
    "A": ROOT / "experiments/annotations/dit_targeted100_review_A_v1_draft.json",
    "B": ROOT / "experiments/annotations/dit_targeted100_review_B_v1_draft.json",
    "C": ROOT / "experiments/annotations/dit_targeted100_review_C_v1_draft.json",
}
BASELINE_ROOT = DATA_ROOT / "cross_scale_evidence/dit_imagenet256"
GRID_ROOT = BASELINE_ROOT / "targeted_scan_v1_review"
DEFAULT_OUTPUT = ROOT / "experiments/annotations/dit_targeted100_consensus_lock_v1"
ORDERED_CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
TARGET_CLASSES = (207, 340, 354, 602, 795)
SEEDS = tuple(range(10, 30))
ALLOWED_FLAGS = {
    "none",
    "global_blur",
    "local_blur",
    "fusion_duplication",
    "limb_object_misalignment",
    "topology_attachment",
    "texture_break",
}
EXPECTED_COUNTS = {"clear_bad": 10, "clean_good": 69, "mild_or_disputed": 21}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def expected_keys() -> set[str]:
    return {
        f"class{class_id:04d}_seed{seed}"
        for class_id in TARGET_CLASSES
        for seed in SEEDS
    }


def validate_declaration(reviewer: str, document: dict[str, Any]) -> None:
    if document.get("independent_review") is not True:
        raise RuntimeError(f"review {reviewer} lacks independent-review declaration")
    if reviewer == "A":
        valid = (
            document.get("reviewer") == "A"
            and document.get("metrics_seen") is False
            and document.get("signals_summaries_or_research_hypotheses_seen") is False
            and document.get("single_reviewer_draft") is True
        )
    elif reviewer == "B":
        valid = (
            document.get("reviewer_id") == "B"
            and document.get("metrics_viewed") is False
            and document.get("review_status") == "single_reviewer_draft"
        )
    else:
        valid = (
            document.get("reviewer") == "C"
            and document.get("metrics_seen") is False
            and document.get("other_reviews_seen") is False
            and document.get("single_reviewer_draft") is True
        )
    if not valid:
        raise RuntimeError(f"review {reviewer} evidence-blind declaration is invalid")


def load_reviews(paths: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    expected = expected_keys()
    reviews: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"review file is missing or indirect: {path}")
        document = load_json(path)
        validate_declaration(reviewer, document)
        annotations = document.get("annotations")
        if not isinstance(annotations, dict) or set(annotations) != expected:
            raise RuntimeError(f"review {reviewer} does not cover the exact 100 keys")
        for key, row in annotations.items():
            if not isinstance(row, dict):
                raise RuntimeError(f"review {reviewer} row is not an object: {key}")
            class_id = row.get("class_id")
            seed = row.get("seed")
            score = row.get("score")
            flags = row.get("flags")
            reason = row.get("reason")
            expected_class, expected_seed = key.split("_seed")
            class_matches = class_id in {
                expected_class,
                int(expected_class.removeprefix("class")),
            }
            if (
                not class_matches
                or type(seed) is not int
                or seed != int(expected_seed)
                or (score != "U" and (type(score) is not int or score not in range(4)))
                or not isinstance(flags, list)
                or not flags
                or len(flags) != len(set(flags))
                or not set(flags).issubset(ALLOWED_FLAGS)
                or ("none" in flags and flags != ["none"])
                or not isinstance(reason, str)
                or not reason.strip()
            ):
                raise RuntimeError(f"invalid review {reviewer} row: {key}")
        reviews[reviewer] = annotations
    return reviews


def inspect_png(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"PNG is missing or indirect: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGB" or image.size != (256, 256):
            raise RuntimeError(f"unexpected native PNG properties: {path}")
        pixels = image.tobytes()
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
        "mode": "RGB",
        "size": [256, 256],
    }


def label_from_scores(scores: list[int | str]) -> str:
    numeric = [score for score in scores if isinstance(score, int)]
    if sum(score >= 2 for score in numeric) >= 2:
        return "clear_bad"
    if sum(score == 0 for score in numeric) >= 2:
        return "clean_good"
    return "mild_or_disputed"


def build_rows(
    reviews: dict[str, dict[str, dict[str, Any]]], baseline_root: Path
) -> list[dict[str, Any]]:
    class_position = {class_id: index for index, class_id in enumerate(ORDERED_CLASSES)}
    rows = []
    for class_id in TARGET_CLASSES:
        for seed in SEEDS:
            key = f"class{class_id:04d}_seed{seed}"
            scores = {reviewer: reviews[reviewer][key]["score"] for reviewer in "ABC"}
            flags = {reviewer: reviews[reviewer][key]["flags"] for reviewer in "ABC"}
            reasons = {reviewer: reviews[reviewer][key]["reason"] for reviewer in "ABC"}
            flag_votes = {
                flag: sum(flag in flags[reviewer] for reviewer in "ABC")
                for flag in sorted(ALLOWED_FLAGS - {"none"})
            }
            majority_flags = [flag for flag, votes in flag_votes.items() if votes >= 2]
            image_path = (
                baseline_root
                / f"targeted_scan_v1_seed{seed}"
                / "images"
                / f"{class_position[class_id]:02d}_class{class_id:04d}.png"
            )
            label = label_from_scores(list(scores.values()))
            rows.append(
                {
                    "sample_key": key,
                    "class_id": class_id,
                    "seed": seed,
                    "review_scores": scores,
                    "review_flags": flags,
                    "review_reasons": reasons,
                    "flag_vote_counts": flag_votes,
                    "majority_flags": majority_flags,
                    "clear_bad_vote_count": sum(
                        isinstance(score, int) and score >= 2 for score in scores.values()
                    ),
                    "clean_good_vote_count": sum(score == 0 for score in scores.values()),
                    "primary_label": label,
                    "binary_primary_included": label in {"clear_bad", "clean_good"},
                    "native_image": inspect_png(image_path),
                }
            )
    counts = {label: sum(row["primary_label"] == label for row in rows) for label in EXPECTED_COUNTS}
    if counts != EXPECTED_COUNTS or len(rows) != 100:
        raise RuntimeError(f"locked consensus counts changed: {counts} != {EXPECTED_COUNTS}")
    return rows


def inspect_grids(grid_root: Path) -> list[dict[str, Any]]:
    records = []
    for view in ("native", "nearest", "smooth"):
        for class_id in TARGET_CLASSES:
            path = grid_root / view / f"class{class_id:04d}.png"
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"review grid is missing or indirect: {path}")
            with Image.open(path) as image:
                image.load()
                records.append(
                    {
                        "view": view,
                        "class_id": class_id,
                        "path": str(path.resolve()),
                        "file_sha256": sha256_file(path),
                        "mode": image.mode,
                        "size": list(image.size),
                    }
                )
    return records


def publish(args: argparse.Namespace) -> Path:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite consensus lock: {args.output_dir}")
    reviews = load_reviews(args.review_paths)
    rows = build_rows(reviews, args.baseline_root)
    grids = inspect_grids(args.grid_root)
    counts = {label: sum(row["primary_label"] == label for row in rows) for label in EXPECTED_COUNTS}
    consensus = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_ANY_TARGETED100_TRAJECTORY_METRIC_JOIN",
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

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output_dir.name}.staging.", dir=args.output_dir.parent)
    )
    try:
        write_json(staging / "consensus_locked.json", consensus)
        for reviewer, source in args.review_paths.items():
            shutil.copyfile(source, staging / f"review_{reviewer}_locked.json")
        shutil.copyfile(Path(__file__).resolve(), staging / "locker_source.py")
        manifest = {
            "schema_version": 1,
            "experiment": "dit_targeted100_visual_consensus_lock_v1",
            "status": consensus["status"],
            "consensus_file_sha256": sha256_file(staging / "consensus_locked.json"),
            "consensus_identity_sha256": consensus["identity_sha256"],
            "review_files": {
                reviewer: {
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "locked_sha256": sha256_file(staging / f"review_{reviewer}_locked.json"),
                }
                for reviewer, source in args.review_paths.items()
            },
            "review_grids": grids,
            "native_image_binding_sha256": canonical_sha256(
                [row["native_image"] for row in rows]
            ),
            "locker_source_sha256": sha256_file(staging / "locker_source.py"),
            "counts": counts,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "consensus_file_sha256": manifest["consensus_file_sha256"],
            "consensus_identity_sha256": consensus["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "manifest_identity_sha256": manifest["identity_sha256"],
            "locked_row_count": len(rows),
        }
        completion["payload_sha256"] = canonical_sha256(completion)
        write_json(staging / "completion.json", completion)
        staging.rename(args.output_dir)
        return args.output_dir
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    assert label_from_scores([2, 2, 0]) == "clear_bad"
    assert label_from_scores([0, 0, 3]) == "clean_good"
    assert label_from_scores([0, 1, 2]) == "mild_or_disputed"
    assert label_from_scores(["U", 2, 2]) == "clear_bad"
    assert len(expected_keys()) == 100
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--grid-root", type=Path, default=GRID_ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.output_dir = args.output_dir.expanduser().resolve()
    args.baseline_root = args.baseline_root.expanduser().resolve()
    args.grid_root = args.grid_root.expanduser().resolve()
    args.review_paths = {key: path.resolve() for key, path in REVIEW_PATHS.items()}
    output = publish(args)
    print(json.dumps({"output": str(output), "counts": EXPECTED_COUNTS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
