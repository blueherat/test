#!/usr/bin/env python3
"""Lock three evidence-blind reviews for the remaining targeted DiT 60 images."""

from __future__ import annotations

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
    reviewer: ROOT / f"experiments/annotations/dit_targeted60_review_{reviewer}_v1_draft.json"
    for reviewer in "DEF"
}
BASELINE_ROOT = DATA_ROOT / "cross_scale_evidence/dit_imagenet256"
GRID_ROOT = BASELINE_ROOT / "targeted_scan_v1_review"
DEFAULT_OUTPUT = ROOT / "experiments/annotations/dit_targeted60_consensus_lock_v1"
ORDERED_CLASSES = (207, 340, 354, 366, 444, 602, 795, 981)
TARGET_CLASSES = (366, 444, 981)
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
EXPECTED_COUNTS = {"clear_bad": 2, "clean_good": 49, "mild_or_disputed": 9}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def expected_keys() -> set[str]:
    return {
        f"class{class_id:04d}_seed{seed}"
        for class_id in TARGET_CLASSES
        for seed in SEEDS
    }


def normalize_review(reviewer: str, document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    declaration = document.get("declarations", document)
    if (
        declaration.get("reviewer") != reviewer
        or declaration.get("independent_review") is not True
        or declaration.get("metrics_seen") is not False
        or declaration.get("trajectories_seen") is not False
        or declaration.get("signals_summaries_or_research_hypotheses_seen") is not False
        or declaration.get("other_reviews_seen") is not False
        or declaration.get("single_reviewer_draft") is not True
    ):
        raise RuntimeError(f"review {reviewer} evidence-blind declaration is invalid")
    annotations = document.get("annotations")
    if isinstance(annotations, list):
        rows = {
            f"class{int(row['class_id']):04d}_seed{int(row['seed'])}": row
            for row in annotations
        }
        if len(rows) != len(annotations):
            raise RuntimeError(f"review {reviewer} contains duplicate list rows")
    elif isinstance(annotations, dict):
        rows = annotations
    else:
        raise RuntimeError(f"review {reviewer} annotations have invalid type")
    if set(rows) != expected_keys():
        raise RuntimeError(f"review {reviewer} does not cover the exact 60 keys")
    for key, row in rows.items():
        expected_class, expected_seed = key.split("_seed")
        flags = row.get("flags")
        score = row.get("score")
        if (
            row.get("class_id") != int(expected_class.removeprefix("class"))
            or row.get("seed") != int(expected_seed)
            or (score != "U" and (type(score) is not int or score not in range(4)))
            or not isinstance(flags, list)
            or not flags
            or len(flags) != len(set(flags))
            or not set(flags).issubset(ALLOWED_FLAGS)
            or ("none" in flags and flags != ["none"])
            or not isinstance(row.get("reason"), str)
            or not row["reason"].strip()
        ):
            raise RuntimeError(f"invalid review {reviewer} row: {key}")
    return rows


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


def consensus_label(scores: list[int | str]) -> str:
    if sum(isinstance(score, int) and score >= 2 for score in scores) >= 2:
        return "clear_bad"
    if sum(score == 0 for score in scores) >= 2:
        return "clean_good"
    return "mild_or_disputed"


def build_rows(
    reviews: dict[str, dict[str, dict[str, Any]]], baseline_root: Path
) -> list[dict[str, Any]]:
    class_position = {class_id: index for index, class_id in enumerate(ORDERED_CLASSES)}
    result = []
    for class_id in TARGET_CLASSES:
        for seed in SEEDS:
            key = f"class{class_id:04d}_seed{seed}"
            scores = {reviewer: reviews[reviewer][key]["score"] for reviewer in "DEF"}
            flags = {reviewer: reviews[reviewer][key]["flags"] for reviewer in "DEF"}
            reasons = {reviewer: reviews[reviewer][key]["reason"] for reviewer in "DEF"}
            vote_counts = {
                flag: sum(flag in flags[reviewer] for reviewer in "DEF")
                for flag in sorted(ALLOWED_FLAGS - {"none"})
            }
            image_path = (
                baseline_root
                / f"targeted_scan_v1_seed{seed}"
                / "images"
                / f"{class_position[class_id]:02d}_class{class_id:04d}.png"
            )
            label = consensus_label(list(scores.values()))
            result.append(
                {
                    "sample_key": key,
                    "class_id": class_id,
                    "seed": seed,
                    "review_scores": scores,
                    "review_flags": flags,
                    "review_reasons": reasons,
                    "flag_vote_counts": vote_counts,
                    "majority_flags": [flag for flag, count in vote_counts.items() if count >= 2],
                    "clear_bad_vote_count": sum(
                        isinstance(score, int) and score >= 2 for score in scores.values()
                    ),
                    "clean_good_vote_count": sum(score == 0 for score in scores.values()),
                    "primary_label": label,
                    "binary_primary_included": label in {"clear_bad", "clean_good"},
                    "native_image": inspect_png(image_path),
                }
            )
    counts = {label: sum(row["primary_label"] == label for row in result) for label in EXPECTED_COUNTS}
    if len(result) != 60 or counts != EXPECTED_COUNTS:
        raise RuntimeError(f"consensus counts changed: {counts} != {EXPECTED_COUNTS}")
    return result


def inspect_grids(root: Path) -> list[dict[str, Any]]:
    records = []
    for view in ("native", "nearest", "smooth"):
        for class_id in TARGET_CLASSES:
            path = root / view / f"class{class_id:04d}.png"
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


def publish(output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise RuntimeError(f"refusing to overwrite consensus lock: {output}")
    reviews = {
        reviewer: normalize_review(reviewer, load_json(path))
        for reviewer, path in REVIEW_PATHS.items()
    }
    rows = build_rows(reviews, BASELINE_ROOT)
    counts = {label: sum(row["primary_label"] == label for row in rows) for label in EXPECTED_COUNTS}
    consensus: dict[str, Any] = {
        "schema_version": 1,
        "status": "LOCKED_BEFORE_ANY_TARGETED60_TRAJECTORY_METRIC_JOIN",
        "rule": {
            "clear_bad": "at least two of three independent scores are 2 or 3",
            "clean_good": "at least two of three independent scores are 0",
            "mild_or_disputed": "neither majority rule holds",
            "metric_or_signal_used": False,
        },
        "counts": counts,
        "rows": rows,
    }
    consensus["identity_sha256"] = canonical_sha256(consensus)
    grids = inspect_grids(GRID_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        write_json(staging / "consensus_locked.json", consensus)
        for reviewer, source in REVIEW_PATHS.items():
            shutil.copy2(source, staging / f"review_{reviewer}_locked.json")
        shutil.copy2(Path(__file__).resolve(), staging / "locker_source.py")
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": consensus["status"],
            "consensus_file_sha256": sha256_file(staging / "consensus_locked.json"),
            "consensus_identity_sha256": consensus["identity_sha256"],
            "review_files": {
                reviewer: {
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "locked_sha256": sha256_file(staging / f"review_{reviewer}_locked.json"),
                }
                for reviewer, source in REVIEW_PATHS.items()
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
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def main() -> int:
    output = publish()
    consensus = load_json(output / "consensus_locked.json")
    print(json.dumps({"output": str(output), "counts": consensus["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
