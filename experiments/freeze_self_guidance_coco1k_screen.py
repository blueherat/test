#!/usr/bin/env python3
"""Freeze a deterministic 1,000-prompt COCO screen for Self-Guidance.

The 1k cohort is a label-free, hash-ranked subset of the released 5k prompt
file.  Selection never depends on generated images or metrics.  Selected
prompts are written in their original source order so upstream output indices
remain easy to audit; ``manifest.csv`` records the hash rank and four fixed
250-prompt uncertainty blocks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/data/users/zhoushunyu/eqvae/baselines/Self-Guidance/"
    "data/coco/coco_val5000_prompts.txt"
)
DEFAULT_OUTPUT = ROOT / "experiments/locks/self_guidance_sd14_coco1k_screen_v1"
EXPECTED_SOURCE_SHA256 = (
    "97b02328e93d2ced00df0fe9221ca05441eaa48356d62e496f6e053074876837"
)
EXPECTED_SOURCE_LINES = 5000
SELECTION_SALT = "eqvae-self-guidance-coco1k-screen-v1-20260828"
SELECTED = 1000
BLOCKS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_text_fsynced(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_fsynced(path: Path, value: Any) -> None:
    write_text_fsynced(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def run(source: Path, output: Path) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"expected a real released prompt file: {source}")
    observed_source_sha = sha256_file(source)
    if observed_source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("released COCO prompt file hash changed")
    prompts = source.read_text(encoding="utf-8").splitlines()
    if len(prompts) != EXPECTED_SOURCE_LINES or any(not prompt.strip() for prompt in prompts):
        raise RuntimeError("released COCO prompt file line contract changed")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen lock: {output}")

    candidates: list[dict[str, Any]] = []
    for source_index, prompt in enumerate(prompts):
        selection_key = hashlib.sha256(
            f"{SELECTION_SALT}\0{source_index}\0{prompt}".encode("utf-8")
        ).hexdigest()
        candidates.append(
            {
                "source_index": source_index,
                "prompt": prompt,
                "selection_key_sha256": selection_key,
            }
        )
    ranked = sorted(candidates, key=lambda row: row["selection_key_sha256"])
    selected = ranked[:SELECTED]
    if len({row["source_index"] for row in selected}) != SELECTED:
        raise RuntimeError("selection contains duplicate source indices")
    for rank, row in enumerate(selected):
        row["selection_rank"] = rank
        row["uncertainty_block"] = rank // (SELECTED // BLOCKS)
    selected.sort(key=lambda row: row["source_index"])
    for output_index, row in enumerate(selected):
        row["output_index"] = output_index

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        prompt_path = staging / "prompts.txt"
        manifest_path = staging / "manifest.csv"
        write_text_fsynced(
            prompt_path, "".join(f"{row['prompt']}\n" for row in selected)
        )
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "output_index",
                    "source_index",
                    "selection_rank",
                    "uncertainty_block",
                    "selection_key_sha256",
                    "prompt",
                ),
            )
            writer.writeheader()
            writer.writerows(selected)
            handle.flush()
            os.fsync(handle.fileno())
        block_counts = {
            str(block): sum(row["uncertainty_block"] == block for row in selected)
            for block in range(BLOCKS)
        }
        lock: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": "SELF_GUIDANCE_SD14_COCO1K_SCREEN_LOCK_V1",
            "selection_is_label_and_image_free": True,
            "selection_rule": (
                "take the 1000 lexicographically smallest SHA256(salt\\0source_index"
                "\\0prompt), then emit in source-index order"
            ),
            "selection_salt": SELECTION_SALT,
            "source": {
                "path": str(source),
                "sha256": observed_source_sha,
                "line_count": len(prompts),
            },
            "selected_count": len(selected),
            "uncertainty_blocks": {
                "count": BLOCKS,
                "assignment": "selection_rank // 250",
                "sizes": block_counts,
            },
            "files": {
                "prompts.txt": sha256_file(prompt_path),
                "manifest.csv": sha256_file(manifest_path),
            },
            "comparison": {
                "baseline": "SD1.4 Euler50 CFG7.5 PAG0.3 SG-prev scale 0",
                "method": "SD1.4 Euler50 CFG7.5 PAG0.3 SG-prev scale 3",
                "paired_seed": 0,
                "only_primary_arm_difference": "self_guidance_scale 0 versus 3",
            },
            "decision_rule": {
                "primary": "delta_FID_1k = FID_method - FID_baseline",
                "green": (
                    "delta <= -0.25, at least 3/4 fixed blocks negative, and paired "
                    "80% bootstrap CI upper bound < 0"
                ),
                "yellow": (
                    "delta < 0 and at least 3/4 blocks negative, but magnitude < 0.25 "
                    "or the 80% CI crosses 0; rerun one untouched fresh 1k"
                ),
                "red": "delta >= 0 or at most 2/4 fixed blocks negative",
                "paper_comparison_boundary": (
                    "local 1k absolute FID is never compared directly with paper 5k/30k/50k FID"
                ),
            },
        }
        lock["identity_sha256"] = canonical_sha256(lock)
        write_json_fsynced(staging / "lock.json", lock)
        descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(staging, output)
    except BaseException:
        for child in staging.iterdir():
            child.unlink()
        staging.rmdir()
        raise
    print(json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.source, arguments.output)
