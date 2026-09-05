#!/usr/bin/env python3
"""Merge globally indexed RAEv2 relative-transport sample shards."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.raev2_training_core import file_sha256  # noqa: E402
from experiments.sample_raev2_relative_transport import BRANCHES, PROTOCOL  # noqa: E402


MERGE_PROTOCOL = "raev2_relative_transport_iteration_merge_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_root = args.shard_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    shard_dirs = sorted(path for path in shard_root.glob("shard*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no shard directories under {shard_root}")
    summaries = [json.loads((path / "summary.json").read_text()) for path in shard_dirs]
    if any(summary.get("protocol") != PROTOCOL for summary in summaries):
        raise ValueError("unexpected shard protocol")
    expected_shards = int(summaries[0]["num_shards"])
    sample_count = int(summaries[0]["global_sample_count"])
    if len(shard_dirs) != expected_shards:
        raise ValueError("shard count is incomplete")
    invariant_keys = ("seed", "global_sample_count", "num_shards", "noise_sha256", "labels_sha256")
    for key in invariant_keys:
        if len({str(summary[key]) for summary in summaries}) != 1:
            raise ValueError(f"shards disagree on {key}")

    ids = np.concatenate([np.load(path / "sample_ids.npy") for path in shard_dirs])
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(sample_count)):
        raise ValueError("sample IDs are incomplete or duplicated")
    output.mkdir(parents=True)
    np.save(output / "sample_ids.npy", ids[order])
    branches = {}
    for name in BRANCHES:
        arrays = []
        for path in shard_dirs:
            with np.load(path / name / "samples.npz") as archive:
                arrays.append(archive[archive.files[0]])
        merged = np.concatenate(arrays, axis=0)[order]
        branch_dir = output / name
        branch_dir.mkdir()
        archive_path = branch_dir / "samples.npz"
        np.savez(archive_path, merged)
        preview = shard_dirs[0] / name / "preview.png"
        if preview.is_file():
            shutil.copy2(preview, branch_dir / "preview.png")
        branches[name] = {
            "samples": len(merged),
            "archive": str(archive_path),
            "archive_sha256": file_sha256(archive_path),
        }
        (branch_dir / "summary.json").write_text(
            json.dumps(branches[name], indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "protocol": MERGE_PROTOCOL,
        "source_protocol": PROTOCOL,
        "shard_root": str(shard_root),
        "shards": [str(path) for path in shard_dirs],
        "samples": sample_count,
        "seed": summaries[0]["seed"],
        "noise_sha256": summaries[0]["noise_sha256"],
        "labels_sha256": summaries[0]["labels_sha256"],
        "branches": branches,
        "complete": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
