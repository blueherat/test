#!/usr/bin/env python3
"""Validate and merge RAEv2 flow-guidance shards without importing GPU code."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PROTOCOL = "raev2_flow_pullback_sampling_v1"
MERGE_PROTOCOL = "raev2_flow_pullback_shard_merge_v1"
LOCAL_REQUEST_KEYS = {"output_dir", "shard_index", "local_sample_count"}
GEOMETRY_METADATA = {"index", "noise_time", "correction_active", "future_time"}
PAIRING_HASH_KEYS = (
    "noise_sha256", "labels_sha256", "initial_generator_sha256", "final_generator_sha256",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sum_costs(costs: list[dict]) -> dict:
    """Sum observed numeric counts recursively; retain identical unit notes."""

    result = {}
    for key in set().union(*(cost.keys() for cost in costs)):
        values = [cost.get(key) for cost in costs]
        present = [value for value in values if value is not None]
        if all(isinstance(value, dict) for value in present):
            result[key] = _sum_costs([value or {} for value in values])
        elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in present):
            _require(all(math.isfinite(value) and value >= 0 for value in present), f"invalid cost {key}")
            result[key] = sum(present)
        else:
            _require(all(value == present[0] for value in values), f"inconsistent cost units: {key}")
            result[key] = present[0]
    return result


def merge_shards(shard_dirs: list[Path], output_dir: Path, *, expected_protocol: str = PROTOCOL) -> dict:
    """Validate all inputs before writing a globally sorted image archive."""

    shard_dirs = [path.expanduser().resolve() for path in shard_dirs]
    output_dir = output_dir.expanduser().resolve()
    _require(bool(shard_dirs) and len(set(shard_dirs)) == len(shard_dirs), "shard directories must be nonempty and unique")
    _require(not output_dir.exists() or not any(output_dir.iterdir()), "merge output directory must be empty")
    records = []
    for directory in shard_dirs:
        request = json.loads((directory / "request.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        _require(request.get("protocol") == expected_protocol and summary.get("protocol") == expected_protocol, f"wrong protocol: {directory}")
        _require(summary.get("complete") is True, f"incomplete shard: {directory}")
        records.append((directory, request, summary))
    common = {key: value for key, value in records[0][1].items() if key not in LOCAL_REQUEST_KEYS}
    num_shards = int(common["num_shards"])
    global_count = int(common["global_sample_count"])
    batch_size = int(common["batch_size"])
    _require(num_shards > 0 and global_count > 0 and batch_size > 0, "invalid global sampling specification")
    _require(common.get("sample_count") == global_count, "sample-count must describe the full global bank")
    _require(len(records) == num_shards, f"expected all {num_shards} shard directories")
    shard_indices = [request["shard_index"] for _, request, _ in records]
    _require(sorted(shard_indices) == list(range(num_shards)), "missing or duplicated shard indices")
    records.sort(key=lambda record: record[1]["shard_index"])
    reference_hashes = {key: records[0][2][key] for key in PAIRING_HASH_KEYS}
    arrays, all_ids, geometries, weights = [], [], [], []
    image_shape = None
    for directory, request, summary in records:
        candidate_common = {key: value for key, value in request.items() if key not in LOCAL_REQUEST_KEYS}
        _require(candidate_common == common, f"sampling/source manifest mismatch: {directory}")
        shard_index = int(request["shard_index"])
        for key, expected in reference_hashes.items():
            _require(summary.get(key) == expected, f"full-bank pairing hash mismatch ({key}): {directory}")
        for key in ("global_sample_count", "num_shards", "shard_index", "mode", "seed", "batch_size", "world_size"):
            _require(summary.get(key) == request.get(key), f"request/summary mismatch ({key}): {directory}")
        archive_path = directory / "samples.npz"
        ids_path = directory / "sample_ids.npy"
        _require(file_sha256(archive_path) == summary.get("archive_sha256"), f"archive hash mismatch: {directory}")
        _require(file_sha256(ids_path) == summary.get("sample_ids_sha256"), f"sample-ID hash mismatch: {directory}")
        ids = np.load(ids_path, allow_pickle=False)
        _require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer), f"invalid sample IDs: {directory}")
        expected_ids = np.array([
            sample_id
            for batch_index, start in enumerate(range(0, global_count, batch_size))
            if batch_index % num_shards == shard_index
            for sample_id in range(start, min(start + batch_size, global_count))
        ], dtype=np.int64)
        _require(np.array_equal(ids, expected_ids), f"IDs do not match assigned original RNG batches: {directory}")
        with np.load(archive_path, allow_pickle=False) as archive:
            _require(archive.files == ["arr_0"], f"unexpected sample archive entries: {directory}")
            images = archive["arr_0"]
        _require(images.ndim == 4 and images.shape[-1] == 3 and images.dtype == np.uint8, f"samples must be uint8 NHWC RGB: {directory}")
        _require(len(ids) == len(images) == summary.get("samples") == summary.get("local_sample_count") == request.get("local_sample_count"), f"local sample count mismatch: {directory}")
        _require(len(ids) > 0, f"empty shard: {directory}")
        if image_shape is None:
            image_shape = images.shape[1:]
        _require(images.shape[1:] == image_shape, f"image dimensions differ: {directory}")
        with (directory / "geometry.csv").open(newline="") as stream:
            geometry = list(csv.DictReader(stream))
        _require(len(geometry) == int(common["num_steps"]), f"geometry step count mismatch: {directory}")
        _require([int(row["index"]) for row in geometry] == list(range(len(geometry))), f"geometry indices are incomplete: {directory}")
        for row in geometry:
            for key, value in row.items():
                if key not in GEOMETRY_METADATA and value != "":
                    _require(math.isfinite(float(value)), f"nonfinite geometry {key}: {directory}")
        arrays.append(images)
        all_ids.append(ids)
        geometries.append(geometry)
        weights.append(len(ids))
    ids = np.concatenate(all_ids)
    order = np.argsort(ids)
    _require(np.array_equal(ids[order], np.arange(global_count)), "shard IDs are not disjoint complete global coverage")
    merged_images = np.concatenate(arrays)[order]
    rows = []
    for step, first in enumerate(geometries[0]):
        _require(all(set(geometry[step]) == set(first) for geometry in geometries), f"geometry fields differ at step {step}")
        row = {}
        for key, value in first.items():
            values = [geometry[step][key] for geometry in geometries]
            if key in GEOMETRY_METADATA or value == "":
                _require(all(candidate == value for candidate in values), f"geometry metadata/missingness differs: step {step}, {key}")
                row[key] = value
            else:
                _require(all(candidate != "" for candidate in values), f"geometry metric missing: step {step}, {key}")
                row[key] = sum(float(candidate) * weight for candidate, weight in zip(values, weights)) / global_count
        rows.append(row)
    cost = _sum_costs([summary["model_cost"] for _, _, summary in records])
    merged_request = {
        "protocol": MERGE_PROTOCOL, "sampling_request_common": common,
        "shards": [{"directory": str(directory), "shard_index": request["shard_index"],
                    "local_sample_count": summary["local_sample_count"],
                    "request_sha256": file_sha256(directory / "request.json"),
                    "summary_sha256": file_sha256(directory / "summary.json")}
                   for directory, request, summary in records],
        "merge_source_sha256": file_sha256(Path(__file__)),
        "geometry_reduction": "sample-count-weighted means at each identical solver index",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "request.json").write_text(json.dumps(merged_request, indent=2) + "\n")
    archive_path = output_dir / "samples.npz"
    np.savez(archive_path, merged_images)
    np.save(output_dir / "sample_ids.npy", ids[order])
    with (output_dir / "geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol": MERGE_PROTOCOL, "sampling_protocol": expected_protocol, "complete": True,
        "mode": common["mode"], "samples": global_count, "global_sample_count": global_count,
        "seed": common["seed"], "batch_size": batch_size, "num_shards": num_shards,
        "local_sample_counts": weights, **reference_hashes,
        "archive_sha256": file_sha256(archive_path),
        "sample_ids_sha256": file_sha256(output_dir / "sample_ids.npy"),
        "model_cost": cost,
        "sum_shard_elapsed_seconds": sum(summary["elapsed_seconds"] for _, _, summary in records),
        "max_shard_elapsed_seconds": max(summary["elapsed_seconds"] for _, _, summary in records),
        "max_memory_allocated_bytes": max(summary["max_memory_allocated_bytes"] for _, _, summary in records),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-protocol", default=PROTOCOL)
    args = parser.parse_args()
    print(json.dumps(merge_shards(args.shard_dir, args.output_dir, expected_protocol=args.expected_protocol), indent=2))


if __name__ == "__main__":
    main()
