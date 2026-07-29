"""Build a lossless random-access store from Hugging Face ImageNet parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKED_FORMAT = "eqvae_imagenet_packed_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _valid_existing_record(
    source_path: Path,
    destination: Path,
) -> dict[str, Any] | None:
    stem = source_path.stem
    metadata_path = destination / f"{stem}.json"
    if not metadata_path.is_file():
        return None
    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        data_path = destination.parent / record["data_file"]
        offsets_path = destination.parent / record["offsets_file"]
        labels_path = destination.parent / record["labels_file"]
        offsets = np.load(offsets_path, allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False)
        valid = (
            int(record["source_size"]) == source_path.stat().st_size
            and int(record["source_mtime_ns"]) == source_path.stat().st_mtime_ns
            and data_path.stat().st_size == int(record["data_bytes"])
            and offsets.shape == (int(record["rows"]) + 1,)
            and labels.shape == (int(record["rows"]),)
            and int(offsets[-1]) == int(record["data_bytes"])
        )
        return record if valid else None
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def pack_shard(source_path_text: str, destination_text: str) -> dict[str, Any]:
    import pyarrow.parquet as pq

    source_path = Path(source_path_text)
    destination = Path(destination_text)
    destination.mkdir(parents=True, exist_ok=True)
    existing = _valid_existing_record(source_path, destination)
    if existing is not None:
        return existing

    stem = source_path.stem
    data_path = destination / f"{stem}.bin"
    offsets_path = destination / f"{stem}.offsets.npy"
    labels_path = destination / f"{stem}.labels.npy"
    metadata_path = destination / f"{stem}.json"
    temporary_data = data_path.with_name(f"{data_path.name}.tmp")

    parquet = pq.ParquetFile(source_path)
    offsets = [0]
    labels: list[int] = []
    content_digest = hashlib.sha256()
    with temporary_data.open("wb") as output:
        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(
                row_group,
                columns=["image", "label"],
            )
            image_rows = table.column("image").to_pylist()
            label_rows = table.column("label").to_pylist()
            for image_info, label_value in zip(image_rows, label_rows):
                image_bytes = image_info.get("bytes")
                if image_bytes is None:
                    image_path = image_info.get("path")
                    if image_path is None:
                        raise ValueError(
                            f"{source_path} contains an image without bytes or path"
                        )
                    image_bytes = (source_path.parent / image_path).read_bytes()
                image_bytes = bytes(image_bytes)
                label = int(label_value)
                output.write(image_bytes)
                offsets.append(offsets[-1] + len(image_bytes))
                labels.append(label)
                content_digest.update(len(image_bytes).to_bytes(8, "little"))
                content_digest.update(image_bytes)
                content_digest.update(label.to_bytes(8, "little", signed=True))
    os.replace(temporary_data, data_path)

    offsets_array = np.asarray(offsets, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int32)
    _write_npy_atomic(offsets_path, offsets_array)
    _write_npy_atomic(labels_path, labels_array)
    record = {
        "source_file": source_path.name,
        "source_size": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "data_file": str(data_path.relative_to(destination.parent)),
        "offsets_file": str(offsets_path.relative_to(destination.parent)),
        "labels_file": str(labels_path.relative_to(destination.parent)),
        "rows": len(labels),
        "data_bytes": data_path.stat().st_size,
        "content_sha256": content_digest.hexdigest(),
    }
    _write_json_atomic(metadata_path, record)
    return record


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    data_dir = source / "data" if (source / "data").is_dir() else source
    files = sorted(data_dir.glob(f"{args.split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {args.split} parquet shards under {data_dir}")

    shard_destination = destination / args.split
    shard_destination.mkdir(parents=True, exist_ok=True)
    records_by_name: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                pack_shard,
                str(path),
                str(shard_destination),
            ): path
            for path in files
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            record = future.result()
            records_by_name[path.name] = record
            print(
                f"[{completed}/{len(files)}] {path.name}: "
                f"{record['rows']} rows, {record['data_bytes'] / 2**20:.1f} MiB",
                flush=True,
            )

    records = [records_by_name[path.name] for path in files]
    manifest = {
        "format": PACKED_FORMAT,
        "version": 1,
        "split": args.split,
        "source_root": str(data_dir),
        "built_at": datetime.now().astimezone().isoformat(),
        "total_rows": sum(int(record["rows"]) for record in records),
        "total_data_bytes": sum(int(record["data_bytes"]) for record in records),
        "shards": records,
    }
    _write_json_atomic(destination / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(destination / "manifest.json"),
                "rows": manifest["total_rows"],
                "data_gib": manifest["total_data_bytes"] / 2**30,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
