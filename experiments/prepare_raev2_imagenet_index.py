"""Build the official RAEv2 lexicographic ImageNet order from local parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.data_path.expanduser().resolve()
    data_dir = root / "data" if (root / "data").exists() else root
    files = sorted(data_dir.glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no train parquet shards under {data_dir}")

    records: list[tuple[str, int]] = []
    offset = 0
    shard_rows = []
    for path in files:
        table = pq.read_table(path, columns=["image.path"])
        paths = table.column("path").to_pylist()
        if any(value is None for value in paths):
            raise ValueError(f"{path} contains an image without a source filename")
        records.extend((value, offset + index) for index, value in enumerate(paths))
        shard_rows.append({"name": path.name, "rows": len(paths), "size": path.stat().st_size})
        offset += len(paths)

    records.sort(key=lambda item: item[0])
    ordered_paths = [path for path, _ in records]
    if len(set(ordered_paths)) != len(ordered_paths):
        raise ValueError("ImageNet source filenames are not unique")
    permutation = np.fromiter(
        (index for _, index in records),
        dtype=np.int64,
        count=len(records),
    )
    if (
        permutation.size != offset
        or int(permutation.min()) != 0
        or int(permutation.max()) != offset - 1
        or np.unique(permutation).size != offset
    ):
        raise RuntimeError("constructed index is not a complete permutation")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, permutation, allow_pickle=False)
    manifest = {
        "protocol": "raev2_imagenet_lexicographic_order_v1",
        "source": str(data_dir),
        "samples": int(offset),
        "shards": shard_rows,
        "first_paths": ordered_paths[:10],
        "last_paths": ordered_paths[-10:],
        "output": str(output),
        "output_sha256": file_sha256(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in ("samples", "output", "output_sha256")}))


if __name__ == "__main__":
    main()
