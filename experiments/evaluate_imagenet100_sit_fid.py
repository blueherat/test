"""Build an ImageNet-100 ADM reference NPZ for official-style FID screening."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from sample_imagenet100_sit_fid import DEFAULT_OUTPUT_DIR
from train_imagenet100_sit_flow import atomic_json_dump, sha256_file


DEFAULT_PARQUET_ROOT = Path("/data/shared/imagenet-1k/data")
DEFAULT_INDEX_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc"
)
IMAGE_SIZE = 256


def center_crop_arr(image: Image.Image, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """ADM center crop copied from the iREPA preprocessing implementation."""

    image = image.convert("RGB")
    while min(image.size) >= 2 * image_size:
        image = image.resize(
            tuple(size // 2 for size in image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(image.size)
    image = image.resize(
        tuple(round(size * scale) for size in image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.uint8)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    cropped = array[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    if cropped.shape != (image_size, image_size, 3):
        raise ValueError(f"unexpected crop shape: {cropped.shape}")
    return cropped


def build_reference_npz(
    *, parquet_root: Path, index_dir: Path, output_path: Path
) -> dict:
    index_manifest_path = index_dir / "manifest.json"
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    classes = index_manifest["classes"]
    original_to_subset = {
        int(record["original_imagenet_label"]): int(record["label"])
        for record in classes
    }
    selected_labels = set(original_to_subset)
    validation_files = sorted(parquet_root.glob("validation-*.parquet"))
    if not validation_files:
        raise FileNotFoundError(f"no validation parquet shards under {parquet_root}")

    images = np.empty((5_000, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty(5_000, dtype=np.int16)
    source_paths: list[str] = []
    cursor = 0

    import pyarrow.parquet as pq

    for parquet_path in validation_files:
        table = pq.ParquetFile(parquet_path).read(
            columns=["image", "label"], use_threads=True
        )
        source_labels = table.column("label").to_numpy(zero_copy_only=False)
        image_column = table.column("image").combine_chunks()
        image_bytes = image_column.field("bytes").to_pylist()
        image_paths = image_column.field("path").to_pylist()
        for row_index, original_label in enumerate(source_labels):
            original_label = int(original_label)
            if original_label not in selected_labels:
                continue
            if cursor >= len(images):
                raise ValueError("selected more than 5,000 validation images")
            with Image.open(io.BytesIO(image_bytes[row_index])) as image:
                images[cursor] = center_crop_arr(image)
            labels[cursor] = original_to_subset[original_label]
            source_paths.append(str(image_paths[row_index]))
            cursor += 1
        print(
            json.dumps(
                {
                    "event": "reference_progress",
                    "shard": parquet_path.name,
                    "selected": cursor,
                }
            ),
            flush=True,
        )

    if cursor != 5_000:
        raise ValueError(f"expected 5,000 reference images, found {cursor}")
    histogram = np.bincount(labels.astype(np.int64), minlength=100)
    if not np.array_equal(histogram, np.full(100, 50, dtype=np.int64)):
        raise ValueError(f"unexpected ImageNet validation class counts: {histogram.tolist()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("wb") as handle:
        np.savez(handle, arr_0=images)
    os.replace(temporary, output_path)
    labels_path = output_path.with_name(output_path.stem + "_labels.npy")
    np.save(labels_path, labels, allow_pickle=False)
    result = {
        "format": "eqvae_imagenet100_cmc_adm_reference_v1",
        "scope": "5k ImageNet-100 validation reference; not the official ImageNet-1K reference",
        "reference_npz": str(output_path),
        "reference_sha256": sha256_file(output_path),
        "labels": str(labels_path),
        "count": 5_000,
        "class_count": 100,
        "samples_per_class": 50,
        "preprocessing": "iREPA/ADM center_crop_arr at 256x256",
        "parquet_root": str(parquet_root),
        "validation_shards": [str(path) for path in validation_files],
        "index_manifest": str(index_manifest_path),
        "index_manifest_sha256": sha256_file(index_manifest_path),
        "first_source_paths": source_paths[:16],
    }
    atomic_json_dump(result, output_path.with_name("reference_manifest.json"))
    return result


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_path = output_dir / "reference_imagenet100_validation_n5000.npz"
    if reference_path.is_file() and not args.force:
        print(json.dumps({"event": "reuse_reference", "path": str(reference_path)}))
        return
    result = build_reference_npz(
        parquet_root=Path(args.parquet_root).expanduser().resolve(),
        index_dir=Path(args.index_dir).expanduser().resolve(),
        output_path=reference_path,
    )
    print(json.dumps(result, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
