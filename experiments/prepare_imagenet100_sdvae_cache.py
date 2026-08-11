"""Extract the CMC ImageNet-100 SD-VAE moments into contiguous NumPy files.

The source cache is the public iREPA ImageNet SD-VAE cache.  Its Arrow rows
follow ImageFolder order and contain posterior mean/std tensors with shape
``[8, 32, 32]``.  This script applies the audited ImageNet-100 index and writes
an exact float32 subset that can be memory-mapped during SiT training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np


DEFAULT_SOURCE_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/irepa_collections/data/"
    "imagenet-latents-sdvae-ft-mse-f8d4"
)
DEFAULT_INDEX_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae"
)
MOMENT_SHAPE = (8, 32, 32)
EXPECTED_SOURCE_COUNTS = {"train": 1_281_167, "validation": 50_000}
EXPECTED_SUBSET_COUNTS = {"train": 126_689, "validation": 5_000}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_split_root(source_root: Path, split: str) -> Path:
    if split == "train":
        return source_root
    if split == "validation":
        return source_root / "val"
    raise ValueError(f"unsupported split: {split}")


def source_shards(root: Path) -> list[Path]:
    state_path = root / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing Hugging Face dataset state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    filenames = [entry["filename"] for entry in state.get("_data_files", [])]
    if not filenames:
        raise ValueError(f"no Arrow shards listed in {state_path}")
    paths = [root / filename for filename in filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        preview = "\n  ".join(missing[:8])
        raise FileNotFoundError(
            f"the latent download is incomplete; missing {len(missing)} shards:\n  {preview}"
        )
    return paths


def arrow_record_batches(paths: list[Path]) -> Iterator[tuple[Path, object]]:
    import pyarrow as pa

    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_stream(source)
            for batch in reader:
                yield path, batch


def arrow_moments_to_numpy(column: object, row_count: int) -> np.ndarray:
    """Return an exact zero-copy view over one Arrow record batch."""
    values = column
    for _ in range(3):
        if not hasattr(values, "values"):
            raise ValueError("unexpected Arrow latent nesting")
        values = values.values
    flat = values.to_numpy(zero_copy_only=True)
    expected = int(row_count * np.prod(MOMENT_SHAPE))
    if flat.dtype != np.float32 or flat.size != expected:
        raise ValueError(
            f"unexpected Arrow latent storage: dtype={flat.dtype}, size={flat.size}, "
            f"expected float32/{expected}"
        )
    return flat.reshape(row_count, *MOMENT_SHAPE)


def _atomic_save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def extract_split(
    *,
    source_root: Path,
    indices: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    split: str,
    hash_moments: bool,
) -> dict:
    if indices.ndim != 1 or labels.ndim != 1 or len(indices) != len(labels):
        raise ValueError("indices and labels must be equally sized one-dimensional arrays")
    if indices.dtype.kind not in "iu" or labels.dtype.kind not in "iu":
        raise ValueError("indices and labels must be integer arrays")
    if len(indices) != EXPECTED_SUBSET_COUNTS[split]:
        raise ValueError(
            f"unexpected {split} subset size {len(indices)}; "
            f"expected {EXPECTED_SUBSET_COUNTS[split]}"
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"duplicate source indices in {split}")
    if labels.min(initial=0) < 0 or labels.max(initial=0) >= 100:
        raise ValueError(f"out-of-range ImageNet-100 labels in {split}")

    split_root = source_split_root(source_root, split)
    shards = source_shards(split_root)
    order = np.argsort(indices, kind="stable")
    sorted_indices = indices[order].astype(np.int64, copy=False)
    if sorted_indices[0] < 0 or sorted_indices[-1] >= EXPECTED_SOURCE_COUNTS[split]:
        raise ValueError(f"source indices fall outside the {split} cache")

    output_dir.mkdir(parents=True, exist_ok=True)
    moments_path = output_dir / f"{split}_moments.npy"
    labels_path = output_dir / f"{split}_labels.npy"
    source_indices_path = output_dir / f"{split}_source_indices.npy"
    temporary_moments = moments_path.with_suffix(".npy.tmp")
    temporary_moments.unlink(missing_ok=True)
    moments = np.lib.format.open_memmap(
        temporary_moments,
        mode="w+",
        dtype=np.float32,
        shape=(len(indices), *MOMENT_SHAPE),
    )

    channel_sum = np.zeros(MOMENT_SHAPE[0], dtype=np.float64)
    channel_square_sum = np.zeros(MOMENT_SHAPE[0], dtype=np.float64)
    value_min = np.full(MOMENT_SHAPE[0], np.inf, dtype=np.float64)
    value_max = np.full(MOMENT_SHAPE[0], -np.inf, dtype=np.float64)
    source_cursor = 0
    written = 0

    for shard_index, (path, batch) in enumerate(arrow_record_batches(shards), start=1):
        row_count = int(batch.num_rows)
        ids = batch.column("id").to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        expected_ids = np.arange(source_cursor, source_cursor + row_count, dtype=np.int64)
        if not np.array_equal(ids, expected_ids):
            raise ValueError(
                f"latent row IDs are not contiguous at {path}; expected "
                f"{source_cursor}..{source_cursor + row_count - 1}"
            )

        end = source_cursor + row_count
        left = int(np.searchsorted(sorted_indices, source_cursor, side="left"))
        right = int(np.searchsorted(sorted_indices, end, side="left"))
        if right > left:
            selected_source = sorted_indices[left:right]
            source_positions = selected_source - source_cursor
            destination_positions = order[left:right]
            batch_moments = arrow_moments_to_numpy(batch.column("data"), row_count)
            selected = batch_moments[source_positions]
            if not np.isfinite(selected).all():
                raise ValueError(f"non-finite latent moments in {path}")
            if (selected[:, 4:] < 0).any():
                raise ValueError(f"negative posterior standard deviation in {path}")
            moments[destination_positions] = selected

            reduced_axes = (0, 2, 3)
            channel_sum += selected.sum(axis=reduced_axes, dtype=np.float64)
            channel_square_sum += np.square(selected, dtype=np.float64).sum(
                axis=reduced_axes, dtype=np.float64
            )
            value_min = np.minimum(value_min, selected.min(axis=reduced_axes))
            value_max = np.maximum(value_max, selected.max(axis=reduced_axes))
            written += right - left

        source_cursor = end
        if shard_index % 100 == 0:
            print(
                f"[{split}] source_rows={source_cursor:,} selected={written:,}/{len(indices):,}",
                flush=True,
            )

    if source_cursor != EXPECTED_SOURCE_COUNTS[split]:
        raise ValueError(
            f"unexpected {split} source count {source_cursor}; "
            f"expected {EXPECTED_SOURCE_COUNTS[split]}"
        )
    if written != len(indices):
        raise ValueError(f"wrote {written} of {len(indices)} selected {split} rows")

    moments.flush()
    del moments
    os.replace(temporary_moments, moments_path)
    _atomic_save_array(labels_path, labels.astype(np.int16, copy=False))
    _atomic_save_array(source_indices_path, indices.astype(np.int32, copy=False))

    scalar_count = len(indices) * MOMENT_SHAPE[1] * MOMENT_SHAPE[2]
    channel_mean = channel_sum / scalar_count
    channel_variance = channel_square_sum / scalar_count - np.square(channel_mean)
    result = {
        "count": int(len(indices)),
        "shape": [int(len(indices)), *MOMENT_SHAPE],
        "dtype": "float32",
        "source_count": int(source_cursor),
        "source_shards": len(shards),
        "moments_path": str(moments_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "source_indices_path": str(source_indices_path.resolve()),
        "moments_bytes": moments_path.stat().st_size,
        "moments_sha256": sha256_file(moments_path) if hash_moments else None,
        "labels_sha256": sha256_file(labels_path),
        "source_indices_sha256": sha256_file(source_indices_path),
        "channel_mean": channel_mean.tolist(),
        "channel_std": np.sqrt(np.maximum(channel_variance, 0.0)).tolist(),
        "channel_min": value_min.tolist(),
        "channel_max": value_max.tolist(),
    }
    print(f"[{split}] complete: {len(indices):,} rows -> {moments_path}", flush=True)
    return result


def prepare_cache(
    *,
    source_root: Path,
    index_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...],
    hash_moments: bool,
) -> dict:
    index_manifest_path = index_dir / "manifest.json"
    if not index_manifest_path.is_file():
        raise FileNotFoundError(f"missing audited subset manifest: {index_manifest_path}")
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    if index_manifest.get("format") != "eqvae_imagenet100_cmc_index_v1":
        raise ValueError(f"unsupported subset manifest: {index_manifest_path}")

    manifest_path = output_dir / "manifest.json"
    split_results = {}
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("format") != "eqvae_imagenet100_cmc_sdvae_moments_v1":
            raise ValueError(f"refusing to overwrite an unrelated manifest: {manifest_path}")
        existing_source = existing.get("source", {})
        if (
            existing_source.get("latent_root") != str(source_root.resolve())
            or existing_source.get("index_manifest_sha256")
            != sha256_file(index_manifest_path)
        ):
            raise ValueError(f"existing cache uses a different source: {manifest_path}")
        split_results.update(existing.get("splits", {}))
    for split in splits:
        indices = np.load(index_dir / f"{split}_indices.npy", allow_pickle=False)
        labels = np.load(index_dir / f"{split}_labels.npy", allow_pickle=False)
        split_results[split] = extract_split(
            source_root=source_root,
            indices=indices,
            labels=labels,
            output_dir=output_dir,
            split=split,
            hash_moments=hash_moments,
        )

    manifest = {
        "format": "eqvae_imagenet100_cmc_sdvae_moments_v1",
        "source": {
            "latent_root": str(source_root.resolve()),
            "index_manifest": str(index_manifest_path.resolve()),
            "index_manifest_sha256": sha256_file(index_manifest_path),
            "posterior_layout": "channels 0:4 mean, channels 4:8 standard deviation",
            "vae": "stabilityai/sd-vae-ft-mse",
            "vae_scaling_factor": 0.18215,
        },
        "preprocessing": {
            "resolution": 256,
            "crop": "source cache center crop",
            "horizontal_flip": False,
            "note": "The public cache has one deterministic latent per image.",
        },
        "splits": split_results,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an exact contiguous ImageNet-100 SD-VAE moment cache."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated subset of train,validation.",
    )
    parser.add_argument(
        "--skip-moments-hash",
        action="store_true",
        help="Skip the final sequential SHA256 pass over each large moments file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = tuple(value.strip() for value in args.splits.split(",") if value.strip())
    if not splits or any(split not in EXPECTED_SOURCE_COUNTS for split in splits):
        raise ValueError("--splits must contain train and/or validation")
    manifest = prepare_cache(
        source_root=args.source_root.expanduser().resolve(),
        index_dir=args.index_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        splits=splits,
        hash_moments=not args.skip_moments_hash,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "splits": {
                    split: values["count"] for split, values in manifest["splits"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
