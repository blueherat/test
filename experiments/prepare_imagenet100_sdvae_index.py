"""Build a reproducible ImageNet-100 view over cached ImageNet SD-VAE latents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASS_LIST = REPO_ROOT / "experiments/configs/imagenet100_cmc_wnids.txt"
DEFAULT_PARQUET_ROOT = Path("/data/shared/imagenet-1k/data")
DEFAULT_OUTPUT_DIR = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc"
)
CMC_SOURCE_URL = "https://github.com/HobbitLong/CMC/blob/7b227be0b10ef4e526c72af07664f5079ed9ee09/imagenet100.txt"
CMC_SOURCE_COMMIT = "7b227be0b10ef4e526c72af07664f5079ed9ee09"


def read_wnids(path: Path) -> list[str]:
    wnids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    wnids = [wnid for wnid in wnids if wnid and not wnid.startswith("#")]
    if len(wnids) != 100:
        raise ValueError(f"expected 100 WordNet IDs in {path}, found {len(wnids)}")
    if len(set(wnids)) != len(wnids):
        raise ValueError(f"duplicate WordNet IDs in {path}")
    if any(len(wnid) != 9 or not wnid.startswith("n") or not wnid[1:].isdigit() for wnid in wnids):
        raise ValueError(f"invalid ImageNet WordNet ID in {path}")
    return wnids


def _parquet_files(root: Path, split: str) -> list[Path]:
    files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet shards under {root}")
    return files


def _class_names(first_train_file: Path) -> list[str]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(first_train_file).schema_arrow.metadata or {}
    huggingface = metadata.get(b"huggingface")
    if huggingface is None:
        raise ValueError(f"missing Hugging Face schema metadata in {first_train_file}")
    parsed = json.loads(huggingface)
    names = parsed["info"]["features"]["label"]["names"]
    if len(names) != 1000:
        raise ValueError(f"expected 1000 ImageNet class names, found {len(names)}")
    return names


def _wnid_from_hf_path(path: str) -> str:
    wnid = Path(path).name.split("_", 1)[0]
    if len(wnid) != 9 or not wnid.startswith("n") or not wnid[1:].isdigit():
        raise ValueError(f"cannot recover WordNet ID from ImageNet path: {path}")
    return wnid


def recover_label_to_wnid(train_files: Iterable[Path], num_classes: int) -> dict[int, str]:
    import pyarrow.parquet as pq

    mapping: dict[int, str] = {}
    reverse: dict[str, int] = {}
    for path in train_files:
        table = pq.ParquetFile(path).read(
            columns=["image.path", "label"], use_threads=False
        )
        image_column = table.column("image").combine_chunks().field("path")
        for image_path, label in zip(
            image_column.to_pylist(), table.column("label").to_pylist()
        ):
            label = int(label)
            wnid = _wnid_from_hf_path(image_path)
            if label in mapping and mapping[label] != wnid:
                raise ValueError(
                    f"label {label} maps to both {mapping[label]} and {wnid}"
                )
            if wnid in reverse and reverse[wnid] != label:
                raise ValueError(
                    f"WordNet ID {wnid} maps to both {reverse[wnid]} and {label}"
                )
            mapping[label] = wnid
            reverse[wnid] = label
        if len(mapping) == num_classes:
            break
    if set(mapping) != set(range(num_classes)):
        missing = sorted(set(range(num_classes)) - set(mapping))
        raise ValueError(f"failed to recover all ImageNet labels; missing {missing}")
    return mapping


def count_labels(files: Iterable[Path], num_classes: int) -> np.ndarray:
    import pyarrow.parquet as pq

    counts = np.zeros(num_classes, dtype=np.int64)
    for path in files:
        labels = pq.ParquetFile(path).read(
            columns=["label"], use_threads=False
        ).column("label").to_numpy(zero_copy_only=False)
        if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
            raise ValueError(f"out-of-range label in {path}")
        counts += np.bincount(labels, minlength=num_classes)
    return counts


def build_subset_indices(
    counts: np.ndarray,
    selected_original_labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64)]
    )
    index_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for new_label, original_label in enumerate(selected_original_labels):
        count = int(counts[original_label])
        start = int(starts[original_label])
        index_parts.append(np.arange(start, start + count, dtype=np.int64))
        label_parts.append(np.full(count, new_label, dtype=np.int64))
    indices = np.concatenate(index_parts)
    labels = np.concatenate(label_parts)
    if np.unique(indices).size != indices.size:
        raise ValueError("subset indices unexpectedly overlap")
    return indices, labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_array(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def build_index(parquet_root: Path, class_list: Path, output_dir: Path) -> dict:
    train_files = _parquet_files(parquet_root, "train")
    val_files = _parquet_files(parquet_root, "validation")
    class_names = _class_names(train_files[0])
    label_to_wnid = recover_label_to_wnid(train_files, len(class_names))
    wnid_to_label = {wnid: label for label, wnid in label_to_wnid.items()}

    source_wnids = read_wnids(class_list)
    sorted_wnids = sorted(source_wnids)
    missing = sorted(set(sorted_wnids) - set(wnid_to_label))
    if missing:
        raise ValueError(f"selected WordNet IDs are absent from ImageNet-1K: {missing}")

    train_counts = count_labels(train_files, len(class_names))
    val_counts = count_labels(val_files, len(class_names))
    selected_original_labels = [wnid_to_label[wnid] for wnid in sorted_wnids]
    train_indices, train_labels = build_subset_indices(
        train_counts, selected_original_labels
    )
    val_indices, val_labels = build_subset_indices(val_counts, selected_original_labels)

    if train_indices.size != 126_689:
        raise ValueError(
            f"CMC ImageNet-100 should contain 126689 train images, found {train_indices.size}"
        )
    if val_indices.size != 5_000:
        raise ValueError(
            f"CMC ImageNet-100 should contain 5000 validation images, found {val_indices.size}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    array_paths = {
        "train_indices": output_dir / "train_indices.npy",
        "train_labels": output_dir / "train_labels.npy",
        "validation_indices": output_dir / "validation_indices.npy",
        "validation_labels": output_dir / "validation_labels.npy",
    }
    _save_array(array_paths["train_indices"], train_indices)
    _save_array(array_paths["train_labels"], train_labels)
    _save_array(array_paths["validation_indices"], val_indices)
    _save_array(array_paths["validation_labels"], val_labels)

    classes = []
    for new_label, wnid in enumerate(sorted_wnids):
        original_label = wnid_to_label[wnid]
        classes.append(
            {
                "label": new_label,
                "wnid": wnid,
                "original_imagenet_label": original_label,
                "name": class_names[original_label],
                "train_count": int(train_counts[original_label]),
                "validation_count": int(val_counts[original_label]),
            }
        )

    manifest = {
        "format": "eqvae_imagenet100_cmc_index_v1",
        "source": {
            "class_list": str(class_list.resolve()),
            "class_list_sha256": _sha256(class_list),
            "cmc_url": CMC_SOURCE_URL,
            "cmc_commit": CMC_SOURCE_COMMIT,
            "imagenet_parquet_root": str(parquet_root.resolve()),
        },
        "ordering": {
            "latent_cache": "ImageFolder order: original class label, then source filename",
            "subset_labels": "lexicographically sorted selected WordNet IDs",
        },
        "splits": {
            "train": {"count": int(train_indices.size)},
            "validation": {"count": int(val_indices.size)},
        },
        "classes": classes,
        "files": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for key, path in array_paths.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CMC ImageNet-100 indices for the cached SD-VAE latent order."
    )
    parser.add_argument("--parquet-root", type=Path, default=DEFAULT_PARQUET_ROOT)
    parser.add_argument("--class-list", type=Path, default=DEFAULT_CLASS_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_index(
        args.parquet_root.expanduser().resolve(),
        args.class_list.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.expanduser().resolve()),
                "train_count": manifest["splits"]["train"]["count"],
                "validation_count": manifest["splits"]["validation"]["count"],
                "class_count": len(manifest["classes"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
