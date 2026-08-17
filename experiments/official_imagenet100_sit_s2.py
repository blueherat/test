"""Utilities for the official ImageNet-1K SiT-S/2 checkpoint.

The public checkpoint has 1,000 class embeddings, while the local controlled
experiments use the CMC ImageNet-100 subset with labels in ``[0, 99]``. The
helpers below extract the corresponding official embeddings without changing
the backbone or output field for those classes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch


HF_REPOSITORY = "nyu-visionx/SiT-collections"
HF_REVISION = "9a43ddf08f80ac7f3525208ce72943e98fe98343"
HF_FILENAME = "SiT-S-2-256.pt"
RAW_CHECKPOINT_BYTES = 131_893_962
RAW_CHECKPOINT_SHA256 = (
    "a245dc6330cd0d5906a5da00718b7d348a417d740e6b6cfeeb504e9d1448d070"
)
PRETRAINED_SOURCE_FORMAT = "eqvae_official_sit_pretrained_source_v1"
SUBSET_CHECKPOINT_FORMAT = "eqvae_official_sit_s2_imagenet100_subset_v1"
CLASS_EMBEDDING_KEY = "y_embedder.embedding_table.weight"
OFFICIAL_CLASS_COUNT = 1_000
SUBSET_CLASS_COUNT = 100

DEFAULT_RAW_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/models/SiT-official/SiT-S-2-256.pt"
)
DEFAULT_INDEX_MANIFEST = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc/manifest.json"
)
DEFAULT_CACHE_MANIFEST = Path(
    "/home/zhoushunyu/data/eqvae/imagenet_sit_flow/imagenet100_cmc_sdvae/manifest.json"
)
DEFAULT_SUBSET_CHECKPOINT = Path(
    "/home/zhoushunyu/data/eqvae/models/SiT-official/"
    "SiT-S-2-256-imagenet100-subset.pt"
)


def load_imagenet100_class_mapping(manifest_path: Path) -> list[dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "eqvae_imagenet100_cmc_index_v1":
        raise ValueError(f"unsupported ImageNet-100 manifest: {manifest_path}")
    classes = payload.get("classes")
    if not isinstance(classes, list) or len(classes) != SUBSET_CLASS_COUNT:
        raise ValueError("ImageNet-100 manifest must contain exactly 100 classes")
    ordered = sorted(classes, key=lambda row: int(row["label"]))
    subset_labels = [int(row["label"]) for row in ordered]
    original_labels = [int(row["original_imagenet_label"]) for row in ordered]
    if subset_labels != list(range(SUBSET_CLASS_COUNT)):
        raise ValueError("ImageNet-100 subset labels are not contiguous")
    if len(set(original_labels)) != SUBSET_CLASS_COUNT:
        raise ValueError("ImageNet-100 original labels are not unique")
    if min(original_labels) < 0 or max(original_labels) >= OFFICIAL_CLASS_COUNT:
        raise ValueError("ImageNet-100 original labels fall outside ImageNet-1K")
    return ordered


def subset_original_labels(classes: list[dict[str, object]]) -> list[int]:
    return [int(row["original_imagenet_label"]) for row in classes]


def subset_official_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    original_labels: list[int],
) -> dict[str, torch.Tensor]:
    if len(original_labels) != SUBSET_CLASS_COUNT:
        raise ValueError("expected 100 original ImageNet labels")
    if len(set(original_labels)) != SUBSET_CLASS_COUNT:
        raise ValueError("original ImageNet labels must be unique")
    if min(original_labels) < 0 or max(original_labels) >= OFFICIAL_CLASS_COUNT:
        raise ValueError("original ImageNet labels fall outside ImageNet-1K")
    if CLASS_EMBEDDING_KEY not in state_dict:
        raise KeyError(f"official checkpoint lacks {CLASS_EMBEDDING_KEY!r}")
    embeddings = state_dict[CLASS_EMBEDDING_KEY]
    if embeddings.ndim != 2 or embeddings.shape[0] != OFFICIAL_CLASS_COUNT + 1:
        raise ValueError(
            "official class embedding must have 1,001 rows, including CFG"
        )
    indices = torch.tensor(
        [*original_labels, OFFICIAL_CLASS_COUNT],
        dtype=torch.long,
        device=embeddings.device,
    )
    subset = dict(state_dict)
    subset[CLASS_EMBEDDING_KEY] = embeddings.index_select(0, indices).clone()
    return subset


def is_verified_pretrained_source(payload: Mapping[str, object]) -> bool:
    source = payload.get("pretrained_source")
    return (
        isinstance(source, Mapping)
        and source.get("format") == PRETRAINED_SOURCE_FORMAT
        and source.get("repository") == HF_REPOSITORY
        and source.get("revision") == HF_REVISION
        and source.get("filename") == HF_FILENAME
        and source.get("raw_sha256") == RAW_CHECKPOINT_SHA256
    )


def source_step(payload: Mapping[str, object]) -> int:
    """Return a source step without inventing one for published final weights."""

    step = int(payload.get("step", -1))
    if step >= 1:
        return step
    if step == 0 and is_verified_pretrained_source(payload):
        return step
    raise ValueError(
        "source checkpoint has neither a valid training step nor verified "
        "official-pretrained provenance"
    )
