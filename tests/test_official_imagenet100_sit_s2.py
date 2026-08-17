from __future__ import annotations

import json

import pytest
import torch

from experiments.official_imagenet100_sit_s2 import (
    CLASS_EMBEDDING_KEY,
    HF_FILENAME,
    HF_REPOSITORY,
    HF_REVISION,
    PRETRAINED_SOURCE_FORMAT,
    RAW_CHECKPOINT_SHA256,
    load_imagenet100_class_mapping,
    source_step,
    subset_official_state_dict,
)


def verified_payload() -> dict[str, object]:
    return {
        "step": 0,
        "pretrained_source": {
            "format": PRETRAINED_SOURCE_FORMAT,
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "filename": HF_FILENAME,
            "raw_sha256": RAW_CHECKPOINT_SHA256,
        },
    }


def test_manifest_mapping_is_sorted_and_validated(tmp_path) -> None:
    classes = [
        {
            "label": index,
            "original_imagenet_label": 999 - index,
            "class_name": f"class-{index}",
        }
        for index in reversed(range(100))
    ]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"format": "eqvae_imagenet100_cmc_index_v1", "classes": classes}),
        encoding="utf-8",
    )

    ordered = load_imagenet100_class_mapping(path)

    assert [row["label"] for row in ordered] == list(range(100))
    assert [row["original_imagenet_label"] for row in ordered] == list(
        range(999, 899, -1)
    )


def test_subset_state_selects_classes_and_unconditional_row_exactly() -> None:
    embeddings = torch.arange(1001 * 3, dtype=torch.float32).reshape(1001, 3)
    untouched = torch.tensor([17.0])
    state = {CLASS_EMBEDDING_KEY: embeddings, "other": untouched}
    labels = list(range(100, 200))

    subset = subset_official_state_dict(state, labels)

    torch.testing.assert_close(subset[CLASS_EMBEDDING_KEY][:-1], embeddings[100:200])
    torch.testing.assert_close(subset[CLASS_EMBEDDING_KEY][-1], embeddings[1000])
    assert subset["other"] is untouched
    assert subset[CLASS_EMBEDDING_KEY].data_ptr() != embeddings.data_ptr()


@pytest.mark.parametrize(
    "labels, message",
    [
        ([0] * 100, "unique"),
        ([*range(99), 1000], "outside"),
    ],
)
def test_subset_state_rejects_invalid_original_labels(labels, message) -> None:
    state = {CLASS_EMBEDDING_KEY: torch.zeros(1001, 2)}
    with pytest.raises(ValueError, match=message):
        subset_official_state_dict(state, labels)


def test_source_step_accepts_regular_and_verified_published_weights() -> None:
    assert source_step({"step": 800_000}) == 800_000
    assert source_step(verified_payload()) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"step": 0},
        {"step": -1},
        {**verified_payload(), "pretrained_source": {"format": "forged"}},
    ],
)
def test_source_step_rejects_unverified_zero_or_invalid_steps(payload) -> None:
    with pytest.raises(ValueError, match="verified official-pretrained provenance"):
        source_step(payload)
