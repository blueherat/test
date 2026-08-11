from pathlib import Path
import json
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.prepare_imagenet100_sdvae_cache as cache_module


def _write_arrow_dataset(root: Path, values: np.ndarray) -> None:
    import pyarrow as pa

    root.mkdir(parents=True)
    path = root / "data-00000-of-00001.arrow"
    nested_type = pa.list_(pa.list_(pa.list_(pa.float32())))
    batch = pa.record_batch(
        [
            pa.array(np.arange(len(values)), type=pa.int32()),
            pa.array(values.tolist(), type=nested_type),
        ],
        names=("id", "data"),
    )
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
    (root / "state.json").write_text(
        json.dumps({"_data_files": [{"filename": path.name}]}),
        encoding="utf-8",
    )


def test_extract_split_preserves_requested_destination_order(
    tmp_path: Path, monkeypatch
) -> None:
    values = np.arange(6 * 8 * 32 * 32, dtype=np.float32).reshape(6, 8, 32, 32)
    source_root = tmp_path / "source"
    _write_arrow_dataset(source_root, values)
    monkeypatch.setitem(cache_module.EXPECTED_SOURCE_COUNTS, "train", 6)
    monkeypatch.setitem(cache_module.EXPECTED_SUBSET_COUNTS, "train", 3)

    result = cache_module.extract_split(
        source_root=source_root,
        indices=np.asarray([4, 1, 5], dtype=np.int64),
        labels=np.asarray([0, 1, 2], dtype=np.int64),
        output_dir=tmp_path / "output",
        split="train",
        hash_moments=False,
    )

    extracted = np.load(result["moments_path"], allow_pickle=False)
    labels = np.load(result["labels_path"], allow_pickle=False)
    assert np.array_equal(extracted, values[[4, 1, 5]])
    assert labels.tolist() == [0, 1, 2]
    assert result["moments_sha256"] is None


def test_arrow_moments_zero_copy_shape(tmp_path: Path) -> None:
    import pyarrow as pa

    values = np.random.default_rng(4).normal(size=(2, 8, 32, 32)).astype(np.float32)
    column = pa.array(
        values.tolist(), type=pa.list_(pa.list_(pa.list_(pa.float32())))
    )
    recovered = cache_module.arrow_moments_to_numpy(column, len(values))
    assert recovered.shape == values.shape
    assert np.array_equal(recovered, values)


def test_partial_prepare_preserves_existing_split_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    index = tmp_path / "index"
    output = tmp_path / "output"
    source.mkdir()
    index.mkdir()
    index_manifest = index / "manifest.json"
    index_manifest.write_text(
        json.dumps({"format": "eqvae_imagenet100_cmc_index_v1"}),
        encoding="utf-8",
    )
    old = {
        "format": "eqvae_imagenet100_cmc_sdvae_moments_v1",
        "source": {
            "latent_root": str(source.resolve()),
            "index_manifest_sha256": cache_module.sha256_file(index_manifest),
        },
        "splits": {"train": {"count": 3}},
    }
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(old), encoding="utf-8")
    np.save(index / "validation_indices.npy", np.asarray([0], dtype=np.int64))
    np.save(index / "validation_labels.npy", np.asarray([0], dtype=np.int64))
    monkeypatch.setattr(
        cache_module,
        "extract_split",
        lambda **kwargs: {"count": 1},
    )

    result = cache_module.prepare_cache(
        source_root=source,
        index_dir=index,
        output_dir=output,
        splits=("validation",),
        hash_moments=False,
    )

    assert result["splits"] == {"train": {"count": 3}, "validation": {"count": 1}}
